from dataclasses import dataclass
import os
from typing import List, Tuple

import cv2
from numba import njit
import numpy as np
from PIL import Image
import streamlit as st
import torch
from transformers import pipeline

# ============================================================
# AI CNC WOOD RELIEF GENERATOR v4 - MULTI-LAYER & ATC EDITION
# ============================================================

st.set_page_config(
    page_title="AI CNC Relief Generator v4 (Multi-Tool CAM)",
    page_icon="🪵",
    layout="wide",
)

# ============================================================
# 1. NUMBA OPTIMIZED CORE (TỐI ƯU TỐC ĐỘ G-CODE & NÉN RDP)
# ============================================================


@njit
def rdp_simplify_1d(points: np.ndarray, epsilon: float) -> np.ndarray:
  """Thuật toán Ramer-Douglas-Peucker 1D nén bớt các điểm nằm trên đường thẳng."""
  n = len(points)
  if n <= 2:
    return np.ones(n, dtype=np.bool_)

  keep = np.ones(n, dtype=np.bool_)
  stack = [(0, n - 1)]

  while len(stack) > 0:
    start, end = stack.pop()
    if end - start <= 1:
      continue

    max_dist = 0.0
    index = start

    x1, z1 = start, points[start]
    x2, z2 = end, points[end]

    dx = x2 - x1
    dz = z2 - z1
    denom = np.sqrt(dx * dx + dz * dz)

    for i in range(start + 1, end):
      if denom == 0:
        dist = np.abs(points[i] - z1)
      else:
        dist = (
            np.abs(dz * i - dx * points[i] + x2 * z1 - z2 * x1) / denom
        )

      if dist > max_dist:
        max_dist = dist
        index = i

    if max_dist > epsilon:
      stack.append((start, index))
      stack.append((index, end))
    else:
      for i in range(start + 1, end):
        keep[i] = False

  return keep


@njit
def compensate_ballnose_kernel(
    depth_mm: np.ndarray, radius_px: int, res_x: float
) -> np.ndarray:
  """Bù bán kính dao cầu (Heightfield / Z-buffer Offset)."""
  h, w = depth_mm.shape
  compensated = np.copy(depth_mm)

  if radius_px < 1:
    return compensated

  for y in range(h):
    for x in range(w):
      max_z = -9999.0
      for dy in range(-radius_px, radius_px + 1):
        ny = y + dy
        if ny < 0 or ny >= h:
          continue
        for dx in range(-radius_px, radius_px + 1):
          nx = x + dx
          if nx < 0 or nx >= w:
            continue

          dist_sq = (dx * dx + dy * dy) * (res_x * res_x)
          r_sq = (radius_px * res_x) ** 2

          if dist_sq <= r_sq:
            sphere_offset = np.sqrt(max(0.0, r_sq - dist_sq)) - (
                radius_px * res_x
            )
            z_val = depth_mm[ny, nx] + sphere_offset
            if z_val > max_z:
              max_z = z_val
      compensated[y, x] = max_z

  return compensated


# ============================================================
# 2. DATA STRUCTURES & LOAD AI MODEL
# ============================================================


@dataclass
class ToolConfig:
  tool_id: int
  name: str
  tool_type: str  # 'Flat', 'Ball', 'Tapered'
  diameter: float
  stepover_pct: float
  pass_depth: float
  feed_rate: float
  plunge_rate: float
  spindle_speed: int


@st.cache_resource
def load_depth_ai(model_type: str):
  device = 0 if torch.cuda.is_available() else -1
  model_id = (
      "depth-anything/Depth-Anything-V2-Large-hf"
      if model_type == "Large (Siêu nét)"
      else "depth-anything/Depth-Anything-V2-Small-hf"
  )
  pipe = pipeline(task="depth-estimation", model=model_id, device=device)
  return pipe


# ============================================================
# 3. ADVANCED PROCESSING
# ============================================================


def process_depth_tiled(
    image_rgb: np.ndarray,
    model_type: str,
    tile_size: int = 1024,
    overlap: int = 128,
) -> np.ndarray:
  pipe = load_depth_ai(model_type)
  h, w, _ = image_rgb.shape

  if h <= tile_size and w <= tile_size:
    res = pipe(Image.fromarray(image_rgb))
    depth = np.array(res["depth"], dtype=np.float32)
    return cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)

  depth_accum = np.zeros((h, w), dtype=np.float32)
  weight_accum = np.zeros((h, w), dtype=np.float32)
  stride = tile_size - overlap

  for y in range(0, h, stride):
    for x in range(0, w, stride):
      y_end = min(y + tile_size, h)
      x_end = min(x + tile_size, w)
      y_start = max(0, y_end - tile_size)
      x_start = max(0, x_end - tile_size)

      tile = image_rgb[y_start:y_end, x_start:x_end]
      res = pipe(Image.fromarray(tile))
      tile_depth = np.array(res["depth"], dtype=np.float32)

      ty, tx = tile_depth.shape
      mask_y = np.sin(np.linspace(0, np.pi, ty)) ** 2
      mask_x = np.sin(np.linspace(0, np.pi, tx)) ** 2
      weight = np.outer(mask_y, mask_x)

      depth_accum[y_start:y_end, x_start:x_end] += tile_depth * weight
      weight_accum[y_start:y_end, x_start:x_end] += weight

  weight_accum[weight_accum == 0] = 1.0
  full_depth = depth_accum / weight_accum
  return cv2.normalize(full_depth, None, 0, 255, cv2.NORM_MINMAX)


def enhance_edges(
    depth_map: np.ndarray, image_rgb: np.ndarray, edge_weight: float = 0.25
) -> np.ndarray:
  gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
  sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
  sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
  magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
  magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

  blur_gray = cv2.GaussianBlur(gray, (0, 0), 3)
  high_pass = cv2.addWeighted(gray, 1.5, blur_gray, -0.5, 0)
  high_pass = cv2.normalize(high_pass, None, 0, 255, cv2.NORM_MINMAX)

  enhanced = (1.0 - edge_weight) * depth_map + edge_weight * (
      0.5 * magnitude + 0.5 * high_pass
  )
  return cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


# ============================================================
# 4. G-CODE GENERATORS FOR MULTI-LAYERS
# ============================================================


def generate_facing_gcode(
    width_mm: float, height_mm: float, face_depth: float, tool: ToolConfig, safe_z: float
) -> List[str]:
  """Layer 1: Phay phẳng mặt gỗ."""
  gcode = [
      "( --- LAYER 1: PHAY MAT PHANG GO (FACING) --- )",
      f"T{tool.tool_id} M06 (Tool: {tool.name})",
      f"M03 S{tool.spindle_speed}",
  ]
  stepover = tool.diameter * (tool.stepover_pct / 100.0)
  y = 0.0
  gcode.append(f"G00 Z{safe_z:.3f}")
  gcode.append(f"G00 X0.000 Y0.000")
  gcode.append(f"G01 Z{-face_depth:.3f} F{tool.plunge_rate:.0f}")

  direction = 1
  while y <= height_mm:
    x_target = width_mm if direction == 1 else 0.0
    gcode.append(f"G01 X{x_target:.3f} Y{y:.3f} F{tool.feed_rate:.0f}")
    y += stepover
    if y <= height_mm:
      gcode.append(f"G01 Y{y:.3f} F{tool.feed_rate:.0f}")
    direction *= -1

  gcode.append(f"G00 Z{safe_z:.3f}")
  return gcode


def generate_roughing_gcode(
    depth_map: np.ndarray,
    width_mm: float,
    height_mm: float,
    max_depth_mm: float,
    tool: ToolConfig,
    safe_z: float,
) -> List[str]:
  """Layer 2: Phá thô (Roughing) bỏ phần gỗ thừa lớn."""
  gcode = [
      "( --- LAYER 2: PHA THO (ROUGHING) --- )",
      f"T{tool.tool_id} M06 (Tool: {tool.name})",
      f"M03 S{tool.spindle_speed}",
  ]

  stepover = tool.diameter * (tool.stepover_pct / 100.0)
  num_passes = int(np.ceil(max_depth_mm / tool.pass_depth))

  img_h, img_w = depth_map.shape
  res_x = width_mm / float(img_w)

  for p in range(1, num_passes + 1):
    target_z = -min(p * tool.pass_depth, max_depth_mm)
    gcode.append(
        f"\n( Pass {p}/{num_passes} - Cut Depth: {target_z:.3f}mm )"
    )

    y_steps = int(height_mm / stepover)
    x_steps = int(width_mm / stepover)

    z_matrix = -(1.0 - depth_map.astype(np.float32) / 255.0) * max_depth_mm
    z_resampled = cv2.resize(
        z_matrix, (x_steps, y_steps), interpolation=cv2.INTER_AREA
    )

    for y_idx in range(y_steps):
      y_pos = (y_idx / max(1, y_steps - 1)) * height_mm
      row_z = z_resampled[y_idx]

      # Chỉ phá thô ở những vị trí mà mô hình 3D đục sâu hơn target_z
      active_indices = np.where(row_z < target_z)[0]
      if len(active_indices) == 0:
        continue

      start_x = (
          active_indices[0] / max(1, x_steps - 1)
      ) * width_mm - tool.diameter
      end_x = (
          active_indices[-1] / max(1, x_steps - 1)
      ) * width_mm + tool.diameter

      start_x = max(0.0, start_x)
      end_x = min(width_mm, end_x)

      gcode.append(f"G00 Z{safe_z:.3f}")
      gcode.append(f"G00 X{start_x:.3f} Y{y_pos:.3f}")
      gcode.append(f"G01 Z{target_z:.3f} F{tool.plunge_rate:.0f}")
      gcode.append(f"G01 X{end_x:.3f} Y{y_pos:.3f} F{tool.feed_rate:.0f}")

  gcode.append(f"G00 Z{safe_z:.3f}")
  return gcode


def generate_finishing_gcode(
    depth_map: np.ndarray,
    width_mm: float,
    height_mm: float,
    max_depth_mm: float,
    tool: ToolConfig,
    safe_z: float,
    rdp_epsilon: float = 0.01,
) -> List[str]:
  """Layer 3: Đục tinh 3D (Finishing) chi tiết nét."""
  gcode = [
      "( --- LAYER 3: DUC TINH 3D (FINISHING) --- )",
      f"T{tool.tool_id} M06 (Tool: {tool.name})",
      f"M03 S{tool.spindle_speed}",
  ]

  img_h, img_w = depth_map.shape
  res_x = width_mm / float(img_w)

  z_matrix = -(1.0 - depth_map.astype(np.float32) / 255.0) * max_depth_mm
  radius_px = int((tool.diameter / 2.0) / res_x)
  z_matrix_comp = compensate_ballnose_kernel(z_matrix, radius_px, res_x)

  stepover_mm = tool.diameter * (tool.stepover_pct / 100.0)
  x_steps = int(width_mm / stepover_mm)
  y_steps = int(height_mm / stepover_mm)

  z_resampled = cv2.resize(
      z_matrix_comp, (x_steps, y_steps), interpolation=cv2.INTER_CUBIC
  )

  for y_idx in range(y_steps):
    x_order = (
        np.arange(x_steps) if y_idx % 2 == 0 else np.arange(x_steps)[::-1]
    )
    z_line = z_resampled[y_idx, x_order]
    keep_mask = rdp_simplify_1d(z_line, rdp_epsilon)

    first = True
    for i, x_idx in enumerate(x_order):
      if not keep_mask[i]:
        continue

      x_pos = (x_idx / max(1, x_steps - 1)) * width_mm
      y_pos = (y_idx / max(1, y_steps - 1)) * height_mm
      z_pos = z_line[i]

      if first:
        gcode.append(f"G00 Z{safe_z:.3f}")
        gcode.append(f"G00 X{x_pos:.3f} Y{y_pos:.3f}")
        gcode.append(f"G01 Z{z_pos:.3f} F{tool.plunge_rate:.0f}")
        first = False
      else:
        gcode.append(
            f"G01 X{x_pos:.3f} Y{y_pos:.3f} Z{z_pos:.3f}"
            f" F{tool.feed_rate:.0f}"
        )

  gcode.append(f"G00 Z{safe_z:.3f}")
  return gcode


def generate_cutout_gcode(
    width_mm: float,
    height_mm: float,
    cut_depth_mm: float,
    tool: ToolConfig,
    safe_z: float,
    use_tabs: bool = True,
    tab_width: float = 10.0,
    tab_height: float = 3.0,
) -> List[str]:
  """Layer 4: Cắt rời khung viền (Cut Out Profile)."""
  gcode = [
      "( --- LAYER 4: CAT KHUNG VIEN (CUTOUT) --- )",
      f"T{tool.tool_id} M06 (Tool: {tool.name})",
      f"M03 S{tool.spindle_speed}",
  ]

  r = tool.diameter / 2.0
  x0, x1 = -r, width_mm + r
  y0, y1 = -r, height_mm + r
  num_passes = int(np.ceil(cut_depth_mm / tool.pass_depth))

  for p in range(1, num_passes + 1):
    target_z = -min(p * tool.pass_depth, cut_depth_mm)
    gcode.append(
        f"\n( Pass {p}/{num_passes} - Depth: {target_z:.3f}mm )"
    )

    if p == 1:
      gcode.append(f"G00 X{x0:.3f} Y{y0:.3f}")
      gcode.append(f"G01 Z{target_z:.3f} F{tool.plunge_rate:.0f}")
    else:
      gcode.append(f"G01 Z{target_z:.3f} F{tool.plunge_rate:.0f}")

    is_bottom = (target_z <= -cut_depth_mm + tab_height) and use_tabs

    # Edge 1: Bottom
    if is_bottom and (x1 - x0) > 2 * tab_width:
      mid_x = (x0 + x1) / 2.0
      gcode.append(
          f"G01 X{mid_x - tab_width/2:.3f} Y{y0:.3f} F{tool.feed_rate:.0f}"
      )
      gcode.append(
          f"G01 Z{target_z + tab_height:.3f} F{tool.plunge_rate:.0f}"
      )
      gcode.append(
          f"G01 X{mid_x + tab_width/2:.3f} Y{y0:.3f} F{tool.feed_rate:.0f}"
      )
      gcode.append(f"G01 Z{target_z:.3f} F{tool.plunge_rate:.0f}")
      gcode.append(f"G01 X{x1:.3f} Y{y0:.3f} F{tool.feed_rate:.0f}")
    else:
      gcode.append(f"G01 X{x1:.3f} Y{y0:.3f} F{tool.feed_rate:.0f}")

    # Edge 2: Right
    if is_bottom and (y1 - y0) > 2 * tab_width:
      mid_y = (y0 + y1) / 2.0
      gcode.append(
          f"G01 X{x1:.3f} Y{mid_y - tab_width/2:.3f} F{tool.feed_rate:.0f}"
      )
      gcode.append(
          f"G01 Z{target_z + tab_height:.3f} F{tool.plunge_rate:.0f}"
      )
      gcode.append(
          f"G01 X{x1:.3f} Y{mid_y + tab_width/2:.3f} F{tool.feed_rate:.0f}"
      )
      gcode.append(f"G01 Z{target_z:.3f} F{tool.plunge_rate:.0f}")
      gcode.append(f"G01 X{x1:.3f} Y{y1:.3f} F{tool.feed_rate:.0f}")
    else:
      gcode.append(f"G01 X{x1:.3f} Y{y1:.3f} F{tool.feed_rate:.0f}")

    # Edge 3: Top
    if is_bottom and (x1 - x0) > 2 * tab_width:
      mid_x = (x0 + x1) / 2.0
      gcode.append(
          f"G01 X{mid_x + tab_width/2:.3f} Y{y1:.3f} F{tool.feed_rate:.0f}"
      )
      gcode.append(
          f"G01 Z{target_z + tab_height:.3f} F{tool.plunge_rate:.0f}"
      )
      gcode.append(
          f"G01 X{mid_x - tab_width/2:.3f} Y{y1:.3f} F{tool.feed_rate:.0f}"
      )
      gcode.append(f"G01 Z{target_z:.3f} F{tool.plunge_rate:.0f}")
      gcode.append(f"G01 X{x0:.3f} Y{y1:.3f} F{tool.feed_rate:.0f}")
    else:
      gcode.append(f"G01 X{x0:.3f} Y{y1:.3f} F{tool.feed_rate:.0f}")

    # Edge 4: Left
    if is_bottom and (y1 - y0) > 2 * tab_width:
      mid_y = (y0 + y1) / 2.0
      gcode.append(
          f"G01 X{x0:.3f} Y{mid_y + tab_width/2:.3f} F{tool.feed_rate:.0f}"
      )
      gcode.append(
          f"G01 Z{target_z + tab_height:.3f} F{tool.plunge_rate:.0f}"
      )
      gcode.append(
          f"G01 X{x0:.3f} Y{mid_y - tab_width/2:.3f} F{tool.feed_rate:.0f}"
      )
      gcode.append(f"G01 Z{target_z:.3f} F{tool.plunge_rate:.0f}")
      gcode.append(f"G01 X{x0:.3f} Y{y0:.3f} F{tool.feed_rate:.0f}")
    else:
      gcode.append(f"G01 X{x0:.3f} Y{y0:.3f} F{tool.feed_rate:.0f}")

  gcode.append(f"G00 Z{safe_z:.3f}")
  return gcode


# ============================================================
# 5. STREAMLIT UI & INTERACTION
# ============================================================

st.title("🪵 AI CNC Wood Relief Generator v4 - Multi-Layer True CAM")
st.caption(
    "Hệ thống tạo G-code phân tầng chuyên nghiệp: Phay mặt phẳng ➔ Phá thô ➔"
    " Đục 3D tinh ➔ Cắt khung viền"
)

col_left, col_right = st.columns([1, 2])

with col_left:
  st.header("1️⃣ Cấu hình Phôi & Chế độ Máy")
  uploaded_file = st.file_uploader(
      "Tải ảnh tranh gỗ (Hỗ trợ 4K/8K)", type=["jpg", "jpeg", "png"]
  )

  machine_type = st.radio(
      "⚙️ Loại máy CNC của bạn:",
      ["Máy thay dao thủ công (Manual / Non-ATC)", "Máy tự động đổi dao (ATC)"],
  )

  model_type = st.selectbox(
      "🤖 Mô hình AI Depth", ["Large (Siêu nét)", "Small (Nhanh)"]
  )
  edge_weight = st.slider(
      "🪛 Nổi bật viền chi tiết (Edge Enhancement)", 0.0, 0.5, 0.25, 0.05
  )

  st.subheader("🖼️ Kích thước phôi gỗ")
  width_mm = st.number_input("Rộng X (mm)", value=300.0)
  height_mm = st.number_input("Cao Y (mm)", value=400.0)
  max_depth_mm = st.number_input("Độ sâu 3D Z-max (mm)", value=12.0)
  cut_depth_mm = st.number_input(
      "Độ dày phôi gỗ cắt đứt (mm)",
      value=15.0,
      help="Nên dày hơn Z-max 3D",
  )
  safe_z = st.number_input("Safe Z (mm)", value=5.0)

  st.markdown("---")
  st.header("2️⃣ Cấu hình 4 Dao cho 4 Layer")

  enable_facing = st.checkbox("Bật Layer 1: Phay mặt ván (Facing)", value=True)
  with st.expander("🛠️ Dao Layer 1: Phay Mặt (Facing Tool)", expanded=False):
    tool1 = ToolConfig(
        tool_id=1,
        name="Dao Phay Mat D25",
        tool_type="Flat",
        diameter=st.number_input(
            "Đường kính dao phay mặt (mm)", value=25.0, key="t1_d"
        ),
        stepover_pct=st.slider("Stepover (%)", 10, 80, 50, key="t1_s"),
        pass_depth=st.number_input(
            "Độ sâu phay mặt (mm)", value=0.5, key="t1_p"
        ),
        feed_rate=st.number_input(
            "Feedrate (mm/phút)", value=3500, key="t1_f"
        ),
        plunge_rate=st.number_input("Plunge Rate", value=800, key="t1_pl"),
        spindle_speed=18000,
    )

  enable_roughing = st.checkbox("Bật Layer 2: Phá thô (Roughing)", value=True)
  with st.expander("🛠️ Dao Layer 2: Phá Thô (Roughing Tool)", expanded=False):
    tool2 = ToolConfig(
        tool_id=2,
        name="Dao Xoan Endmill D6",
        tool_type="Flat",
        diameter=st.number_input(
            "Đường kính dao phá thô (mm)", value=6.0, key="t2_d"
        ),
        stepover_pct=st.slider("Stepover (%)", 10, 60, 40, key="t2_s"),
        pass_depth=st.number_input(
            "Mỗi lớp cắt bớt Z (Stepdown mm)", value=4.0, key="t2_p"
        ),
        feed_rate=st.number_input(
            "Feedrate (mm/phút)", value=3000, key="t2_f"
        ),
        plunge_rate=st.number_input("Plunge Rate", value=600, key="t2_pl"),
        spindle_speed=18000,
    )

  with st.expander("🛠️ Dao Layer 3: Đục Tinh 3D (Finishing Tool)", expanded=True):
    tool3 = ToolConfig(
        tool_id=3,
        name="Dao Cau Ballnose D2",
        tool_type="Ball",
        diameter=st.number_input(
            "Đường kính dao cầu (mm)", value=2.0, key="t3_d"
        ),
        stepover_pct=st.slider("Stepover (%)", 5, 30, 10, key="t3_s"),
        pass_depth=max_depth_mm,
        feed_rate=st.number_input(
            "Feedrate 3D (mm/phút)", value=2500, key="t3_f"
        ),
        plunge_rate=st.number_input("Plunge Rate", value=500, key="t3_pl"),
        spindle_speed=20000,
    )
    rdp_epsilon = st.select_slider(
        "Mức độ nén G-code (RDP Epsilon)",
        options=[0.001, 0.005, 0.01, 0.02],
        value=0.01,
    )

  with st.expander("🛠️ Dao Layer 4: Cắt Khung Viền (Cutout Tool)", expanded=True):
    tool4 = ToolConfig(
        tool_id=4,
        name="Dao Cat Flat D4",
        tool_type="Flat",
        diameter=st.number_input(
            "Đường kính dao cắt (mm)", value=4.0, key="t4_d"
        ),
        stepover_pct=100.0,
        pass_depth=st.number_input(
            "Mỗi lớp cắt sâu (Stepdown mm)", value=3.0, key="t4_p"
        ),
        feed_rate=st.number_input(
            "Cutout Feedrate (mm/phút)", value=1500, key="t4_f"
        ),
        plunge_rate=st.number_input("Plunge Rate", value=400, key="t4_pl"),
        spindle_speed=18000,
    )
    use_tabs = st.checkbox(
        "Tạo Cầu Nối Giữ Phôi (Tabs / Bridges)", value=True
    )
    if use_tabs:
      col_tab1, col_tab2 = st.columns(2)
      with col_tab1:
        tab_width = st.number_input("Rộng Tab (mm)", value=10.0)
      with col_tab2:
        tab_height = st.number_input("Cao Tab (mm)", value=3.0)
    else:
      tab_width, tab_height = 10.0, 3.0

  process_btn = st.button(
      "🚀 XUẤT FULL G-CODE ĐA LAYER", type="primary", use_container_width=True
  )

with col_right:
  st.header("3️⃣ Kết quả Xuất G-Code")

  if uploaded_file is None:
    st.info("Vui lòng tải ảnh lên để bắt đầu.")
  else:
    raw_image = Image.open(uploaded_file).convert("RGB")
    image_rgb = np.array(raw_image)
    st.image(
        image_rgb,
        caption=f"Ảnh phôi ({image_rgb.shape[1]}x{image_rgb.shape[0]} px)",
        use_container_width=True,
    )

    if process_btn:
      with st.spinner("AI đang trích xuất Depth Map & Phân tích Toolpaths..."):
        raw_depth = process_depth_tiled(image_rgb, model_type)
        enhanced_depth = enhance_edges(raw_depth, image_rgb, edge_weight)

      st.image(
          enhanced_depth, caption="Depth Map AI Ultra-HD", use_container_width=True
      )

      with st.spinner("Đang tổng hợp đường chạy dao cho các Layer..."):
        generated_layers = {}

        # 1. Facing
        if enable_facing:
          generated_layers["01_facing.nc"] = generate_facing_gcode(
              width_mm, height_mm, tool1.pass_depth, tool1, safe_z
          )

        # 2. Roughing
        if enable_roughing:
          generated_layers["02_roughing.nc"] = generate_roughing_gcode(
              enhanced_depth,
              width_mm,
              height_mm,
              max_depth_mm,
              tool2,
              safe_z,
          )

        # 3. Finishing
        generated_layers["03_finishing_3d.nc"] = generate_finishing_gcode(
            enhanced_depth,
            width_mm,
            height_mm,
            max_depth_mm,
            tool3,
            safe_z,
            rdp_epsilon,
        )

        # 4. Cutout
        generated_layers["04_cutout.nc"] = generate_cutout_gcode(
            width_mm,
            height_mm,
            cut_depth_mm,
            tool4,
            safe_z,
            use_tabs,
            tab_width,
            tab_height,
        )

        # Xử lý chèn M00 nếu là máy Manual
        header_common = [
            "(==================================================)",
            f"( PROJECT: AI CNC MULTI-LAYER RELIEF )",
            f"( MACHINE TYPE: {machine_type} )",
            "(==================================================)",
            "G21 (Don vi mm)\nG90 (Toa do tuyet doi)\nG17 (Mat phang XY)",
        ]

        file_texts = {}
        combined_lines = list(header_common)

        for filename, lines in generated_layers.items():
          # Nếu là máy Manual, chèn lệnh tạm dừng M00 để thợ đo lại Z0
          layer_header = []
          if "Manual" in machine_type:
            layer_header.extend([
                "\n(--------------------------------------------------)",
                "( PHƯƠNG THỨC THAY DAO THỦ CÔNG )",
                "M05 (Tắt trục chính)",
                f"G00 Z{safe_z + 20:.3f} (Nâng dao an toàn để thay)",
                (
                    "M00 (TẠM DÙNG MÁY - VUI LÒNG LẮP DAO VÀ ĐO LẠI GỐC"
                    " Z-ZERO)"
                ),
                "(--------------------------------------------------)\n",
            ])

          full_layer_code = (
              header_common + layer_header + lines + ["M05", "G00 Z50.000"]
          )
          file_texts[filename] = "\n".join(full_layer_code)

          combined_lines.extend(layer_header)
          combined_lines.extend(lines)

        combined_lines.extend(
            ["\nM05 (Tắt trục chính)", "G00 Z50.000", "G00 X0 Y0", "M30"]
        )
        combined_text = "\n".join(combined_lines)

      st.success("🎉 TẠO THÀNH CÔNG FULL BỘ G-CODE CHO CẢ 4 LAYER!")

      # Khu vực Tải File
      st.markdown("### 📥 Tải File G-code Cho Xưởng Của Bạn")

      col_d1, col_d2 = st.columns(2)

      with col_d1:
        st.subheader("1. File Tách Lẻ (Máy Thay Dao Thủ Công)")
        for fname, fcontent in file_texts.items():
          st.download_button(
              label=f"💾 Tải {fname}",
              data=fcontent,
              file_name=fname,
              mime="text/plain",
              use_container_width=True,
          )

      with col_d2:
        st.subheader("2. File Tổng Hợp (Máy ATC / Tự Đổi Dao)")
        st.download_button(
            label="💾 TẢI FILE TỔNG HỢP FULL_PROCESS.NC",
            data=combined_text,
            file_name="full_machining_atc.nc",
            mime="text/plain",
            type="primary",
            use_container_width=True,
            help=(
                "Chứa toàn bộ các Layer nối tiếp nhau, tự động phát lệnh M06"
                " đổi dao."
            ),
        )

      # Xem trước G-code
      st.markdown("### 👁️ Xem Trước Mã G-Code Từng Layer")
      tabs_view = st.tabs(list(file_texts.keys()) + ["FULL_COMBINED.NC"])
      for idx, (fname, fcontent) in enumerate(file_texts.items()):
        with tabs_view[idx]:
          st.code(fcontent[:5000], language="gcode")
      with tabs_view[-1]:
        st.code(combined_text[:10000], language="gcode")
