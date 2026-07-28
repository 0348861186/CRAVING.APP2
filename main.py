import streamlit as st
import numpy as np
import cv2
from PIL import Image
import torch
from transformers import pipeline
from dataclasses import dataclass
from typing import List, Tuple
from numba import njit

# ============================================================
# AI CNC WOOD RELIEF GENERATOR v3 - ULTRA HD EDITION
# ============================================================

st.set_page_config(
    page_title="AI CNC Relief Generator v3 (Ultra HD)",
    page_icon="🪵",
    layout="wide"
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
    
    # Sử dụng stack thay cho đệ quy để tối ưu Numba
    stack = [(0, n - 1)]
    
    while len(stack) > 0:
        start, end = stack.pop()
        if end - start <= 1:
            continue
            
        # Tìm điểm xa nhất so với đoạn thẳng nối start - end
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
                dist = np.abs(dz * i - dx * points[i] + x2 * z1 - z2 * x1) / denom
                
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
def compensate_ballnose_kernel(depth_mm: np.ndarray, radius_px: int, res_x: float) -> np.ndarray:
    """Bù bán kính dao cầu (Heightfield / Z-buffer Offset)."""
    h, w = depth_mm.shape
    compensated = np.copy(depth_mm)
    
    if radius_px < 1:
        return compensated
        
    for y in range(h):
        for x in range(w):
            max_z = -9999.0
            # Lướt qua cửa sổ tròn xung quanh bán kính dao
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
                        sphere_offset = np.sqrt(max(0.0, r_sq - dist_sq)) - (radius_px * res_x)
                        z_val = depth_mm[ny, nx] + sphere_offset
                        if z_val > max_z:
                            max_z = z_val
            compensated[y, x] = max_z
            
    return compensated

# ============================================================
# 2. DATA STRUCTURES & LOAD AI MODEL
# ============================================================

@dataclass
class MachiningLayer:
    name: str
    mask: np.ndarray
    depth_min: float
    depth_max: float
    tool_type: str
    strategy: str
    color: Tuple[int, int, int]

@st.cache_resource
def load_depth_ai(model_type: str):
    device = 0 if torch.cuda.is_available() else -1
    model_id = "depth-anything/Depth-Anything-V2-Large-hf" if model_type == "Large (Siêu nét)" else "depth-anything/Depth-Anything-V2-Small-hf"
    pipe = pipeline(task="depth-estimation", model=model_id, device=device)
    return pipe

# ============================================================
# 3. ADVANCED PROCESSING (TILE INFERENCE & EDGE ENHANCEMENT)
# ============================================================

def process_depth_tiled(image_rgb: np.ndarray, model_type: str, tile_size: int = 1024, overlap: int = 128) -> np.ndarray:
    """Chia ảnh lớn thành các Tile nhỏ có overlap để AI giữ chi tiết cao tần."""
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

            # Tạo weighting mask mịn ở viền ô
            ty, tx = tile_depth.shape
            mask_y = np.sin(np.linspace(0, np.pi, ty)) ** 2
            mask_x = np.sin(np.linspace(0, np.pi, tx)) ** 2
            weight = np.outer(mask_y, mask_x)

            depth_accum[y_start:y_end, x_start:x_end] += tile_depth * weight
            weight_accum[y_start:y_end, x_start:x_end] += weight

    weight_accum[weight_accum == 0] = 1.0
    full_depth = depth_accum / weight_accum
    return cv2.normalize(full_depth, None, 0, 255, cv2.NORM_MINMAX)

def enhance_edges(depth_map: np.ndarray, image_rgb: np.ndarray, edge_weight: float = 0.25) -> np.ndarray:
    """Tăng cường biên nét chi tiết mảnh (Mắt, mũi, vảy cá) vào Depth Map."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    
    # Trích xuất viền sắc nét bằng Sobel
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # Lọc thông cao (High-pass filter)
    blur_gray = cv2.GaussianBlur(gray, (0, 0), 3)
    high_pass = cv2.addWeighted(gray, 1.5, blur_gray, -0.5, 0)
    high_pass = cv2.normalize(high_pass, None, 0, 255, cv2.NORM_MINMAX)

    # Trộn biên vào Depth Map
    enhanced = (1.0 - edge_weight) * depth_map + edge_weight * (0.5 * magnitude + 0.5 * high_pass)
    return cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# ============================================================
# 4. SEGMENTATION & TOOLPATH
# ============================================================

def create_smooth_regions(image_rgb: np.ndarray, depth_map: np.ndarray, region_count: int) -> List[np.ndarray]:
    h, w = depth_map.shape
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    
    features = np.zeros((h * w, 4), dtype=np.float32)
    features[:, 0] = lab[:, :, 0].flatten() / 255.0
    features[:, 1] = lab[:, :, 1].flatten() / 255.0
    features[:, 2] = lab[:, :, 2].flatten() / 255.0
    features[:, 3] = (depth_map.flatten() / 255.0) * 2.0  # Tăng trọng số cho Depth

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.2)
    _, labels, _ = cv2.kmeans(features, region_count, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(h, w)

    regions = []
    region_info = []

    for idx in range(region_count):
        mask = (labels == idx).astype(np.uint8) * 255
        pixel_values = depth_map[mask > 0]
        if len(pixel_values) < 50:
            continue
        region_info.append((float(np.mean(pixel_values)), mask))

    region_info.sort(key=lambda x: x[0])

    for _, mask in region_info:
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        regions.append(mask)

    return regions

def build_machining_layers(image_rgb: np.ndarray, depth_map: np.ndarray, region_count: int, max_depth_mm: float) -> List[MachiningLayer]:
    regions = create_smooth_regions(image_rgb, depth_map, region_count)
    layers = []
    colors = [(255, 0, 0), (0, 180, 0), (0, 100, 255), (255, 150, 0), (180, 0, 180)]

    for idx, mask in enumerate(regions):
        pixels = depth_map[mask > 0]
        if len(pixels) == 0:
            continue
        z_min = -(1.0 - float(np.percentile(pixels, 90)) / 255.0) * max_depth_mm
        z_max = -(1.0 - float(np.percentile(pixels, 10)) / 255.0) * max_depth_mm

        layers.append(MachiningLayer(
            name=f"Layer {idx + 1} - Phay Chi Tiết",
            mask=mask,
            depth_min=z_min,
            depth_max=z_max,
            tool_type="Dao cầu Ultra-HD",
            strategy="3D Raster Offset",
            color=colors[idx % len(colors)]
        ))
    return layers

def generate_layer_raster_optimized(
    depth_map: np.ndarray,
    mask: np.ndarray,
    width_mm: float,
    height_mm: float,
    max_depth_mm: float,
    tool_dia: float,
    stepover_mm: float,
    feed_rate: float,
    safe_z: float,
    x_offset: float,
    y_offset: float,
    rdp_epsilon: float = 0.01
) -> List[str]:
    
    img_h, img_w = depth_map.shape
    res_x = width_mm / float(img_w)
    
    # Đổi depth sáng tối thành chiều sâu Z (mm)
    z_matrix = -(1.0 - depth_map.astype(np.float32) / 255.0) * max_depth_mm
    
    # Bù bán kính dao cầu (3D Ballnose Compensation)
    radius_px = int((tool_dia / 2.0) / res_x)
    z_matrix_comp = compensate_ballnose_kernel(z_matrix, radius_px, res_x)

    x_steps = int(width_mm / stepover_mm)
    y_steps = int(height_mm / stepover_mm)

    z_resampled = cv2.resize(z_matrix_comp, (x_steps, y_steps), interpolation=cv2.INTER_CUBIC)
    mask_resampled = cv2.resize(mask, (x_steps, y_steps), interpolation=cv2.INTER_NEAREST)

    gcode = []

    for y_idx in range(y_steps):
        row_mask = mask_resampled[y_idx] > 0
        if not np.any(row_mask):
            continue

        valid_x = np.where(row_mask)[0]
        x_order = valid_x if y_idx % 2 == 0 else valid_x[::-1]

        z_line = z_resampled[y_idx, x_order]
        
        # Nén đường chạy bằng RDP algorithm
        keep_mask = rdp_simplify_1d(z_line, rdp_epsilon)

        first = True
        for i, x_idx in enumerate(x_order):
            if not keep_mask[i]:
                continue

            x_pos = (x_idx / max(1, x_steps - 1)) * width_mm - x_offset
            y_pos = (y_idx / max(1, y_steps - 1)) * height_mm - y_offset
            z_pos = z_line[i]

            if first:
                gcode.append(f"G00 Z{safe_z:.3f}")
                gcode.append(f"G00 X{x_pos:.3f} Y{y_pos:.3f}")
                gcode.append(f"G01 Z{z_pos:.3f} F{max(300, int(feed_rate / 2))}")
                first = False
            else:
                gcode.append(f"G01 X{x_pos:.3f} Y{y_pos:.3f} Z{z_pos:.3f} F{feed_rate:.0f}")

    gcode.append(f"G00 Z{safe_z:.3f}")
    return gcode

# ============================================================
# 5. STREAMLIT UI
# ============================================================

st.title("🪵 AI CNC Wood Relief Generator v3 - Ultra HD")
st.caption("Nâng cấp: Tile Inference 4K + Model Large + Bù bán kính dao cầu + Nén G-code Numba RDP")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("1️⃣ Cấu hình & Tranh Ultra-HD")
    uploaded_file = st.file_uploader("Tải ảnh tranh gỗ (Hỗ trợ 4K/8K)", type=["jpg", "jpeg", "png"])

    model_type = st.selectbox("🤖 Mô hình AI Depth", ["Large (Siêu nét)", "Small (Nhanh)"])
    max_size = st.number_input("Kích thước tối đa xử lý (px)", value=3000, step=500)
    edge_weight = st.slider("🪛 Trộn chi tiết viền (Edge Enhancement)", 0.0, 0.5, 0.25, 0.05)

    st.subheader("🖼️ Kích thước thực tế CNC")
    width_mm = st.number_input("Rộng X (mm)", value=300.0)
    height_mm = st.number_input("Cao Y (mm)", value=400.0)
    max_depth_mm = st.number_input("Độ sâu 3D Z max (mm)", value=12.0)

    st.subheader("⚙️ Thông số dao & Nén G-code")
    tool_dia = st.number_input("Đường kính dao cầu (mm)", value=2.0, step=0.5)
    stepover_pct = st.slider("Stepover (%)", 5, 40, 10)
    feed_rate = st.number_input("Feedrate (mm/phút)", value=2500)
    safe_z = st.number_input("Safe Z (mm)", value=5.0)
    rdp_epsilon = st.select_slider("Mức độ nén G-code (RDP Epsilon)", options=[0.001, 0.005, 0.01, 0.02], value=0.01)

    process_button = st.button("🚀 TẠO G-CODE ULTRA-HD", type="primary", use_container_width=True)

with col2:
    st.header("2️⃣ Hiển thị & Kết xuất")

    if uploaded_file is None:
        st.info("Vui lòng tải ảnh lên để bắt đầu.")
    else:
        raw_image = Image.open(uploaded_file).convert("RGB")
        w, h = raw_image.size
        scale = min(1.0, max_size / max(w, h))
        if scale < 1.0:
            raw_image = raw_image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        
        image_rgb = np.array(raw_image)
        st.image(image_rgb, caption=f"Ảnh đầu vào ({image_rgb.shape[1]}x{image_rgb.shape[0]} px)", use_container_width=True)

        if process_button:
            with st.spinner("AI đang xử lý Tiled Depth Map 4K..."):
                raw_depth = process_depth_tiled(image_rgb, model_type)
                enhanced_depth = enhance_edges(raw_depth, image_rgb, edge_weight)

            st.subheader("🧠 Bản đồ độ sâu AI Ultra-HD (Đã trộn biên nét)")
            st.image(enhanced_depth, caption="Chi tiết nổi rõ ràng sắc nét", use_container_width=True)

            with st.spinner("Đang tính toán bù bán kính dao cầu & Nén G-code bằng Numba..."):
                layers = build_machining_layers(image_rgb, enhanced_depth, 3, max_depth_mm)
                
                stepover_mm = tool_dia * (stepover_pct / 100.0)
                gcode_lines = [
                    "(==================================================)",
                    "( AI CNC WOOD RELIEF GENERATOR v3 - ULTRA HD )",
                    "(==================================================)",
                    "G21\nG90\nG17\nM03 S18000",
                    f"G00 Z{safe_z:.3f}"
                ]

                for layer in layers:
                    layer_code = generate_layer_raster_optimized(
                        depth_map=enhanced_depth,
                        mask=layer.mask,
                        width_mm=width_mm,
                        height_mm=height_mm,
                        max_depth_mm=max_depth_mm,
                        tool_dia=tool_dia,
                        stepover_mm=stepover_mm,
                        feed_rate=feed_rate,
                        safe_z=safe_z,
                        x_offset=0.0,
                        y_offset=0.0,
                        rdp_epsilon=rdp_epsilon
                    )
                    gcode_lines.extend(layer_code)

                gcode_lines.extend([f"G00 Z{safe_z:.3f}", "M05", "M30"])
                gcode_txt = "\n".join(gcode_lines)

            st.success(f"🎉 TẠO G-CODE THÀNH CÔNG! Số dòng lệnh: {len(gcode_lines):,}")
            
            st.download_button(
                label="💾 TẢI FILE G-CODE (.NC)",
                data=gcode_txt,
                file_name="relief_ultrahd.nc",
                mime="text/plain",
                use_container_width=True
            )

            with st.expander("👁️ Xem trước G-code"):
                st.code(gcode_txt[:10000], language="gcode")
