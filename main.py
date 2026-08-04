CODE1
import io
import json
import math
import os
import tempfile
import time
from typing import List, Literal, Optional

import cv2
import ezdxf
from ezdxf import path
from google import genai
from google.genai import types
import matplotlib.pyplot as plt
from matplotlib.patches import Arc as MplArc
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field, ValidationError
from shapely.geometry import MultiPolygon, Polygon
import streamlit as st


# ==============================================================================
# 1. ENHANCED GEMINI SCHEMA (BỔ SUNG CHÚ THÍCH VÀ TRÍCH XUẤT KÍCH THƯỚC CHI TIẾT)
# ==============================================================================
class ParametricShapeModel(BaseModel):
  name: str = Field(default="shape", description="Tên đối tượng hình học")
  shape_type: Literal[
      "RECTANGLE", "CIRCLE", "REGULAR_POLYGON", "ARC", "POLYGON"
  ] = Field(description="Loại hình học nhận diện được trong ảnh")
  description_vi: str = Field(
      default="Chưa rõ",
      description=(
          "Mô tả chi tiết bằng tiếng Việt, ví dụ: 'Hình sau xử lý là hình chữ"
          " nhật (100mm x 50mm)'"
      ),
  )
  width: Optional[float] = Field(
      default=None, description="Chiều rộng đọc từ kích thước ghi chú (mm)"
  )
  height: Optional[float] = Field(
      default=None, description="Chiều cao đọc từ kích thước ghi chú (mm)"
  )
  radius: Optional[float] = Field(
      default=None, description="Bán kính đọc từ ghi chú (mm)"
  )
  sides: Optional[int] = Field(default=None, description="Số cạnh đa giác")
  center: Optional[List[float]] = Field(
      default=None, description="Tọa độ tâm [X, Y]"
  )
  origin: Optional[List[float]] = Field(
      default=None, description="Gốc tọa độ [X, Y]"
  )
  side_length: Optional[float] = Field(
      default=None, description="Chiều dài 1 cạnh nếu là đa giác đều"
  )
  start_angle: Optional[float] = Field(
      default=None, description="Góc bắt đầu cho ARC (độ)"
  )
  end_angle: Optional[float] = Field(
      default=None, description="Góc kết thúc cho ARC (độ)"
  )
  points: Optional[List[List[float]]] = Field(
      default=None, description="Danh sách tọa độ các đỉnh [[x1,y1],...]"
  )
  rotation: Optional[float] = Field(
      default=0.0, description="Góc xoay hình (độ)"
  )
  confidence: Optional[float] = Field(
      default=1.0, description="Độ tin cậy trích xuất [0.0 - 1.0]"
  )
  layer: Optional[str] = Field(default="DEFAULT", description="Tên lớp CAD")


class GeminiResponseModel(BaseModel):
  shapes: List[ParametricShapeModel]


# ==============================================================================
# 2. SHAPE ENGINE: TRÍCH XUẤT HÌNH HỌC VÀ TÍNH KÍCH THƯỚC CẠNH
# ==============================================================================
class ShapeEngine:

  @staticmethod
  def rotate_point(pt, angle_deg, center=(0.0, 0.0)):
    if angle_deg == 0.0:
      return pt
    rad = math.radians(angle_deg)
    cx, cy = center
    x, y = pt[0] - cx, pt[1] - cy
    rx = x * math.cos(rad) - y * math.sin(rad)
    ry = x * math.sin(rad) + y * math.cos(rad)
    return (round(rx + cx, 4), round(ry + cy, 4))

  @classmethod
  def create_regular_polygon(
      cls, center, radius, sides, start_angle=90.0, rotation=0.0
  ):
    cx, cy = center
    pts = []
    angle_step = 360.0 / sides
    for i in range(sides):
      deg = start_angle + i * angle_step + rotation
      rad = math.radians(deg)
      x = cx + radius * math.cos(rad)
      y = cy + radius * math.sin(rad)
      pts.append((round(x, 4), round(y, 4)))
    pts.append(pts[0])  # Khép kín
    return pts

  @classmethod
  def create_rectangle(cls, width, height, origin=(0.0, 0.0), rotation=0.0):
    ox, oy = origin
    pts = [
        (ox, oy),
        (ox + width, oy),
        (ox + width, oy + height),
        (ox, oy + height),
        (ox, oy),
    ]
    if rotation != 0.0:
      center = (ox + width / 2.0, oy + height / 2.0)
      pts = [cls.rotate_point(p, rotation, center) for p in pts]
    return pts

  @classmethod
  def compile_parametric_to_geometry(
      cls, parametric_shapes: List[ParametricShapeModel], filename=""
  ):
    compiled_shapes = []
    for idx, item in enumerate(parametric_shapes):
      s_name = f"{filename}_{item.name}_{idx+1}"
      stype = item.shape_type
      rot = item.rotation if item.rotation else 0.0
      desc = item.description_vi or f"Hình sau xử lý là {stype}"

      if stype == "RECTANGLE":
        w = item.width if (item.width and item.width > 0) else 100.0
        h = item.height if (item.height and item.height > 0) else 50.0
        ox, oy = (
            item.origin
            if item.origin and len(item.origin) >= 2
            else [0.0, 0.0]
        )
        pts = cls.create_rectangle(w, h, (ox, oy), rotation=rot)

        compiled_shapes.append({
            "name": s_name,
            "type": "POLYLINE",
            "shape_kind": "RECTANGLE",
            "points": pts,
            "description": desc,
            "width": w,
            "height": h,
            "process_type": "Profile",
            "tool_offset": "Outside",
            "layer": item.layer,
            "confidence": item.confidence,
        })

      elif stype == "CIRCLE":
        cx, cy = (
            item.center
            if item.center and len(item.center) >= 2
            else [0.0, 0.0]
        )
        r = float(item.radius if item.radius else 25.0)
        compiled_shapes.append({
            "name": s_name,
            "type": "CIRCLE",
            "shape_kind": "CIRCLE",
            "center": (float(cx), float(cy)),
            "radius": r,
            "description": desc,
            "process_type": "Drill",
            "tool_offset": "Center",
            "layer": item.layer,
            "confidence": item.confidence,
        })

      elif stype == "REGULAR_POLYGON":
        sides = item.sides if item.sides and item.sides >= 3 else 5
        cx, cy = (
            item.center
            if item.center and len(item.center) >= 2
            else [0.0, 0.0]
        )
        if item.radius:
          radius = float(item.radius)
        elif item.side_length:
          radius = float(item.side_length) / (2 * math.sin(math.pi / sides))
        else:
          radius = 50.0

        pts = cls.create_regular_polygon((cx, cy), radius, sides, rotation=rot)
        compiled_shapes.append({
            "name": s_name,
            "type": "POLYLINE",
            "shape_kind": "REGULAR_POLYGON",
            "points": pts,
            "description": desc,
            "sides": sides,
            "process_type": "Profile",
            "tool_offset": "Outside",
            "layer": item.layer,
            "confidence": item.confidence,
        })

      elif stype == "ARC":
        cx, cy = (
            item.center
            if item.center and len(item.center) >= 2
            else [0.0, 0.0]
        )
        radius = float(item.radius if item.radius else 20.0)
        sa_deg = (
            float(item.start_angle if item.start_angle is not None else 0.0)
            + rot
        )
        ea_deg = (
            float(item.end_angle if item.end_angle is not None else 90.0) + rot
        )

        sa_rad, ea_rad = math.radians(sa_deg), math.radians(ea_deg)
        start_p = (
            cx + radius * math.cos(sa_rad),
            cy + radius * math.sin(sa_rad),
        )
        end_p = (cx + radius * math.cos(ea_rad), cy + radius * math.sin(ea_rad))

        compiled_shapes.append({
            "name": s_name,
            "type": "ARC",
            "shape_kind": "ARC",
            "center": (cx, cy),
            "radius": radius,
            "start_angle": sa_deg,
            "end_angle": ea_deg,
            "start_p": start_p,
            "end_p": end_p,
            "description": desc,
            "process_type": "Profile",
            "tool_offset": "Center",
            "cw": True,
            "layer": item.layer,
            "confidence": item.confidence,
        })

      else:  # POLYGON
        raw_pts = item.points if item.points else []
        pts = [(float(p[0]), float(p[1])) for p in raw_pts if len(p) >= 2]
        if pts:
          if rot != 0.0:
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            pts = [cls.rotate_point(p, rot, (cx, cy)) for p in pts]
          if pts[0] != pts[-1]:
            pts.append(pts[0])
          compiled_shapes.append({
              "name": s_name,
              "type": "POLYLINE",
              "shape_kind": "POLYGON",
              "points": pts,
              "description": desc,
              "process_type": "Profile",
              "tool_offset": "Outside",
              "layer": item.layer,
              "confidence": item.confidence,
          })

    return compiled_shapes


# ==============================================================================
# 3. GEMINI PARSER NÂNG CẤP ĐỌC KÍCH THƯỚC TRÊN ẢNH
# ==============================================================================
def parse_image_with_gemini_ai(image_bytes, api_key, filename):
  if not api_key:
    st.error("Vui lòng nhập Gemini API Key!")
    return []

  client = genai.Client(api_key=api_key)
  img = Image.open(io.BytesIO(image_bytes))

  prompt = """
    Bạn là một chuyên gia OCR và Kỹ sư CAM/CAD hàng đầu.
    Nhiệm vụ của bạn là phân tích ảnh vẽ tay hoặc phác thảo cơ khí trên giấy:
    1. QUAN TRỌNG NHẤT: Đọc tất cả các chữ số kích thước (dimensions) được ghi chú trên tờ giấy (ví dụ: 100mm, 50mm, R20, Ø30,...).
    2. Xác định chính xác loại hình vật thể: "RECTANGLE" (Hình chữ nhật), "CIRCLE" (Hình tròn), "REGULAR_POLYGON" (Đa giác đều), "ARC" (Cung tròn), hoặc "POLYGON" (Hình đa giác tự do).
    3. Nếu là RECTANGLE: Đọc chiều rộng (width) và chiều cao (height) từ các thông số ghi trên giấy. Đặt origin=[0,0].
    4. Cung cấp câu chú thích rõ ràng tiếng Việt trong trường `description_vi` theo mẫu: "Hình sau xử lý là [Tên hình] ([kích thước])". 
       Ví dụ: "Hình sau xử lý là hình chữ nhật (100mm x 50mm)" hoặc "Hình sau xử lý là hình tròn (Bán kính 30mm)".

    Hãy xuất ra định dạng JSON tuân thủ hoàn toàn theo schema dưới đây.
    """

  candidate_models = [
      "gemini-2.5-flash",
      "gemini-2.0-flash",
      "gemini-1.5-flash",
  ]
  max_retries_per_model = 2

  for model_name in candidate_models:
    for attempt in range(max_retries_per_model):
      try:
        response = client.models.generate_content(
            model=model_name,
            contents=[img, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=GeminiResponseModel,
            ),
        )

        if hasattr(GeminiResponseModel, "model_validate_json"):
          validated_data = GeminiResponseModel.model_validate_json(
              response.text.strip()
          )
        else:
          raw_json = json.loads(response.text.strip())
          validated_data = GeminiResponseModel(**raw_json)

        return ShapeEngine.compile_parametric_to_geometry(
            validated_data.shapes, filename
        )

      except ValidationError as ve:
        st.error(f"❌ Lỗi Validation JSON Schema ({filename}):\n{ve}")
        return []
      except Exception as e:
        err_msg = str(e)
        if (
            "503" in err_msg
            or "UNAVAILABLE" in err_msg
            or "high demand" in err_msg
        ):
          wait_time = (attempt + 1) * 2
          st.warning(
              f"⚠️ Model {model_name} quá tải. Thử lại"
              f" {attempt + 1}/{max_retries_per_model}..."
          )
          time.sleep(wait_time)
        else:
          st.error(f"❌ Lỗi xử lý ảnh ({filename}): {e}")
          return []

  st.error("❌ Tất cả các model Gemini AI đều quá tải. Vui lòng thử lại sau!")
  return []


# ==============================================================================
# 3.1. ĐỌC DXF CHUYÊN NGHIỆP (TỰ ĐỘNG GIẢI MÃ BINARY/ASCII & EXPLODE BLOCK)
# ==============================================================================
def parse_dxf_with_ezdxf(file_upload, filename):
  shapes = []
  tmp_path = None
  try:
    # Ghi dữ liệu ra file tạm thời để ezdxf tự phát hiện cấu trúc ASCII hoặc Binary DXF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp_file:
      tmp_file.write(file_upload.getvalue())
      tmp_path = tmp_file.name

    doc = ezdxf.readfile(tmp_path)
  except Exception as ex:
    st.error(f"Lỗi đọc DXF ({filename}): {ex}")
    if tmp_path and os.path.exists(tmp_path):
      os.remove(tmp_path)
    return []

  finally:
    if tmp_path and os.path.exists(tmp_path):
      os.remove(tmp_path)

  msp = doc.modelspace()

  # Lấy danh sách các đối tượng (Tự động rã Block/INSERT ra thành các nét vẽ đơn)
  entities = []
  for e in msp:
    if e.dxftype() == "INSERT":
      try:
        entities.extend(e.virtual_entities())
      except Exception:
        pass
    else:
      entities.append(e)

  for idx, entity in enumerate(entities):
    dxftype = entity.dxftype()
    s_name = f"{filename}_{idx+1}_{dxftype}"

    if dxftype == "CIRCLE":
      center = (float(entity.dxf.center.x), float(entity.dxf.center.y))
      radius = float(entity.dxf.radius)
      shapes.append({
          "name": s_name,
          "type": "CIRCLE",
          "shape_kind": "CIRCLE",
          "center": center,
          "radius": radius,
          "description": f"Hình từ DXF: Hình tròn (R={radius:.1f}mm)",
          "process_type": "Drill",
          "tool_offset": "Center",
      })

    elif dxftype == "ARC":
      center = (float(entity.dxf.center.x), float(entity.dxf.center.y))
      radius = float(entity.dxf.radius)
      start_angle = float(entity.dxf.start_angle)
      end_angle = float(entity.dxf.end_angle)
      sa_rad, ea_rad = math.radians(start_angle), math.radians(end_angle)
      start_p = (
          center[0] + radius * math.cos(sa_rad),
          center[1] + radius * math.sin(sa_rad),
      )
      end_p = (
          center[0] + radius * math.cos(ea_rad),
          center[1] + radius * math.sin(ea_rad),
      )

      shapes.append({
          "name": s_name,
          "type": "ARC",
          "shape_kind": "ARC",
          "center": center,
          "radius": radius,
          "start_angle": start_angle,
          "end_angle": end_angle,
          "start_p": start_p,
          "end_p": end_p,
          "description": f"Hình từ DXF: Cung tròn (R={radius:.1f}mm)",
          "process_type": "Profile",
          "tool_offset": "Center",
          "cw": True,
      })

    elif dxftype == "LINE":
      pts = [
          (float(entity.dxf.start.x), float(entity.dxf.start.y)),
          (float(entity.dxf.end.x), float(entity.dxf.end.y)),
      ]
      shapes.append({
          "name": s_name,
          "type": "POLYLINE",
          "shape_kind": "POLYGON",
          "points": pts,
          "description": "Hình từ DXF: Đoạn thẳng (LINE)",
          "process_type": "Profile",
          "tool_offset": "Outside",
      })

    elif dxftype in ["LWPOLYLINE", "POLYLINE", "ELLIPSE", "SPLINE"]:
      try:
        p = path.make_path(entity)
        vertices = list(path.to_vertices(p, distance=0.05))
        pts = [(float(v.x), float(v.y)) for v in vertices]
        if len(pts) >= 2:
          shapes.append({
              "name": s_name,
              "type": "POLYLINE",
              "shape_kind": "POLYGON",
              "points": pts,
              "description": f"Hình từ DXF: Đa giác/Polyline ({len(pts)} đỉnh)",
              "process_type": "Profile",
              "tool_offset": "Outside",
          })
      except Exception:
        continue

  return shapes


# ==============================================================================
# 4. HẬU KỲ: WORK ZERO, OFFSET, POCKET, SAFETY & OPTIMIZATION
# ==============================================================================
def apply_work_zero_offset(shapes, bed_w, bed_h, work_zero):
  offset_x, offset_y = 0.0, 0.0
  if work_zero == "Top Left":
    offset_y = -bed_h
  elif work_zero == "Top Right":
    offset_x, offset_y = -bed_w, -bed_h
  elif work_zero == "Bottom Right":
    offset_x = -bed_w
  elif work_zero == "Center":
    offset_x, offset_y = -bed_w / 2.0, -bed_h / 2.0

  transformed = []
  for shape in shapes:
    ns = shape.copy()
    if "points" in shape:
      ns["points"] = [
          (p[0] + offset_x, p[1] + offset_y) for p in shape["points"]
      ]
    if "center" in shape:
      cx, cy = shape["center"]
      ns["center"] = (cx + offset_x, cy + offset_y)
    if "start_p" in shape and "end_p" in shape:
      sx, sy = shape["start_p"]
      ex, ey = shape["end_p"]
      ns["start_p"] = (sx + offset_x, sy + offset_y)
      ns["end_p"] = (ex + offset_x, ey + offset_y)

    transformed.append(ns)
  return transformed


def apply_tool_offset(pts, tool_dia, offset_type):
  if len(pts) < 3 or offset_type == "Center":
    return pts
  closed_pts = list(pts)
  if closed_pts[0] != closed_pts[-1]:
    closed_pts.append(closed_pts[0])

  poly = Polygon(closed_pts)
  if not poly.is_valid:
    poly = poly.buffer(0)

  radius = tool_dia / 2.0
  buffer_dist = radius if offset_type == "Outside" else -radius
  offset_poly = poly.buffer(buffer_dist)

  if offset_poly.is_empty:
    return pts
  if isinstance(offset_poly, Polygon):
    return list(offset_poly.exterior.coords)
  elif isinstance(offset_poly, MultiPolygon):
    return list(offset_poly.geoms[0].exterior.coords)
  return pts


def generate_pocket_toolpaths(pts, tool_dia, stepover_ratio=0.6):
  closed_pts = list(pts)
  if closed_pts[0] != closed_pts[-1]:
    closed_pts.append(closed_pts[0])

  poly = Polygon(closed_pts)
  if not poly.is_valid:
    poly = poly.buffer(0)

  step = tool_dia * stepover_ratio
  if abs(step) < 0.001:
    return []

  paths = []
  current_poly = poly.buffer(-tool_dia / 2.0)

  max_loops = 500
  loop_count = 0

  while not current_poly.is_empty and loop_count < max_loops:
    loop_count += 1
    if current_poly.area < 0.01:
      break

    if isinstance(current_poly, Polygon):
      paths.append(list(current_poly.exterior.coords))
      current_poly = current_poly.buffer(-step)
    elif isinstance(current_poly, MultiPolygon):
      for p in current_poly.geoms:
        if p.area >= 0.01:
          paths.append(list(p.exterior.coords))
      current_poly = current_poly.buffer(-step)
    else:
      break

  return paths


def check_safety_and_collisions(shapes, bed_w, bed_h):
  warnings = []
  polygons = []

  for s in shapes:
    pts = []
    if "points" in s:
      pts = s["points"]
    elif s["type"] in ["CIRCLE", "DRILL"]:
      cx, cy = s["center"]
      r = s.get("radius", 5)
      pts = [
          (cx + r * math.cos(a), cy + r * math.sin(a))
          for a in np.linspace(0, 2 * math.pi, 16)
      ]

    if pts:
      xs, ys = [p[0] for p in pts], [p[1] for p in pts]
      if min(xs) < 0 or max(xs) > bed_w or min(ys) < 0 or max(ys) > bed_h:
        warnings.append(
            f"⚠️ Chi tiết '{s['name']}' vượt khổ bàn phôi ({bed_w}x{bed_h}mm)."
        )

      if len(pts) >= 3:
        poly = Polygon(pts)
        if poly.is_valid:
          for existing in polygons:
            if poly.intersects(existing) and not poly.touches(existing):
              warnings.append(f"🚨 Phát hiện va chạm tại '{s['name']}'.")
          polygons.append(poly)
  return warnings


def optimize_toolpath_order(shapes):
  if not shapes:
    return []

  def get_start(s):
    if "points" in s and s["points"]:
      return s["points"][0]
    elif "center" in s:
      return s["center"]
    elif "start_p" in s:
      return s["start_p"]
    return (0.0, 0.0)

  unvisited = shapes.copy()
  optimized = []
  curr = (0.0, 0.0)

  while unvisited:
    nearest_idx = 0
    min_dist = float("inf")
    for idx, s in enumerate(unvisited):
      sp = get_start(s)
      dist = math.hypot(sp[0] - curr[0], sp[1] - curr[1])
      if dist < min_dist:
        min_dist = dist
        nearest_idx = idx
    sel = unvisited.pop(nearest_idx)
    optimized.append(sel)
    curr = get_start(sel)
  return optimized


# ==============================================================================
# 5. G-CODE COMPILER
# ==============================================================================
def compile_shape_to_gcode(
    s,
    tool_dia,
    feed,
    plunge,
    target_z,
    step_down,
    arc_mode="IJK",
    lead_in_type="None",
):
  lines = []
  proc_type = s.get("process_type", "Profile")
  offset_type = s.get("tool_offset", "Outside")
  total_depth = abs(target_z)
  num_passes = (
      math.ceil(total_depth / abs(step_down)) if step_down != 0 else 1
  )
  z_levels = [
      -min((i + 1) * abs(step_down), total_depth) for i in range(num_passes)
  ]

  lines.append(f"(--- TOOLPATH: {s['name']} | TYPE: {proc_type} ---)")

  if s["type"] == "ARC":
    cx, cy = s["center"]
    sp, ep = s["start_p"], s["end_p"]
    r_val = s.get("radius", 10.0)
    g_cmd = "G2" if s.get("cw", True) else "G3"

    lines.append(f"G0 X{sp[0]:.3f} Y{sp[1]:.3f}")
    for z in z_levels:
      lines.append(f"G1 Z{z:.3f} F{plunge}")
      if arc_mode == "R Radius":
        lines.append(f"{g_cmd} X{ep[0]:.3f} Y{ep[1]:.3f} R{r_val:.3f} F{feed}")
      else:
        i_val = cx - sp[0]
        j_val = cy - sp[1]
        lines.append(
            f"{g_cmd} X{ep[0]:.3f} Y{ep[1]:.3f} I{i_val:.3f} J{j_val:.3f} F{feed}"
        )

  elif proc_type == "Drill" or s["type"] == "CIRCLE":
    cx, cy = s["center"]
    lines.append(f"G0 X{cx:.3f} Y{cy:.3f}")
    for z in z_levels:
      lines.append(f"G81 X{cx:.3f} Y{cy:.3f} Z{z:.3f} R2.000 F{plunge}")
    lines.append("G80")

  elif proc_type == "Pocket" and "points" in s:
    pocket_paths = generate_pocket_toolpaths(s["points"], tool_dia)
    for path_pts in pocket_paths:
      if not path_pts:
        continue
      lines.append(f"G0 X{path_pts[0][0]:.3f} Y{path_pts[0][1]:.3f}")
      for z in z_levels:
        lines.append(f"G1 Z{z:.3f} F{plunge}")
        for p in path_pts[1:]:
          lines.append(f"G1 X{p[0]:.3f} Y{p[1]:.3f} F{feed}")

  elif "points" in s:
    actual_pts = (
        apply_tool_offset(s["points"], tool_dia, offset_type)
        if proc_type == "Profile"
        else s["points"]
    )
    p0 = actual_pts[0]
    p1 = actual_pts[1] if len(actual_pts) > 1 else p0

    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    dist = math.hypot(dx, dy)
    lead_len = 5.0

    if dist > 0:
      nx, ny = -dy / dist, dx / dist
      lead_start = (p0[0] + nx * lead_len, p0[1] + ny * lead_len)
    else:
      lead_start = (p0[0] - lead_len, p0[1])

    if lead_in_type == "Linear":
      lines.append(f"G0 X{lead_start[0]:.3f} Y{lead_start[1]:.3f}")
    else:
      lines.append(f"G0 X{p0[0]:.3f} Y{p0[1]:.3f}")

    for z in z_levels:
      lines.append(f"G1 Z{z:.3f} F{plunge}")
      if lead_in_type == "Linear":
        lines.append(f"G1 X{p0[0]:.3f} Y{p0[1]:.3f} F{feed}")
      for p in actual_pts[1:]:
        lines.append(f"G1 X{p[0]:.3f} Y{p[1]:.3f} F{feed}")

  lines.append("G0 Z15.000\n")
  return "\n".join(lines)


def build_full_gcode(
    shapes,
    wcs,
    spindle,
    tool_dia,
    feed,
    plunge,
    target_z,
    step_down,
    arc_mode,
    lead_in,
):
  header = [
      "(--- STREAMLIT CAM STUDIO PRO - ISO G-CODE ---)",
      "G21 G90 G17 G94",
      f"{wcs}",
      f"M3 S{spindle}",
      "G0 Z15.000\n",
  ]
  body = [
      compile_shape_to_gcode(
          s,
          tool_dia,
          feed,
          plunge,
          target_z,
          step_down,
          arc_mode,
          lead_in,
      )
      for s in shapes
  ]
  footer = ["M5 M9", f"{wcs} G0 X0.000 Y0.000", "M30"]
  return "\n".join(header + body + footer)


# ==============================================================================
# 6. GIAO DIỆN STREAMLIT & PREVIEW HIỂN THỊ CHÚ THÍCH + KÍCH THƯỚC CẠNH
# ==============================================================================
st.set_page_config(page_title="CAM Studio Pro v5", layout="wide")
st.title("⚙️ Parametric CAM Studio - Visual Dimensions & OCR Processing Engine")

if "loaded_shapes" not in st.session_state:
  st.session_state["loaded_shapes"] = []

# SIDEBAR CONFIGURATION
st.sidebar.header("🔧 Cấu Hình Gia Công & Bàn Cắt")
wcs_option = st.sidebar.selectbox(
    "Gốc WCS", ["G54", "G55", "G56", "G57", "G58", "G59"]
)
work_zero_pos = st.sidebar.selectbox(
    "Work Zero Phôi",
    ["Bottom Left", "Top Left", "Top Right", "Bottom Right", "Center"],
)

bed_width = st.sidebar.number_input("Chiều rộng bàn phôi X (mm)", value=600.0)
bed_height = st.sidebar.number_input("Chiều dài bàn phôi Y (mm)", value=400.0)

tool_diameter = st.sidebar.number_input("Đường kính dao (mm)", value=3.175)
target_depth = st.sidebar.number_input("Độ sâu cắt Z (mm)", value=-6.0)
step_down = st.sidebar.number_input("Chiều sâu mỗi Pass Z (mm)", value=2.0)

feed_rate = st.sidebar.number_input("Feedrate (mm/p)", value=1800)
plunge_rate = st.sidebar.number_input("Plunge Rate (mm/p)", value=400)
spindle_speed = st.sidebar.number_input("Spindle (RPM)", value=18000)

st.sidebar.divider()
st.sidebar.header("🎛️ Nâng Cấp CNC Controller")
arc_mode = st.sidebar.radio("Chế độ G2/G3 Arc", ["IJK", "R Radius"])
lead_in_option = st.sidebar.radio(
    "Cắt Thâm Nhập (Lead-in)", ["None", "Linear"]
)

api_key = st.sidebar.text_input("Gemini API Key (Dành cho ảnh)", type="password")

st.sidebar.divider()
st.sidebar.header("💾 Quản Lý Project (JSON)")

current_settings = {
    "wcs": wcs_option,
    "work_zero": work_zero_pos,
    "bed_width": bed_width,
    "bed_height": bed_height,
    "tool_diameter": tool_diameter,
    "target_depth": target_depth,
    "step_down": step_down,
    "feed_rate": feed_rate,
    "plunge_rate": plunge_rate,
    "spindle_speed": spindle_speed,
    "arc_mode": arc_mode,
    "lead_in": lead_in_option,
}
project_data = {
    "settings": current_settings,
    "shapes": st.session_state["loaded_shapes"],
}

st.sidebar.download_button(
    "💾 Tải Project Hiện Tại (.json)",
    data=json.dumps(project_data, indent=2),
    file_name="cam_project.json",
    mime="application/json",
)

uploaded_project = st.sidebar.file_uploader(
    "📂 Nạp File Project (.json)", type=["json"]
)
if uploaded_project is not None:
  try:
    proj = json.load(uploaded_project)
    st.session_state["loaded_shapes"] = proj.get("shapes", [])
    st.sidebar.success("✅ Nạp Project thành công!")
  except Exception as e:
    st.sidebar.error(f"Lỗi nạp Project JSON: {e}")

# SECTION 1: INPUT
st.subheader("1. 📥 INPUT: Nạp DXF hoặc Ảnh Phác Thảo Gemini AI")
input_type = st.radio(
    "Chọn định dạng đầu vào:",
    [
        "DXF (ezdxf)",
        "IMAGE/SKETCH (Gemini AI Validation & Dimension Reading)",
    ],
    horizontal=True,
)

if input_type == "DXF (ezdxf)":
  uploaded_dxfs = st.file_uploader(
      "Thả file DXF tại đây", type=["dxf"], accept_multiple_files=True
  )
  if uploaded_dxfs and st.button("🔄 Đọc dữ liệu DXF"):
    shapes = []
    for f in uploaded_dxfs:
      shapes.extend(parse_dxf_with_ezdxf(f, f.name))
    st.session_state["loaded_shapes"] = shapes
    if shapes:
      st.success(f"Đã đọc {len(shapes)} đối tượng từ DXF!")
    else:
      st.warning(
          "Không tìm thấy hình vẽ hoặc thực thể hợp lệ trong file DXF!"
      )

else:
  uploaded_imgs = st.file_uploader(
      "Thả ảnh phác thảo/bản vẽ tại đây",
      type=["png", "jpg", "jpeg"],
      accept_multiple_files=True,
  )
  if uploaded_imgs and st.button(
      "🤖 Phân Tích Gemini AI & Validated Shape Engine"
  ):
    if not api_key:
      st.error("Vui lòng nhập Gemini API Key ở Sidebar!")
    else:
      shapes = []
      with st.spinner(
          "Gemini AI đang đọc thông số ghi chú & trích xuất hình học..."
      ):
        for img in uploaded_imgs:
          shapes.extend(
              parse_image_with_gemini_ai(img.getvalue(), api_key, img.name)
          )
      st.session_state["loaded_shapes"] = shapes
      if shapes:
        st.success(f"Đã trích xuất thành công {len(shapes)} đối tượng từ ảnh!")

# SECTION 2: SHAPE ENGINE & TOOLPATH
st.divider()
col_left, col_right = st.columns([1.2, 1])

with col_left:
  st.subheader("2. 🛠️ SHAPE ENGINE & THÔNG TIN HÌNH HỌC")
  if st.session_state["loaded_shapes"]:
    if st.button("⚡ Tối ưu đường chạy dao (Optimize - Nearest Neighbor)"):
      st.session_state["loaded_shapes"] = optimize_toolpath_order(
          st.session_state["loaded_shapes"]
      )
      st.success("Đã tối ưu thứ tự gia công!")

    proc_options = ["Profile", "Pocket", "Drill"]
    off_options = ["Outside", "Inside", "Center"]

    for idx, s in enumerate(st.session_state["loaded_shapes"]):
      st.markdown(f"**📌 {s['name']}**")
      st.info(
          "💡 **Chú thích từ AI/DXF:**"
          f" {s.get('description', 'Chưa có chú thích')}"
      )

      c1, c2 = st.columns([1, 1])
      with c1:
        s["process_type"] = st.selectbox(
            "Kiểu cắt", proc_options, key=f"proc_{idx}", index=0
        )
      with c2:
        s["tool_offset"] = st.selectbox(
            "Offset dao", off_options, key=f"off_{idx}", index=0
        )
      st.divider()

with col_right:
  st.subheader("3. 👁️ Dashboard Preview CAD (Có Chú Thích & Kích Thước Cạnh)")
  transformed_shapes = apply_work_zero_offset(
      st.session_state["loaded_shapes"], bed_width, bed_height, work_zero_pos
  )

  warnings = check_safety_and_collisions(
      transformed_shapes, bed_width, bed_height
  )
  if warnings:
    for w in warnings:
      st.error(w)
  else:
    st.info("✅ Kiểm tra an toàn: Bàn phôi OK.")

  fig, ax = plt.subplots(figsize=(7, 6))
  ax.set_aspect("equal")
  ax.grid(True, linestyle="--", alpha=0.5)
  ax.axhline(0, color="red", linewidth=1)
  ax.axvline(0, color="red", linewidth=1)

  # VẼ VẬT THỂ & CHÚ THÍCH KÍCH THƯỚC TRỰC QUAN
  for s in transformed_shapes:
    desc_text = s.get("description", "")

    if "points" in s and len(s["points"]) >= 2:
      pts = np.array(s["points"])
      ax.plot(pts[:, 0], pts[:, 1], "b-", linewidth=2)

      # Tính và vẽ kích thước từng cạnh
      num_pts = len(pts)
      for i in range(num_pts - 1):
        p1, p2 = pts[i], pts[i + 1]
        length = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        mid_x = (p1[0] + p2[0]) / 2.0
        mid_y = (p1[1] + p2[1]) / 2.0

        # Hiển thị độ dài cạnh bằng nhãn
        if length > 0.1:
          ax.text(
              mid_x,
              mid_y,
              f"{length:.1f}mm",
              fontsize=8,
              color="darkred",
              fontweight="bold",
              bbox=dict(
                  boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.6
              ),
          )

      # Chú thích Tên / Loại hình ở trọng tâm
      cx = np.mean(pts[:, 0])
      cy = np.mean(pts[:, 1])
      ax.text(
          cx,
          cy,
          f"📝 {desc_text}",
          fontsize=9,
          color="black",
          fontweight="bold",
          ha="center",
          bbox=dict(
              boxstyle="square,pad=0.3",
              facecolor="white",
              edgecolor="blue",
              alpha=0.8,
          ),
      )

    elif s["type"] == "CIRCLE":
      cx, cy = s["center"]
      r = s["radius"]
      circle = plt.Circle((cx, cy), r, color="g", fill=False, linewidth=2)
      ax.add_patch(circle)

      # Vẽ chú thích Bán kính & Chú thích hình
      ax.plot([cx, cx + r], [cy, cy], "g--")
      ax.text(
          cx + r / 2,
          cy + 2,
          f"R = {r:.1f}mm",
          color="green",
          fontweight="bold",
          fontsize=8,
      )
      ax.text(
          cx,
          cy,
          f"📝 {desc_text}",
          fontsize=9,
          color="black",
          fontweight="bold",
          ha="center",
          bbox=dict(
              boxstyle="square,pad=0.3",
              facecolor="white",
              edgecolor="green",
              alpha=0.8,
          ),
      )

    elif s["type"] == "ARC":
      cx, cy = s["center"]
      r = s.get("radius", 10.0)
      sa = s.get("start_angle", 0.0)
      ea = s.get("end_angle", 90.0)

      arc_patch = MplArc(
          (cx, cy),
          r * 2,
          r * 2,
          angle=0,
          theta1=sa,
          theta2=ea,
          color="magenta",
          linewidth=2,
      )
      ax.add_patch(arc_patch)
      ax.text(
          cx,
          cy,
          f"📝 {desc_text} (R={r:.1f}mm)",
          color="magenta",
          fontweight="bold",
          fontsize=8,
      )

  st.pyplot(fig)

# SECTION 3: EXPORT ISO G-CODE
st.divider()
st.subheader("4. 🚀 Xuất G-Code ISO")
if st.session_state["loaded_shapes"]:
  gcode_text = build_full_gcode(
      transformed_shapes,
      wcs_option,
      spindle_speed,
      tool_diameter,
      feed_rate,
      plunge_rate,
      target_depth,
      step_down,
      arc_mode,
      lead_in_option,
  )
  st.download_button(
      "💾 Tải File ISO G-Code (.nc)",
      data=gcode_text,
      file_name="OUTPUT_PROGRAM.nc",
      mime="text/plain",
  )
