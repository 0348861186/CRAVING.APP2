import streamlit as st
import numpy as np
import cv2
import ezdxf
from ezdxf import path
import math
import io
import json
import time
from PIL import Image
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Literal

from google import genai
from google.genai import types
from shapely.geometry import Polygon, MultiPolygon

# ==============================================================================
# 1. VALIDATION SCHEMA DÙNG PYDANTIC CHO GEMINI OUTPUT
# ==============================================================================
class ParametricShapeModel(BaseModel):
    name: str = Field(default="shape", description="Tên đối tượng hình học")
    shape_type: Literal["RECTANGLE", "CIRCLE", "REGULAR_POLYGON", "ARC", "POLYGON"] = Field(
        description="Loại hình học tham số"
    )
    width: Optional[float] = Field(default=None, description="Chiều rộng (mm)")
    height: Optional[float] = Field(default=None, description="Chiều cao (mm)")
    radius: Optional[float] = Field(default=None, description="Bán kính (mm)")
    sides: Optional[int] = Field(default=None, description="Số cạnh đa giác đều")
    center: Optional[List[float]] = Field(default=None, description="Tọa độ tâm [X, Y]")
    origin: Optional[List[float]] = Field(default=None, description="Gốc tọa độ [X, Y]")
    side_length: Optional[float] = Field(default=None, description="Chiều dài 1 cạnh đa giác")
    start_angle: Optional[float] = Field(default=None, description="Góc bắt đầu cho ARC (độ)")
    end_angle: Optional[float] = Field(default=None, description="Góc kết thúc cho ARC (độ)")
    points: Optional[List[List[float]]] = Field(default=None, description="Danh sách tọa độ [[x1,y1],...]")

class GeminiResponseModel(BaseModel):
    shapes: List[ParametricShapeModel]

# ==============================================================================
# 2. SHAPE ENGINE: DỰNG TỌA ĐỘ 2D TỪ PARAMETRIC SHAPE
# ==============================================================================
class ShapeEngine:
    @staticmethod
    def create_regular_polygon(center, radius, sides, start_angle=90.0):
        cx, cy = center
        pts = []
        angle_step = 360.0 / sides
        for i in range(sides):
            deg = start_angle + i * angle_step
            rad = math.radians(deg)
            x = cx + radius * math.cos(rad)
            y = cy + radius * math.sin(rad)
            pts.append((round(x, 4), round(y, 4)))
        pts.append(pts[0])  # Khép kín
        return pts

    @staticmethod
    def create_rectangle(width, height, origin=(0.0, 0.0)):
        ox, oy = origin
        return [
            (ox, oy),
            (ox + width, oy),
            (ox + width, oy + height),
            (ox, oy + height),
            (ox, oy)
        ]

    @classmethod
    def compile_parametric_to_geometry(cls, parametric_shapes: List[ParametricShapeModel], filename=""):
        compiled_shapes = []
        for idx, item in enumerate(parametric_shapes):
            s_name = f"{filename}_{item.name}_{idx+1}"
            stype = item.shape_type

            if stype == "RECTANGLE":
                w = item.width if item.width else 100.0
                h = item.height if item.height else 100.0
                ox, oy = item.origin if item.origin and len(item.origin) >= 2 else [0.0, 0.0]
                pts = cls.create_rectangle(w, h, (ox, oy))
                compiled_shapes.append({
                    "name": s_name, "type": "POLYLINE", "points": pts,
                    "process_type": "Profile", "tool_offset": "Outside"
                })

            elif stype == "CIRCLE":
                cx, cy = item.center if item.center and len(item.center) >= 2 else [0.0, 0.0]
                compiled_shapes.append({
                    "name": s_name, "type": "CIRCLE", "center": (float(cx), float(cy)),
                    "radius": float(item.radius if item.radius else 10.0),
                    "process_type": "Drill", "tool_offset": "Center"
                })

            elif stype == "REGULAR_POLYGON":
                sides = item.sides if item.sides and item.sides >= 3 else 5
                cx, cy = item.center if item.center and len(item.center) >= 2 else [0.0, 0.0]
                
                if item.radius:
                    radius = float(item.radius)
                elif item.side_length:
                    radius = float(item.side_length) / (2 * math.sin(math.pi / sides))
                else:
                    radius = 50.0

                pts = cls.create_regular_polygon((cx, cy), radius, sides)
                compiled_shapes.append({
                    "name": s_name, "type": "POLYLINE", "points": pts,
                    "process_type": "Profile", "tool_offset": "Outside"
                })

            elif stype == "ARC":
                cx, cy = item.center if item.center and len(item.center) >= 2 else [0.0, 0.0]
                radius = float(item.radius if item.radius else 20.0)
                sa_deg = float(item.start_angle if item.start_angle is not None else 0.0)
                ea_deg = float(item.end_angle if item.end_angle is not None else 90.0)
                
                sa_rad, ea_rad = math.radians(sa_deg), math.radians(ea_deg)
                start_p = (cx + radius * math.cos(sa_rad), cy + radius * math.sin(sa_rad))
                end_p = (cx + radius * math.cos(ea_rad), cy + radius * math.sin(ea_rad))

                compiled_shapes.append({
                    "name": s_name, "type": "ARC", "center": (cx, cy), "radius": radius,
                    "start_angle": sa_deg, "end_angle": ea_deg,
                    "start_p": start_p, "end_p": end_p,
                    "process_type": "Profile", "tool_offset": "Center", "cw": True
                })

            else:  # POLYGON
                raw_pts = item.points if item.points else []
                pts = [(float(p[0]), float(p[1])) for p in raw_pts if len(p) >= 2]
                if pts:
                    if pts[0] != pts[-1]: pts.append(pts[0])
                    compiled_shapes.append({
                        "name": s_name, "type": "POLYLINE", "points": pts,
                        "process_type": "Profile", "tool_offset": "Outside"
                    })

        return compiled_shapes

# ==============================================================================
# 3. INPUT PARSERS (GEMINI + AUTO RETRY 503 & EZDXF)
# ==============================================================================
def parse_image_with_gemini_ai(image_bytes, api_key, filename):
    """
    Trích xuất Parametric Shape từ ảnh bằng Gemini AI
    Bổ sung: Auto-Retry khi gặp 503 & Fallback Model dự phòng + Pydantic Validation
    """
    if not api_key:
        st.error("Vui lòng nhập Gemini API Key!")
        return []

    client = genai.Client(api_key=api_key)
    img = Image.open(io.BytesIO(image_bytes))

    prompt = """
    Bạn là kỹ sư CAD/CAM chuyên nghiệp. Hãy phân tích bản vẽ/ảnh phác thảo và trích xuất các thông số hình học:
    Trả về JSON đúng cấu trúc schema sau:
    {
      "shapes": [
        {
          "name": "ten_hinh",
          "shape_type": "RECTANGLE" | "CIRCLE" | "REGULAR_POLYGON" | "ARC" | "POLYGON",
          "width": 100.0,
          "height": 50.0,
          "radius": 25.0,
          "sides": 5,
          "center": [0.0, 0.0],
          "origin": [0.0, 0.0],
          "start_angle": 0.0,
          "end_angle": 90.0,
          "points": [[0,0], [10,0], [10,10]]
        }
      ]
    }
    """

    candidate_models = ['gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-1.5-flash']
    max_retries_per_model = 3

    for model_name in candidate_models:
        for attempt in range(max_retries_per_model):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[img, prompt],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=GeminiResponseModel
                    )
                )

                # Validation dữ liệu trả về với Pydantic (Xử lý được cả pydantic v1/v2)
                if hasattr(GeminiResponseModel, 'model_validate_json'):
                    validated_data = GeminiResponseModel.model_validate_json(response.text.strip())
                else:
                    raw_json = json.loads(response.text.strip())
                    validated_data = GeminiResponseModel(**raw_json)

                return ShapeEngine.compile_parametric_to_geometry(validated_data.shapes, filename)

            except ValidationError as ve:
                st.error(f"❌ Lỗi Validation JSON Schema ({filename}):\n{ve}")
                return []
            except Exception as e:
                err_msg = str(e)
                if "503" in err_msg or "UNAVAILABLE" in err_msg or "high demand" in err_msg:
                    wait_time = (attempt + 1) * 2
                    st.warning(f"⚠️ Model {model_name} đang quá tải (503). Thử lại lần {attempt + 1}/{max_retries_per_model} sau {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    st.error(f"❌ Lỗi xử lý ảnh ({filename}): {e}")
                    return []

        st.warning(f"🔄 Chuyển sang model dự phòng tiếp theo...")

    st.error(f"❌ Tất cả các model Gemini AI đều đang quá tải (503). Vui lòng bấm thử lại sau giây lát!")
    return []

def parse_dxf_with_ezdxf(file_bytes, filename):
    """Trích xuất dữ liệu hình học từ File DXF bằng ezdxf"""
    try:
        content = file_bytes.getvalue().decode('utf-8', errors='ignore')
        doc = ezdxf.read(io.StringIO(content))
    except Exception:
        try:
            doc = ezdxf.read(io.BytesIO(file_bytes.getvalue()))
        except Exception as ex:
            st.error(f"Lỗi đọc DXF ({filename}): {ex}")
            return []

    msp = doc.modelspace()
    shapes = []

    for idx, entity in enumerate(msp):
        dxftype = entity.dxftype()
        s_name = f"{filename}_{idx+1}_{dxftype}"

        if dxftype == 'CIRCLE':
            center = (float(entity.dxf.center.x), float(entity.dxf.center.y))
            radius = float(entity.dxf.radius)
            shapes.append({
                "name": s_name, "type": "CIRCLE", "center": center, "radius": radius,
                "process_type": "Drill", "tool_offset": "Center"
            })

        elif dxftype == 'ARC':
            center = (float(entity.dxf.center.x), float(entity.dxf.center.y))
            radius = float(entity.dxf.radius)
            start_angle = float(entity.dxf.start_angle)
            end_angle = float(entity.dxf.end_angle)
            sa_rad, ea_rad = math.radians(start_angle), math.radians(end_angle)
            start_p = (center[0] + radius * math.cos(sa_rad), center[1] + radius * math.sin(sa_rad))
            end_p = (center[0] + radius * math.cos(ea_rad), center[1] + radius * math.sin(ea_rad))

            shapes.append({
                "name": s_name, "type": "ARC", "center": center, "radius": radius,
                "start_angle": start_angle, "end_angle": end_angle,
                "start_p": start_p, "end_p": end_p,
                "process_type": "Profile", "tool_offset": "Center", "cw": True
            })

        elif dxftype in ['LWPOLYLINE', 'POLYLINE', 'ELLIPSE', 'SPLINE', 'LINE']:
            try:
                p = path.make_path(entity)
                vertices = list(path.to_vertices(p, distance=0.05))
                pts = [(float(v.x), float(v.y)) for v in vertices]
                if len(pts) >= 2:
                    shapes.append({
                        "name": s_name, "type": "POLYLINE", "points": pts,
                        "process_type": "Profile", "tool_offset": "Outside"
                    })
            except Exception:
                continue

    return shapes

# ==============================================================================
# 4. HẬU KỲ: WORK ZERO, OFFSET, POCKET, COLLISION & OPTIMIZATION
# ==============================================================================
def apply_work_zero_offset(shapes, bed_w, bed_h, work_zero):
    offset_x, offset_y = 0.0, 0.0
    if work_zero == "Top Left": offset_y = -bed_h
    elif work_zero == "Top Right": offset_x, offset_y = -bed_w, -bed_h
    elif work_zero == "Bottom Right": offset_x = -bed_w
    elif work_zero == "Center": offset_x, offset_y = -bed_w / 2.0, -bed_h / 2.0

    transformed = []
    for shape in shapes:
        ns = shape.copy()
        if "points" in shape:
            ns["points"] = [(p[0] + offset_x, p[1] + offset_y) for p in shape["points"]]
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
    if len(pts) < 3 or offset_type == "Center": return pts
    closed_pts = list(pts)
    if closed_pts[0] != closed_pts[-1]: closed_pts.append(closed_pts[0])

    poly = Polygon(closed_pts)
    if not poly.is_valid: poly = poly.buffer(0)

    radius = tool_dia / 2.0
    buffer_dist = radius if offset_type == "Outside" else -radius
    offset_poly = poly.buffer(buffer_dist)

    if offset_poly.is_empty: return pts
    if isinstance(offset_poly, Polygon): return list(offset_poly.exterior.coords)
    elif isinstance(offset_poly, MultiPolygon): return list(offset_poly.geoms[0].exterior.coords)
    return pts

def generate_pocket_toolpaths(pts, tool_dia, stepover_ratio=0.6):
    closed_pts = list(pts)
    if closed_pts[0] != closed_pts[-1]: closed_pts.append(closed_pts[0])

    poly = Polygon(closed_pts)
    if not poly.is_valid: poly = poly.buffer(0)

    step = tool_dia * stepover_ratio
    paths = []
    current_poly = poly.buffer(-tool_dia / 2.0)

    while not current_poly.is_empty:
        if isinstance(current_poly, Polygon):
            paths.append(list(current_poly.exterior.coords))
            current_poly = current_poly.buffer(-step)
        elif isinstance(current_poly, MultiPolygon):
            for p in current_poly.geoms: paths.append(list(p.exterior.coords))
            current_poly = current_poly.buffer(-step)
        else: break
    return paths

def check_safety_and_collisions(shapes, bed_w, bed_h):
    warnings = []
    polygons = []

    for s in shapes:
        pts = []
        if "points" in s: pts = s["points"]
        elif s["type"] in ["CIRCLE", "DRILL"]:
            cx, cy = s["center"]
            r = s.get("radius", 5)
            pts = [(cx + r*math.cos(a), cy + r*math.sin(a)) for a in np.linspace(0, 2*math.pi, 16)]

        if pts:
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            if min(xs) < 0 or max(xs) > bed_w or min(ys) < 0 or max(ys) > bed_h:
                warnings.append(f"⚠️ Chi tiết '{s['name']}' vượt khổ bàn phôi ({bed_w}x{bed_h}mm).")

            if len(pts) >= 3:
                poly = Polygon(pts)
                if poly.is_valid:
                    for existing in polygons:
                        if poly.intersects(existing) and not poly.touches(existing):
                            warnings.append(f"🚨 Phát hiện va chạm tại '{s['name']}'.")
                    polygons.append(poly)
    return warnings

def optimize_toolpath_order(shapes):
    if not shapes: return []
    def get_start(s):
        if "points" in s and s["points"]: return s["points"][0]
        elif "center" in s: return s["center"]
        elif "start_p" in s: return s["start_p"]
        return (0.0, 0.0)

    unvisited = shapes.copy()
    optimized = []
    curr = (0.0, 0.0)

    while unvisited:
        nearest_idx = 0
        min_dist = float('inf')
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
# 5. G-CODE COMPILER WITH ARC COMMANDS (G2/G3 INTEGRATED)
# ==============================================================================
def compile_shape_to_gcode(s, tool_dia, feed, plunge, target_z, step_down):
    lines = []
    proc_type = s.get("process_type", "Profile")
    offset_type = s.get("tool_offset", "Outside")
    total_depth = abs(target_z)
    num_passes = math.ceil(total_depth / abs(step_down)) if step_down != 0 else 1
    z_levels = [-min((i + 1) * abs(step_down), total_depth) for i in range(num_passes)]

    lines.append(f"(--- TOOLPATH: {s['name']} | TYPE: {proc_type} ---)")

    # 1. ARC CUNG TRÒN (NỘI SUY G2 / G3)
    if s["type"] == "ARC":
        cx, cy = s["center"]
        sp, ep = s["start_p"], s["end_p"]
        i_val = cx - sp[0]
        j_val = cy - sp[1]
        g_cmd = "G2" if s.get("cw", True) else "G3"

        lines.append(f"G0 X{sp[0]:.3f} Y{sp[1]:.3f}")
        for z in z_levels:
            lines.append(f"G1 Z{z:.3f} F{plunge}")
            lines.append(f"{g_cmd} X{ep[0]:.3f} Y{ep[1]:.3f} I{i_val:.3f} J{j_val:.3f} F{feed}")

    # 2. DRILL / KHOAN TRÒN
    elif proc_type == "Drill" or s["type"] == "CIRCLE":
        cx, cy = s["center"]
        lines.append(f"G0 X{cx:.3f} Y{cy:.3f}")
        for z in z_levels:
            lines.append(f"G81 X{cx:.3f} Y{cy:.3f} Z{z:.3f} R2.000 F{plunge}")
        lines.append("G80")

    # 3. POCKET / PHAY HỐC
    elif proc_type == "Pocket" and "points" in s:
        pocket_paths = generate_pocket_toolpaths(s["points"], tool_dia)
        for path_pts in pocket_paths:
            if not path_pts: continue
            lines.append(f"G0 X{path_pts[0][0]:.3f} Y{path_pts[0][1]:.3f}")
            for z in z_levels:
                lines.append(f"G1 Z{z:.3f} F{plunge}")
                for p in path_pts[1:]:
                    lines.append(f"G1 X{p[0]:.3f} Y{p[1]:.3f} F{feed}")

    # 4. PROFILE / PHAY BIÊN DẠNG POLYLINE
    elif "points" in s:
        actual_pts = apply_tool_offset(s["points"], tool_dia, offset_type) if proc_type == "Profile" else s["points"]
        lines.append(f"G0 X{actual_pts[0][0]:.3f} Y{actual_pts[0][1]:.3f}")
        for z in z_levels:
            lines.append(f"G1 Z{z:.3f} F{plunge}")
            for p in actual_pts[1:]:
                lines.append(f"G1 X{p[0]:.3f} Y{p[1]:.3f} F{feed}")

    lines.append("G0 Z15.000\n")
    return "\n".join(lines)

def build_full_gcode(shapes, wcs, spindle, tool_dia, feed, plunge, target_z, step_down):
    header = [
        "(--- STREAMLIT CAM STUDIO PRO - ISO G-CODE ---)",
        "G21 G90 G17 G94",
        f"{wcs}",
        f"M3 S{spindle}",
        "G0 Z15.000\n"
    ]
    body = [compile_shape_to_gcode(s, tool_dia, feed, plunge, target_z, step_down) for s in shapes]
    footer = ["M5 M9", f"{wcs} G0 X0.000 Y0.000", "M30"]
    return "\n".join(header + body + footer)

# ==============================================================================
# 6. GIAO DIỆN STREAMLIT & QUẢN LÝ PROJECT JSON
# ==============================================================================
st.set_page_config(page_title="CAM Studio Pro v4", layout="wide")
st.title("⚙️ Parametric CAM Studio - Arc G-code, Project JSON & Validation")

if "loaded_shapes" not in st.session_state:
    st.session_state["loaded_shapes"] = []

# SIDEBAR CONFIGURATION
st.sidebar.header("🔧 Cấu Hình Gia Công & Bàn Cắt")
wcs_option = st.sidebar.selectbox("Gốc WCS", ["G54", "G55", "G56", "G57", "G58", "G59"])
work_zero_pos = st.sidebar.selectbox("Work Zero Phôi", ["Bottom Left", "Top Left", "Top Right", "Bottom Right", "Center"])

bed_width = st.sidebar.number_input("Chiều rộng bàn phôi X (mm)", value=600.0)
bed_height = st.sidebar.number_input("Chiều dài bàn phôi Y (mm)", value=400.0)

tool_diameter = st.sidebar.number_input("Đường kính dao (mm)", value=3.175)
target_depth = st.sidebar.number_input("Độ sâu cắt Z (mm)", value=-6.0)
step_down = st.sidebar.number_input("Chiều sâu mỗi Pass Z (mm)", value=2.0)

feed_rate = st.sidebar.number_input("Feedrate (mm/p)", value=1800)
plunge_rate = st.sidebar.number_input("Plunge Rate (mm/p)", value=400)
spindle_speed = st.sidebar.number_input("Spindle (RPM)", value=18000)

api_key = st.sidebar.text_input("Gemini API Key (Dành cho ảnh)", type="password")

# PROJECT SAVE & LOAD IN SIDEBAR
st.sidebar.divider()
st.sidebar.header("💾 Quản Lý Project (JSON)")

# 1. LƯU PROJECT
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
    "spindle_speed": spindle_speed
}
project_data = {
    "settings": current_settings,
    "shapes": st.session_state["loaded_shapes"]
}

project_json_str = json.dumps(project_data, indent=2)
st.sidebar.download_button(
    "💾 Tải Project Hiện Tại (.json)",
    data=project_json_str,
    file_name="cam_project.json",
    mime="application/json"
)

# 2. NẠP PROJECT
uploaded_project = st.sidebar.file_uploader("📂 Nạp File Project (.json)", type=["json"])
if uploaded_project is not None:
    try:
        proj = json.load(uploaded_project)
        st.session_state["loaded_shapes"] = proj.get("shapes", [])
        st.sidebar.success("✅ Nạp Project thành công!")
    except Exception as e:
        st.sidebar.error(f"Lỗi nạp Project JSON: {e}")

# SECTION 1: INPUT
st.subheader("1. 📥 INPUT: Nạp DXF hoặc Ảnh Phác Thảo Gemini AI")
input_type = st.radio("Chọn định dạng đầu vào:", ["DXF (ezdxf)", "IMAGE/SKETCH (Gemini AI Validation)"], horizontal=True)

if input_type == "DXF (ezdxf)":
    uploaded_dxfs = st.file_uploader("Thả file DXF tại đây", type=["dxf"], accept_multiple_files=True)
    if uploaded_dxfs and st.button("🔄 Đọc dữ liệu DXF"):
        shapes = []
        for f in uploaded_dxfs:
            shapes.extend(parse_dxf_with_ezdxf(f, f.name))
        st.session_state["loaded_shapes"] = shapes
        st.success(f"Đã đọc {len(shapes)} đối tượng từ DXF!")

else:
    uploaded_imgs = st.file_uploader("Thả ảnh phác thảo/bản vẽ tại đây", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    if uploaded_imgs and st.button("🤖 Phân Tích Gemini AI & Validated Shape Engine"):
        if not api_key:
            st.error("Vui lòng nhập Gemini API Key ở Sidebar!")
        else:
            shapes = []
            with st.spinner("Gemini AI đang trích xuất & Kiểm tra Validation Pydantic Schema..."):
                for img in uploaded_imgs:
                    shapes.extend(parse_image_with_gemini_ai(img.getvalue(), api_key, img.name))
            st.session_state["loaded_shapes"] = shapes
            if shapes:
                st.success(f"Đã tạo thành công {len(shapes)} hình học chuẩn xác từ ảnh!")

# SECTION 2: SHAPE ENGINE & TOOLPATH
st.divider()
col_left, col_right = st.columns([1.2, 1])

with col_left:
    st.subheader("2. 🛠️ SHAPE ENGINE: Cấu hình Toolpath, Offset & Pocket")
    if st.session_state["loaded_shapes"]:
        if st.button("⚡ Tối ưu đường chạy dao (Optimize - Nearest Neighbor)"):
            st.session_state["loaded_shapes"] = optimize_toolpath_order(st.session_state["loaded_shapes"])
            st.success("Đã tối ưu thứ tự gia công!")

        proc_options = ["Profile", "Pocket", "Drill"]
        off_options = ["Outside", "Inside", "Center"]

        for idx, s in enumerate(st.session_state["loaded_shapes"]):
            c1, c2, c3 = st.columns([2, 1.5, 1.5])
            with c1: st.caption(f"**{s['name']}** ({s['type']})")
            with c2: s["process_type"] = st.selectbox("Kiểu cắt", proc_options, key=f"proc_{idx}", index=0)
            with c3: s["tool_offset"] = st.selectbox("Offset dao", off_options, key=f"off_{idx}", index=0)

with col_right:
    st.subheader("3. 👁️ Mô phỏng Arc/Polyline & Collision Check")
    transformed_shapes = apply_work_zero_offset(st.session_state["loaded_shapes"], bed_width, bed_height, work_zero_pos)
    
    warnings = check_safety_and_collisions(transformed_shapes, bed_width, bed_height)
    if warnings:
        for w in warnings: st.error(w)
    else:
        st.info("✅ Kiểm tra an toàn: Không phát hiện va chạm hoặc vượt khổ bàn phôi.")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.axhline(0, color='red', linewidth=1)
    ax.axvline(0, color='red', linewidth=1)

    for s in transformed_shapes:
        if "points" in s:
            pts = np.array(s["points"])
            ax.plot(pts[:, 0], pts[:, 1], 'b-')
        elif s["type"] == "CIRCLE":
            cx, cy = s["center"]
            circle = plt.Circle((cx, cy), s["radius"], color='g', fill=False)
            ax.add_patch(circle)
        elif s["type"] == "ARC":
            sp, ep = s["start_p"], s["end_p"]
            ax.plot([sp[0], ep[0]], [sp[1], ep[1]], 'm-o', label="Arc Boundary")

    st.pyplot(fig)

# SECTION 3: EXPORT ISO G-CODE (SUPPORT G2/G3 ARC)
st.divider()
st.subheader("4. 🚀 Xuất G-Code ISO (Hỗ trợ Arc G2/G3)")
if st.session_state["loaded_shapes"]:
    gcode_text = build_full_gcode(
        transformed_shapes, wcs_option, spindle_speed, tool_diameter,
        feed_rate, plunge_rate, target_depth, step_down
    )
    st.download_button("💾 Tải File ISO G-Code (.nc)", data=gcode_text, file_name="OUTPUT_PROGRAM.nc", mime="text/plain")
