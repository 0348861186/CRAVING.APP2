import streamlit as st
import numpy as np
import cv2
import ezdxf
from ezdxf import path
import math
import io
import json
import re
from PIL import Image
import matplotlib.pyplot as plt
from google import genai
from google.genai import types

from shapely.geometry import Polygon, LineString, Point, MultiPolygon

# ==============================================================================
# 1. SHAPE ENGINE: DỰNG TỌA ĐỘ TỪ PARAMETRIC SHAPE (TOÁN HỌC CHÍNH XÁC 100%)
# ==============================================================================
class ShapeEngine:
    @staticmethod
    def create_regular_polygon(center, radius, sides, start_angle=90.0):
        """Tạo tọa độ chính xác cho Đa giác đều (Ngũ giác, Lục giác, Tam giác...)"""
        cx, cy = center
        pts = []
        angle_step = 360.0 / sides
        for i in range(sides):
            deg = start_angle + i * angle_step
            rad = math.radians(deg)
            x = cx + radius * math.cos(rad)
            y = cy + radius * math.sin(rad)
            pts.append((round(x, 4), round(y, 4)))
        pts.append(pts[0]) # Khép kín
        return pts

    @staticmethod
    def create_rectangle(width, height, origin=(0.0, 0.0)):
        """Tạo tọa độ chuẩn cho Hình chữ nhật / Hình vuông"""
        ox, oy = origin
        return [
            (ox, oy),
            (ox + width, oy),
            (ox + width, oy + height),
            (ox, oy + height),
            (ox, oy)
        ]

    @classmethod
    def compile_parametric_to_geometry(cls, parametric_data, filename=""):
        """
        Nhận vào JSON Parametric Shape và quy đổi thành đối tượng Geometry 2D
        """
        compiled_shapes = []
        for idx, item in enumerate(parametric_data):
            s_name = f"{filename}_{item.get('name', f'shape_{idx+1}')}"
            stype = str(item.get("shape_type", "POLYGON")).upper()

            if stype == "RECTANGLE":
                w = float(item.get("width", 100.0))
                h = float(item.get("height", 100.0))
                ox, oy = item.get("origin", [0.0, 0.0])
                pts = cls.create_rectangle(w, h, (ox, oy))
                compiled_shapes.append({
                    "name": s_name,
                    "type": "POLYLINE",
                    "points": pts,
                    "process_type": "Profile",
                    "tool_offset": "Outside"
                })

            elif stype == "CIRCLE":
                cx, cy = item.get("center", [0.0, 0.0])
                compiled_shapes.append({
                    "name": s_name,
                    "type": "CIRCLE",
                    "center": (float(cx), float(cy)),
                    "radius": float(item.get("radius", 10.0)),
                    "process_type": "Drill",
                    "tool_offset": "Center"
                })

            elif stype == "REGULAR_POLYGON":
                sides = int(item.get("sides", 5)) # Mặc định ngũ giác
                cx, cy = item.get("center", [0.0, 0.0])
                
                if "radius" in item:
                    radius = float(item["radius"])
                elif "side_length" in item:
                    side_len = float(item["side_length"])
                    radius = side_len / (2 * math.sin(math.pi / sides))
                else:
                    radius = 50.0

                pts = cls.create_regular_polygon((cx, cy), radius, sides)
                compiled_shapes.append({
                    "name": s_name,
                    "type": "POLYLINE",
                    "points": pts,
                    "process_type": "Profile",
                    "tool_offset": "Outside"
                })

            else:  # POLYGON tự do
                pts = [(float(p[0]), float(p[1])) for p in item.get("points", [])]
                if pts and pts[0] != pts[-1]:
                    pts.append(pts[0])
                compiled_shapes.append({
                    "name": s_name,
                    "type": "POLYLINE",
                    "points": pts,
                    "process_type": "Profile",
                    "tool_offset": "Outside"
                })

        return compiled_shapes

# ==============================================================================
# 2. INPUT PARSERS: GEMINI AI (IMAGE) & EZDXF (DXF)
# ==============================================================================
def parse_image_with_gemini_ai(image_bytes, api_key, filename):
    """
    Trích xuất PARAMETRIC SHAPE từ ảnh bằng Gemini AI
    """
    try:
        client = genai.Client(api_key=api_key)
        img = Image.open(io.BytesIO(image_bytes))

        prompt = """
        Bạn là kỹ sư CAD/CAM. Hãy trích xuất THAM SỐ HÌNH HỌC (Parametric Shapes) từ ảnh phác thảo:

        QUY TẮC TRÍCH XUẤT:
        1. HÌNH CHỮ NHẬT / VUÔNG:
           'shape_type': 'RECTANGLE', 'width': W (mm), 'height': H (mm), 'origin': [x, y]
        2. HÌNH TRÒN / LỖ KHOAN:
           'shape_type': 'CIRCLE', 'center': [x, y], 'radius': R (mm)
        3. HÌNH ĐA GIÁC ĐỀU (Ngũ giác, Lục giác, Bát giác, Tam giác đều...):
           'shape_type': 'REGULAR_POLYGON', 'sides': N (số cạnh, ví dụ 5 cho ngũ giác), 'center': [x, y], 'radius': R (bán kính ngoại tiếp mm) HOẶC 'side_length': L (độ dài cạnh mm)
        4. HÌNH TỰ DO:
           'shape_type': 'POLYGON', 'points': [[x1,y1], [x2,y2]...]

        Trả về duy nhất mảng JSON hợp lệ:
        [
            {
                "name": "Ngu_Giac_Deu",
                "shape_type": "REGULAR_POLYGON",
                "sides": 5,
                "center": [0.0, 0.0],
                "radius": 50.0
            }
        ]
        """

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[img, prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )

        parametric_data = json.loads(response.text.strip())
        return ShapeEngine.compile_parametric_to_geometry(parametric_data, filename)

    except Exception as e:
        st.error(f"Lỗi phân tích Gemini AI ({filename}): {e}")
        return []

def parse_dxf_with_ezdxf(file_bytes, filename):
    """
    Trích xuất dữ liệu hình học từ File DXF bằng ezdxf
    """
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
            sa_rad = math.radians(float(entity.dxf.start_angle))
            ea_rad = math.radians(float(entity.dxf.end_angle))
            start_p = (center[0] + radius * math.cos(sa_rad), center[1] + radius * math.sin(sa_rad))
            end_p = (center[0] + radius * math.cos(ea_rad), center[1] + radius * math.sin(ea_rad))

            shapes.append({
                "name": s_name, "type": "ARC", "center": center, "radius": radius,
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
# 3. WORK ZERO, OFFSET, POCKET, COLLISION & OPTIMIZATION
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
                            warnings.append(f"🚨 Phát hiện va chạm giữa các đường cắt tại '{s['name']}'.")
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
# 4. G-CODE GENERATOR
# ==============================================================================
def compile_shape_to_gcode(s, tool_dia, feed, plunge, target_z, step_down):
    lines = []
    proc_type = s.get("process_type", "Profile")
    offset_type = s.get("tool_offset", "Outside")
    total_depth = abs(target_z)
    num_passes = math.ceil(total_depth / abs(step_down)) if step_down != 0 else 1
    z_levels = [-min((i + 1) * abs(step_down), total_depth) for i in range(num_passes)]

    lines.append(f"(--- TOOLPATH: {s['name']} | TYPE: {proc_type} ---)")

    if proc_type == "Drill" or s["type"] == "CIRCLE":
        cx, cy = s["center"]
        lines.append(f"G0 X{cx:.3f} Y{cy:.3f}")
        for z in z_levels:
            lines.append(f"G81 X{cx:.3f} Y{cy:.3f} Z{z:.3f} R2.000 F{plunge}")
        lines.append("G80")

    elif proc_type == "Pocket" and "points" in s:
        pocket_paths = generate_pocket_toolpaths(s["points"], tool_dia)
        for path_pts in pocket_paths:
            if not path_pts: continue
            lines.append(f"G0 X{path_pts[0][0]:.3f} Y{path_pts[0][1]:.3f}")
            for z in z_levels:
                lines.append(f"G1 Z{z:.3f} F{plunge}")
                for p in path_pts[1:]:
                    lines.append(f"G1 X{p[0]:.3f} Y{p[1]:.3f} F{feed}")

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
# 5. GIAO DIỆN STREAMLIT
# ==============================================================================
st.set_page_config(page_title="CAM Studio Pro - Parametric Engine", layout="wide")
st.title("⚙️ Parametric Shape CAM Studio - AI & DXF Pipeline")

# SIDEBAR
st.sidebar.header("🔧 Cấu Hình Gia Công")
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

if "loaded_shapes" not in st.session_state:
    st.session_state["loaded_shapes"] = []

# SECTION 1: INPUT
st.subheader("1. 📥 INPUT: Nạp File Bản Vẽ hoặc Ảnh Phác Thảo")
input_type = st.radio("Chọn định dạng đầu vào:", ["DXF (ezdxf)", "IMAGE/SKETCH (Gemini AI)"], horizontal=True)

if input_type == "DXF (ezdxf)":
    uploaded_dxfs = st.file_uploader("Thả file DXF tại đây", type=["dxf"], accept_multiple_files=True)
    if uploaded_dxfs and st.button("🔄 Đọc dữ liệu DXF"):
        shapes = []
        for f in uploaded_dxfs:
            shapes.extend(parse_dxf_with_ezdxf(f, f.name))
        st.session_state["loaded_shapes"] = shapes
        st.success(f"Đã đọc {len(shapes)} đối tượng từ DXF!")

else:
    uploaded_imgs = st.file_uploader("Thả ảnh phác thảo/bản vẽ tay tại đây", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    if uploaded_imgs and st.button("🤖 Phân Tích Gemini AI & Parametric Engine"):
        if not api_key:
            st.error("Vui lòng nhập Gemini API Key ở Sidebar!")
        else:
            shapes = []
            with st.spinner("Gemini AI đang trích xuất Parametric Shape & Shape Engine đang tạo tọa độ 2D..."):
                for img in uploaded_imgs:
                    shapes.extend(parse_image_with_gemini_ai(img.getvalue(), api_key, img.name))
            st.session_state["loaded_shapes"] = shapes
            st.success(f"Đã tạo thành công {len(shapes)} hình học chuẩn xác từ ảnh!")

# SECTION 2: SHAPE ENGINE & HẬU KỲ
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
            with c1: st.caption(f"**{s['name']}**")
            with c2: s["process_type"] = st.selectbox("Kiểu cắt", proc_options, key=f"proc_{idx}", index=0)
            with c3: s["tool_offset"] = st.selectbox("Offset dao", off_options, key=f"off_{idx}", index=0)

with col_right:
    st.subheader("3. 👁️ Mô phỏng & Collision Warning")
    transformed_shapes = apply_work_zero_offset(st.session_state["loaded_shapes"], bed_width, bed_height, work_zero_pos)
    
    # Check Safety & Collision
    warnings = check_safety_and_collisions(transformed_shapes, bed_width, bed_height)
    if warnings:
        for w in warnings: st.error(w)
    else:
        st.info("✅ Kiểm tra an toàn: Không có va chạm hoặc vượt khổ bàn phôi.")

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

    st.pyplot(fig)

# SECTION 3: G-CODE EXPORT
st.divider()
st.subheader("4. 🚀 G-CODE Export")
if st.session_state["loaded_shapes"]:
    gcode_text = build_full_gcode(
        transformed_shapes, wcs_option, spindle_speed, tool_diameter,
        feed_rate, plunge_rate, target_depth, step_down
    )
    st.download_button("💾 Tải File G-Code (.nc)", data=gcode_text, file_name="OUTPUT_PROGRAM.nc", mime="text/plain")
