import streamlit as st
import numpy as np
import cv2
import ezdxf
from ezdxf import path
import math
import io
from PIL import Image
import matplotlib.pyplot as plt

from shapely.geometry import Polygon, LineString, Point, MultiPolygon

# ==============================================================================
# 1. BÀN CẮT VÀ CHUYỂN ĐỔI GỐC TỌA ĐỘ (WORK ZERO ALIGNMENT)
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

    transformed_shapes = []
    for shape in shapes:
        new_shape = shape.copy()
        if "points" in shape:
            new_shape["points"] = [(p[0] + offset_x, p[1] + offset_y) for p in shape["points"]]
        if "center" in shape:
            cx, cy = shape["center"]
            new_shape["center"] = (cx + offset_x, cy + offset_y)
        if "start_p" in shape and "end_p" in shape:
            sx, sy = shape["start_p"]
            ex, ey = shape["end_p"]
            new_shape["start_p"] = (sx + offset_x, sy + offset_y)
            new_shape["end_p"] = (ex + offset_x, ey + offset_y)
        transformed_shapes.append(new_shape)
        
    return transformed_shapes

# ==============================================================================
# 2. XỬ LÝ HÌNH HỌC DXF
# ==============================================================================
def parse_dxf_geometry_v2(file_bytes, filename):
    try:
        content = file_bytes.getvalue().decode('utf-8', errors='ignore')
        doc = ezdxf.read(io.StringIO(content))
    except Exception:
        try:
            doc = ezdxf.read(io.BytesIO(file_bytes.getvalue()))
        except Exception as ex:
            st.error(f"Lỗi đọc file DXF ({filename}): {ex}")
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
# 3. TOOL OFFSET & POCKET PASSES
# ==============================================================================
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
    paths = []
    current_poly = poly.buffer(-tool_dia / 2.0)

    while not current_poly.is_empty:
        if isinstance(current_poly, Polygon):
            paths.append(list(current_poly.exterior.coords))
            current_poly = current_poly.buffer(-step)
        elif isinstance(current_poly, MultiPolygon):
            for p in current_poly.geoms:
                paths.append(list(p.exterior.coords))
            current_poly = current_poly.buffer(-step)
        else:
            break
    return paths

# ==============================================================================
# 4. TỐI ƯU THỨ TỰ CHẠY DAO (TSP NEAREST NEIGHBOR)
# ==============================================================================
def optimize_toolpath_order(shapes):
    if not shapes: 
        return []
    
    def get_start_point(s):
        if "points" in s and s["points"]: 
            return s["points"][0]
        elif "center" in s: 
            return s["center"]
        elif "start_p" in s: 
            return s["start_p"]
        return (0.0, 0.0)

    unvisited = shapes.copy()
    optimized = []
    current_pos = (0.0, 0.0)

    while unvisited:
        nearest_idx = 0
        min_dist = float('inf')

        for idx, s in enumerate(unvisited):
            sp = get_start_point(s)
            dist = math.hypot(sp[0] - current_pos[0], sp[1] - current_pos[1])
            if dist < min_dist:
                min_dist = dist
                nearest_idx = idx

        selected_shape = unvisited.pop(nearest_idx)
        optimized.append(selected_shape)
        current_pos = get_start_point(selected_shape)

    return optimized

# ==============================================================================
# 5. KIỂM TRA VA CHẠM VÀ VƯỢT KHỔ BÀN CẮT
# ==============================================================================
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
            pts = [(cx + r*math.cos(a), cy + r*math.sin(a)) for a in np.linspace(0, 2*math.pi, 16)]

        if pts:
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            if min(xs) < 0 or max(xs) > bed_w or min(ys) < 0 or max(ys) > bed_h:
                warnings.append(f"⚠️ Chi tiết '{s['name']}' vượt khổ bàn ({bed_w}x{bed_h}mm).")

            if len(pts) >= 3:
                poly = Polygon(pts)
                if poly.is_valid:
                    for existing_poly in polygons:
                        if poly.intersects(existing_poly) and not poly.touches(existing_poly):
                            warnings.append(f"🚨 Phát hiện va chạm tại '{s['name']}'.")
                    polygons.append(poly)

    return warnings

# ==============================================================================
# 6. TRÌNH BIÊN DỊCH G-CODE ISO
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
        lines.append("G80 (Cancel Drill Cycle)")

    elif s["type"] == "ARC":
        cx, cy = s["center"]
        sp, ep = s["start_p"], s["end_p"]
        i_val, j_val = cx - sp[0], cy - sp[1]
        g_cmd = "G2" if s.get("cw", True) else "G3"
        
        lines.append(f"G0 X{sp[0]:.3f} Y{sp[1]:.3f}")
        for z in z_levels:
            lines.append(f"G1 Z{z:.3f} F{plunge}")
            lines.append(f"{g_cmd} X{ep[0]:.3f} Y{ep[1]:.3f} I{i_val:.3f} J{j_val:.3f} F{feed}")

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

    lines.append("G0 Z15.000 (Safe Height Retract)\n")
    return "\n".join(lines)

def build_full_gcode_program(shapes, wcs, spindle, tool_dia, feed, plunge, target_z, step_down):
    header = [
        "(--- STREAMLIT CAM STUDIO PRO - ISO G-CODE ---)",
        "G21 G90 G17 G94",
        f"{wcs} (Work Coordinate System)",
        f"M3 S{spindle} (Spindle ON CW)",
        "G0 Z15.000",
        "G4 P2.0\n"
    ]
    
    body = [compile_shape_to_gcode(s, tool_dia, feed, plunge, target_z, step_down) for s in shapes]

    footer = [
        "(--- PROGRAM END ---)",
        "G0 Z15.000",
        "M5 M9",
        f"{wcs} G0 X0.000 Y0.000",
        "M30"
    ]
    return "\n".join(header + body + footer)

# ==============================================================================
# 7. GIAO DIỆN STREAMLIT
# ==============================================================================
st.set_page_config(page_title="Streamlit CAM Studio Pro v2", layout="wide")
st.title("⚙️ Streamlit CAM Studio Pro - Full Suite CNC Toolpath")

# SIDEBAR CONFIGURATION
st.sidebar.header("🔧 Cấu Hình Máy & Công Nghệ Gia Công")
wcs_option = st.sidebar.selectbox("Gốc tọa độ WCS", ["G54", "G55", "G56", "G57", "G58", "G59"])
work_zero_pos = st.sidebar.selectbox("Vị trí Work Zero trên bàn phôi", ["Bottom Left", "Top Left", "Top Right", "Bottom Right", "Center"])

bed_width = st.sidebar.number_input("Chiều rộng bàn cắt X (mm)", value=600.0, step=50.0)
bed_height = st.sidebar.number_input("Chiều dài bàn cắt Y (mm)", value=400.0, step=50.0)

tool_diameter = st.sidebar.number_input("Đường kính dao Phay (mm)", value=3.175, step=0.1)
target_depth = st.sidebar.number_input("Độ sâu cắt Z (mm)", value=-6.0, step=0.5)
step_down = st.sidebar.number_input("Chiều sâu mỗi pass Z (mm)", value=2.0, step=0.5)

feed_rate = st.sidebar.number_input("Tốc độ cắt Feedrate (mm/p)", value=1800, step=100)
plunge_rate = st.sidebar.number_input("Tốc độ cắm dao Plunge (mm/p)", value=400, step=50)
spindle_speed = st.sidebar.number_input("Tốc độ Trục chính Spindle (RPM)", value=18000, step=1000)

if "loaded_shapes" not in st.session_state:
    st.session_state["loaded_shapes"] = []

# WORKSPACE: FILE UPLOADER
st.subheader("1. 📥 Nạp File DXF")
uploaded_files = st.file_uploader(
    "Thả hoặc chọn nhiều file DXF tại đây", 
    type=["dxf"], 
    accept_multiple_files=True
)

if uploaded_files and st.button("🔄 Đọc dữ liệu các File đã tải lên"):
    all_shapes = []
    for f in uploaded_files:
        if f.name.lower().endswith(".dxf"):
            parsed = parse_dxf_geometry_v2(f, f.name)
            all_shapes.extend(parsed)
    st.session_state["loaded_shapes"] = all_shapes
    st.success(f"Đã trích xuất thành công {len(all_shapes)} đối tượng hình học!")

st.divider()
col_left, col_right = st.columns([1.2, 1])

with col_left:
    st.subheader("2. 🔀 Sắp Xếp Thứ Tự Cắt & Cấu Hình Toolpath")
    
    if st.session_state["loaded_shapes"]:
        if st.button("⚡ Tối ưu đường chạy dao ngắn nhất (Nearest Neighbor)"):
            st.session_state["loaded_shapes"] = optimize_toolpath_order(st.session_state["loaded_shapes"])
            st.success("Đã tối ưu thứ tự gia công!")

        # Sắp xếp danh sách an toàn không dùng thư viện ngoài
        all_names = [s["name"] for s in st.session_state["loaded_shapes"]]
        selected_order = st.multiselect(
            "Chọn/Sắp xếp thứ tự cắt theo ý muốn (Chọn lần lượt):",
            options=all_names,
            default=all_names
        )

        if selected_order:
            reordered_shapes = []
            for name in selected_order:
                for s in st.session_state["loaded_shapes"]:
                    if s["name"] == name:
                        reordered_shapes.append(s)
                        break
            st.session_state["loaded_shapes"] = reordered_shapes

        st.write("🛠️ **Cấu hình Kiểu gia công & Tool Offset:**")
        proc_options = ["Profile", "Pocket", "Drill", "Engrave"]
        off_options = ["Outside", "Inside", "Center"]

        for idx, s in enumerate(st.session_state["loaded_shapes"]):
            c1, c2, c3 = st.columns([2, 1.5, 1.5])
            with c1: 
                st.caption(f"**{s['name']}**")
            with c2: 
                curr_proc = s.get("process_type", "Profile")
                proc_idx = proc_options.index(curr_proc) if curr_proc in proc_options else 0
                s["process_type"] = st.selectbox("Kiểu cắt", proc_options, key=f"proc_{idx}", index=proc_idx)
            with c3:
                curr_off = s.get("tool_offset", "Outside")
                off_idx = off_options.index(curr_off) if curr_off in off_options else 0
                s["tool_offset"] = st.selectbox("Offset dao", off_options, key=f"off_{idx}", index=off_idx)

with col_right:
    st.subheader("3. 👁️ Mô Phỏng Bàn Cắt & Cảnh Báo An Toàn")
    
    transformed_shapes = apply_work_zero_offset(st.session_state["loaded_shapes"], bed_width, bed_height, work_zero_pos)
    
    safety_warnings = check_safety_and_collisions(transformed_shapes, bed_width, bed_height)
    if safety_warnings:
        for w in safety_warnings: 
            st.error(w)
    else:
        st.info("✅ Kiểm tra an toàn: Không phát hiện va chạm hoặc vượt khổ bàn cắt.")

    fig, ax = plt.subplots(figsize=(6, 5))
    
    if work_zero_pos == "Bottom Left":
        ax.set_xlim(0, bed_width); ax.set_ylim(0, bed_height)
    elif work_zero_pos == "Center":
        ax.set_xlim(-bed_width/2, bed_width/2); ax.set_ylim(-bed_height/2, bed_height/2)
    else:
        ax.set_xlim(-bed_width, bed_width); ax.set_ylim(-bed_height, bed_height)

    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.axhline(0, color='red', linewidth=1); ax.axvline(0, color='red', linewidth=1)
    ax.plot(0, 0, 'rX', markersize=10, label=f"Work Zero {wcs_option}")

    for s in transformed_shapes:
        if "points" in s:
            pts = np.array(s["points"])
            ax.plot(pts[:, 0], pts[:, 1], 'b-', linewidth=1.2)
        elif s["type"] == "CIRCLE":
            cx, cy = s["center"]
            circle_patch = plt.Circle((cx, cy), s["radius"], color='g', fill=False, linewidth=1.2)
            ax.add_patch(circle_patch)
        elif s["type"] == "ARC":
            sp, ep = s["start_p"], s["end_p"]
            ax.plot([sp[0], ep[0]], [sp[1], ep[1]], 'm--', linewidth=1.2)

    ax.legend(loc="upper right")
    st.pyplot(fig)

# BOTTOM SECTION: G-CODE GENERATION & DOWNLOAD
st.divider()
st.subheader("4. 🚀 Xuất Chương Trình G-Code ISO")

if st.session_state["loaded_shapes"]:
    c_btn1, c_btn2 = st.columns(2)
    
    with c_btn1:
        full_gcode = build_full_gcode_program(
            transformed_shapes, wcs_option, spindle_speed, tool_diameter, 
            feed_rate, plunge_rate, target_depth, step_down
        )
        st.download_button(
            "💾 Tải File G-Code TỔNG (.nc)", 
            data=full_gcode, file_name="FULL_PROGRAM.nc", mime="text/plain"
        )

    with c_btn2:
        st.write("📦 **Tải file G-Code lẻ từng chi tiết:**")
        for s in transformed_shapes:
            single_gcode = build_full_gcode_program(
                [s], wcs_option, spindle_speed, tool_diameter, 
                feed_rate, plunge_rate, target_depth, step_down
            )
            st.download_button(
                f"💾 File: {s['name']}.nc", 
                data=single_gcode, file_name=f"{s['name']}.nc", mime="text/plain"
            )
