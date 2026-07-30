import streamlit as st

# ---------------------------------------------------------
# CẤU HÌNH TRANG STREAMLIT
# ---------------------------------------------------------
try:
    st.set_page_config(
        page_title="CAM 3D Relief Generator for CNC with Real AI",
        page_layout="wide",
        initial_sidebar_state="expanded"
    )
except Exception:
    pass

# ---------------------------------------------------------
# IMPORT THƯ VIỆN KHÁC
# ---------------------------------------------------------
import numpy as np
import cv2
from PIL import Image
import trimesh
import matplotlib.pyplot as plt

# Tích hợp OpenAI làm AI thật (Bạn có thể đổi sang google.generativeai nếu muốn dùng Gemini)
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# ---------------------------------------------------------
# HÀM BỔ TRỢ & GỌI AI THẬT
# ---------------------------------------------------------
def call_real_ai_advisor(api_key, layer_name, material, width, height, depth, tool_info, spindle, feed, stepover, stepdown=None):
    """Hàm gọi API OpenAI thật để tư vấn thông số kỹ thuật cho từng Layer"""
    if not api_key:
        return "⚠️ Vui lòng nhập OpenAI API Key ở thanh bên (Sidebar) để kích hoạt AI thật tư vấn."
    
    if not OPENAI_AVAILABLE:
        return "⚠️ Chưa cài đặt thư viện `openai`. Vui lòng chạy command: `pip install openai`"

    try:
        client = openai.OpenAI(api_key=api_key)
        
        prompt = f"""
        Bạn là một chuyên gia lập trình CAM và vận hành máy CNC khắc tranh 3D Relief gỗ/kim loại nhiều năm kinh nghiệm.
        Hãy đưa ra đánh giá, tư vấn và cảnh báo kỹ thuật ngắn gọn, súc tích (khoảng 3-4 dòng) cho Layer gia công sau:

        - Lớp gia công: {layer_name}
        - Vật liệu phôi: {material}
        - Kích thước bức tranh: {width}x{height} mm, Độ sâu Z: {depth} mm
        - Loại dao & đường kính: {tool_info}
        - Tốc độ Spindle: {spindle} RPM
        - Tốc độ tiến dao F: {feed} mm/min
        - Độ dịch dao ngang (Stepover): {stepover}%
        {f"- Độ sâu lát cắt (Stepdown): {stepdown} mm" if stepdown else ""}

        Yêu cầu:
        1. Phân tích xem thông số (Spindle, Feedrate, Stepover, Stepdown) trên có hợp lý cho {material} hay chưa?
        2. Cảnh báo nguy cơ (gãy dao, cháy gỗ, bề mặt thô, gồ ghề...) nếu có.
        3. Gợi ý điều chỉnh tối ưu nhất.
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini", # Hoặc gpt-4o / gpt-3.5-turbo
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=250
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Lỗi khi kết nối tới AI API: {str(e)}"

def depth_map_to_mesh(depth_img, width_mm, height_mm, max_depth_mm, work_zero="Góc Dưới Trái (Bottom-Left)"):
    """Chuyển đổi Depth Map 2D thành Mesh 3D tính theo Gốc Tọa Độ (Work Zero) chọn trước"""
    h, w = depth_img.shape
    norm_depth = (depth_img - depth_img.min()) / (depth_img.max() - depth_img.min() + 1e-6)
    
    # Tính toán tọa độ X, Y theo Work Zero
    if work_zero == "Góc Dưới Trái (Bottom-Left)":
        x = np.linspace(0, width_mm, w)
        y = np.linspace(0, height_mm, h)
    elif work_zero == "Góc Trên Trái (Top-Left)":
        x = np.linspace(0, width_mm, w)
        y = np.linspace(-height_mm, 0, h)
    elif work_zero == "Góc Dưới Phải (Bottom-Right)":
        x = np.linspace(-width_mm, 0, w)
        y = np.linspace(0, height_mm, h)
    elif work_zero == "Góc Trên Phải (Top-Right)":
        x = np.linspace(-width_mm, 0, w)
        y = np.linspace(-height_mm, 0, h)
    elif work_zero == "Tâm Phôi (Center)":
        x = np.linspace(-width_mm / 2, width_mm / 2, w)
        y = np.linspace(-height_mm / 2, height_mm / 2, h)
    else:
        x = np.linspace(0, width_mm, w)
        y = np.linspace(0, height_mm, h)

    x_grid, y_grid = np.meshgrid(x, y)
    z_grid = norm_depth * max_depth_mm

    vertices = np.column_stack((x_grid.ravel(), y_grid.ravel(), z_grid.ravel()))
    
    faces = []
    for i in range(h - 1):
        for j in range(w - 1):
            idx = i * w + j
            faces.append([idx, idx + 1, idx + w])
            faces.append([idx + 1, idx + w + 1, idx + w])
            
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return mesh, x_grid, y_grid, z_grid

def generate_layer_gcode(layer_name, x_grid, y_grid, z_grid, step_over_mm, feed_rate, spindle_speed, safe_z, tool_info):
    """Tạo G-Code riêng cho từng lớp dao"""
    gcode = [
        f"(--- G-CODE LAYER: {layer_name.upper()} ---)",
        f"(Thong so dao: {tool_info})",
        "G21 (Don vi: mm)",
        "G90 (Toa do tuyet doi)",
        f"G0 Z{safe_z:.2f} (Dua dao len Z an toan)",
        f"M3 S{spindle_speed} (Bat truc chinh {spindle_speed} RPM)"
    ]
    
    rows, cols = z_grid.shape
    pixel_y_step = max(1, int((y_grid[1, 0] - y_grid[0, 0]) * (step_over_mm / (abs(y_grid[1, 0] - y_grid[0, 0]) + 1e-5))))
    
    for i in range(0, rows, pixel_y_step):
        y_val = y_grid[i, 0]
        col_range = range(cols) if (i // pixel_y_step) % 2 == 0 else range(cols - 1, -1, -1)
        
        first_j = list(col_range)[0]
        gcode.append(f"G0 X{x_grid[i, first_j]:.3f} Y{y_val:.3f}")
        gcode.append(f"G1 Z{z_grid[i, first_j]:.3f} F{feed_rate / 2:.0f}")
        
        for j in col_range:
            gcode.append(f"G1 X{x_grid[i, j]:.3f} Y{y_val:.3f} Z{z_grid[i, j]:.3f} F{feed_rate:.0f}")
            
        gcode.append(f"G0 Z{safe_z:.2f}")

    gcode.append("M5 (Tat truc chinh)")
    gcode.append("G0 Z50 (Nang cao dao)")
    gcode.append("M30 (Ket thuc layer)")
    return "\n".join(gcode)

# ---------------------------------------------------------
# TIÊU ĐỀ
# ---------------------------------------------------------
st.title("🖼️ Hệ Thống CAM 3D Relief & AI Tư Vấn Thật Cho Máy CNC")
st.caption("Tạo Layer độc lập, Cấu hình Work Zero, Tích hợp AI LLM tư vấn thông số & Xuất G-Code riêng")

# ---------------------------------------------------------
# SIDEBAR: CẤU HÌNH & API KEY
# ---------------------------------------------------------
st.sidebar.header("🔑 Tích Hợp AI Thật")
api_key = st.sidebar.text_input("Nhập OpenAI API Key:", type="password", help="Nhập API Key để kích hoạt AI tư vấn trực tiếp từ LLM")

st.sidebar.markdown("---")
st.sidebar.header("⚙️ 1. Thông Số Phôi & Gốc Tọa Độ")
width_mm = st.sidebar.number_input("Chiều rộng (X - mm)", value=300, step=10)
height_mm = st.sidebar.number_input("Chiều cao (Y - mm)", value=400, step=10)
max_depth_mm = st.sidebar.slider("Độ sâu phù điêu (Z - mm)", min_value=3.0, max_value=50.0, value=15.0, step=0.5)
safe_z = st.sidebar.number_input("Chiều cao an toàn (Safe Z - mm)", value=10.0)

st.sidebar.markdown("---")
st.sidebar.header("📍 2. Chọn Work Zero (G54)")
work_zero_option = st.sidebar.selectbox(
    "Gốc tọa độ [0,0]:",
    [
        "Góc Dưới Trái (Bottom-Left)",
        "Góc Trên Trái (Top-Left)",
        "Góc Dưới Phải (Bottom-Right)",
        "Góc Trên Phải (Top-Right)",
        "Tâm Phôi (Center)"
    ]
)

st.sidebar.markdown("---")
st.sidebar.header("🪵 3. Vật Liệu Gia Công")
material = st.sidebar.selectbox("Loại phôi:", ["Gỗ Trắc/Hương/Cẩm (Cứng)", "Gỗ Gụ/Sồi/Tần Bì (Trung bình)", "Gỗ Mềm/MDF", "Nhôm/Đồng", "Mica/Acrylic"])

# ---------------------------------------------------------
# MAIN INTERFACE
# ---------------------------------------------------------
uploaded_img = st.file_uploader("📥 Tải lên ảnh mẫu (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_img:
    img_raw = Image.open(uploaded_img).convert("L")
    st.image(img_raw, caption="Ảnh mẫu gốc", width=250)

    # NÚT KÍCH HOẠT XỬ LÝ & KÍCH HOẠT AI (YÊU CẦU 3)
    st.markdown("---")
    if st.button("🚀 KÍCH HOẠT AI TƯ VẤN, XỬ LÝ CÁC LAYER & TẠO G-CODE", type="primary", use_column_width=True):
        with st.spinner("AI đang xử lý ảnh, dựng Mesh 3D, tính toán Work Zero và chuẩn bị các Layer..."):
            # 1. Tiền xử lý ảnh
            img_np = np.array(img_raw)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced_img = clahe.apply(img_np)
            final_depth = cv2.GaussianBlur(enhanced_img, (3, 3), 0)
            
            # 2. Tạo Mesh 3D theo Work Zero đã chọn (Yêu cầu 2)
            mesh, x_g, y_g, z_g = depth_map_to_mesh(final_depth, width_mm, height_mm, max_depth_mm, work_zero_option)
            
            # Lưu state
            st.session_state['processed'] = True
            st.session_state['final_depth'] = final_depth
            st.session_state['mesh'] = mesh
            st.session_state['x_g'] = x_g
            st.session_state['y_g'] = y_g
            st.session_state['z_g'] = z_g

if st.session_state.get('processed', False):
    st.success(f"✅ Đã xử lý xong! Gốc tọa độ Work Zero: **{work_zero_option}**")
    
    tab_view, tab_layers = st.tabs([
        "👁️ Phối Cảnh Mesh 3D",
        "🪓 Quản Lý Layer, AI Tư Vấn Thật & Xuất G-Code"
    ])
    
    with tab_view:
        col_m1, col_m2 = st.columns([1, 2])
        with col_m1:
            st.metric("Vertices", f"{len(st.session_state['mesh'].vertices):,}")
            st.metric("Faces", f"{len(st.session_state['mesh'].faces):,}")
            st.info(f"Gốc Zero đặt tại: **{work_zero_option}**")
        with col_m2:
            fig = plt.figure(figsize=(7, 4))
            ax = fig.add_subplot(111, projection='3d')
            x_g, y_g, z_g = st.session_state['x_g'], st.session_state['y_g'], st.session_state['z_g']
            stride = max(1, int(x_g.shape[0] / 80))
            surf = ax.plot_surface(x_g[::stride, ::stride], y_g[::stride, ::stride], z_g[::stride, ::stride], 
                                   cmap='gist_earth', linewidth=0, antialiased=False)
            ax.set_zlim(0, max_depth_mm * 1.5)
            fig.colorbar(surf, shrink=0.5, aspect=5)
            st.pyplot(fig)

    with tab_layers:
        st.subheader("Cấu Hình Chi Tiết - AI Tư Vấn - Xuất G-Code Cho Mỗi Layer (Yêu Cầu 1)")
        
        layer_tabs = st.tabs([
            "Layer 1: Phá Thô (Roughing)",
            "Layer 2: Bán Tinh (Semi-Finish)",
            "Layer 3: Phay Tinh 3D (Finishing)",
            "Layer 4: Khắc Chi Tiết (V-Bit)",
            "Layer 5: Cắt Biên (Profile)"
        ])
        
        # ---------------------------------------------------------
        # LAYER 1: PHÁ THÔ
        # ---------------------------------------------------------
        with layer_tabs[0]:
            st.markdown("### 🔨 Layer 1: Phay Phá Thô (Roughing)")
            c1, c2 = st.columns(2)
            with c1:
                l1_tool_type = st.selectbox("Loại dao:", ["Endmill Flat", "Bullnose Mill"], key="l1_type")
                l1_tool_dia = st.number_input("Đường kính dao (mm):", value=6.0, key="l1_dia")
                l1_spindle = st.number_input("Tốc độ Spindle (RPM):", value=18000, key="l1_rpm")
                l1_feed = st.number_input("Tốc độ cắt F (mm/min):", value=3000, key="l1_f")
                l1_stepdown = st.number_input("Độ sâu cắt Z (Stepdown - mm):", value=3.0, key="l1_sd")
                l1_stepover = st.slider("Độ dịch dao ngang (%)", 30, 70, 50, key="l1_so")
            
            with c2:
                st.write("🤖 **AI Thật Tư Vấn Cho Layer 1:**")
                if st.button("🧠 Hỏi AI tư vấn cho Layer 1", key="btn_ai_l1"):
                    with st.spinner("AI đang phân tích thông số Layer 1..."):
                        advice = call_real_ai_advisor(
                            api_key, "Phay Phá Thô", material, width_mm, height_mm, max_depth_mm,
                            f"{l1_tool_type} Ø{l1_tool_dia}mm", l1_spindle, l1_feed, l1_stepover, l1_stepdown
                        )
                        st.info(advice)
            
            st.markdown("---")
            if st.button("💾 Xuất Mã G-Code (Layer 1 - Phá Thô)", key="btn_gc_l1"):
                so_mm = l1_tool_dia * (l1_stepover / 100.0)
                gcode_l1 = generate_layer_gcode("Roughing", st.session_state['x_g'], st.session_state['y_g'], st.session_state['z_g'], so_mm, l1_feed, l1_spindle, safe_z, f"{l1_tool_type} Ø{l1_tool_dia}mm")
                st.text_area("G-Code Layer 1 preview:", gcode_l1[:300] + "\n...", height=120)
                st.download_button("📥 Tải File G-Code Layer 1 (.nc)", gcode_l1, file_name="Layer1_PhaTho.nc", mime="text/plain")

        # ---------------------------------------------------------
        # LAYER 2: BÁN TINH
        # ---------------------------------------------------------
        with layer_tabs[1]:
            st.markdown("### 🪓 Layer 2: Phay Bán Tinh (Semi-Finishing)")
            c1, c2 = st.columns(2)
            with c1:
                l2_tool_type = st.selectbox("Loại dao:", ["Ballnose (Dao Cầu)", "Bullnose (Mũi Bo)"], key="l2_type")
                l2_tool_dia = st.number_input("Đường kính dao (mm):", value=4.0, key="l2_dia")
                l2_spindle = st.number_input("Tốc độ Spindle (RPM):", value=20000, key="l2_rpm")
                l2_feed = st.number_input("Tốc độ cắt F (mm/min):", value=2500, key="l2_f")
                l2_stepover = st.slider("Độ dịch dao ngang (%)", 10, 30, 20, key="l2_so")
            
            with c2:
                st.write("🤖 **AI Thật Tư Vấn Cho Layer 2:**")
                if st.button("🧠 Hỏi AI tư vấn cho Layer 2", key="btn_ai_l2"):
                    with st.spinner("AI đang phân tích thông số Layer 2..."):
                        advice = call_real_ai_advisor(
                            api_key, "Phay Bán Tinh", material, width_mm, height_mm, max_depth_mm,
                            f"{l2_tool_type} Ø{l2_tool_dia}mm", l2_spindle, l2_feed, l2_stepover
                        )
                        st.info(advice)
            
            st.markdown("---")
            if st.button("💾 Xuất Mã G-Code (Layer 2 - Bán Tinh)", key="btn_gc_l2"):
                so_mm = l2_tool_dia * (l2_stepover / 100.0)
                gcode_l2 = generate_layer_gcode("Semi-Finishing", st.session_state['x_g'], st.session_state['y_g'], st.session_state['z_g'], so_mm, l2_feed, l2_spindle, safe_z, f"{l2_tool_type} Ø{l2_tool_dia}mm")
                st.text_area("G-Code Layer 2 preview:", gcode_l2[:300] + "\n...", height=120)
                st.download_button("📥 Tải File G-Code Layer 2 (.nc)", gcode_l2, file_name="Layer2_BanTinh.nc", mime="text/plain")

        # ---------------------------------------------------------
        # LAYER 3: PHAY TINH 3D
        # ---------------------------------------------------------
        with layer_tabs[2]:
            st.markdown("### ✨ Layer 3: Phay Tinh 3D (Finishing)")
            c1, c2 = st.columns(2)
            with c1:
                l3_tool_type = st.selectbox("Loại dao:", ["Tapered Ballnose (Cầu Nón)", "Ballnose (Cầu Thẳng)"], key="l3_type")
                l3_tool_dia = st.number_input("Bán kính đầu dao R (mm):", value=1.0, key="l3_dia")
                l3_spindle = st.number_input("Tốc độ Spindle (RPM):", value=22000, key="l3_rpm")
                l3_feed = st.number_input("Tốc độ cắt F (mm/min):", value=2000, key="l3_f")
                l3_stepover = st.slider("Độ dịch dao ngang (%)", 5, 15, 8, key="l3_so")
            
            with c2:
                st.write("🤖 **AI Thật Tư Vấn Cho Layer 3:**")
                if st.button("🧠 Hỏi AI tư vấn cho Layer 3", key="btn_ai_l3"):
                    with st.spinner("AI đang phân tích thông số Layer 3..."):
                        advice = call_real_ai_advisor(
                            api_key, "Phay Tinh 3D", material, width_mm, height_mm, max_depth_mm,
                            f"{l3_tool_type} R{l3_tool_dia}mm", l3_spindle, l3_feed, l3_stepover
                        )
                        st.info(advice)
            
            st.markdown("---")
            if st.button("💾 Xuất Mã G-Code (Layer 3 - Phay Tinh)", key="btn_gc_l3"):
                so_mm = l3_tool_dia * (l3_stepover / 100.0)
                gcode_l3 = generate_layer_gcode("Finishing", st.session_state['x_g'], st.session_state['y_g'], st.session_state['z_g'], so_mm, l3_feed, l3_spindle, safe_z, f"{l3_tool_type} R{l3_tool_dia}mm")
                st.text_area("G-Code Layer 3 preview:", gcode_l3[:300] + "\n...", height=120)
                st.download_button("📥 Tải File G-Code Layer 3 (.nc)", gcode_l3, file_name="Layer3_PhayTinh.nc", mime="text/plain")

        # ---------------------------------------------------------
        # LAYER 4: KHẮC CHI TIẾT
        # ---------------------------------------------------------
        with layer_tabs[3]:
            st.markdown("### 🔍 Layer 4: Khắc Chi Tiết Sắc Nhọn (V-Bit Engraving)")
            c1, c2 = st.columns(2)
            with c1:
                l4_angle = st.selectbox("Góc Dao V-Bit (°):", [15, 20, 30, 60, 90], index=2, key="l4_angle")
                l4_spindle = st.number_input("Tốc độ Spindle (RPM):", value=24000, key="l4_rpm")
                l4_feed = st.number_input("Tốc độ cắt F (mm/min):", value=1500, key="l4_f")
            
            with c2:
                st.write("🤖 **AI Thật Tư Vấn Cho Layer 4:**")
                if st.button("🧠 Hỏi AI tư vấn cho Layer 4", key="btn_ai_l4"):
                    with st.spinner("AI đang phân tích thông số Layer 4..."):
                        advice = call_real_ai_advisor(
                            api_key, "Khắc Chi Tiết V-Bit", material, width_mm, height_mm, max_depth_mm,
                            f"Dao V-Bit {l4_angle} độ", l4_spindle, l4_feed, 5
                        )
                        st.info(advice)
            
            st.markdown("---")
            if st.button("💾 Xuất Mã G-Code (Layer 4 - Khắc Chi Tiết)", key="btn_gc_l4"):
                gcode_l4 = generate_layer_gcode("V-Bit Engraving", st.session_state['x_g'], st.session_state['y_g'], st.session_state['z_g'], 0.2, l4_feed, l4_spindle, safe_z, f"V-Bit {l4_angle} deg")
                st.text_area("G-Code Layer 4 preview:", gcode_l4[:300] + "\n...", height=120)
                st.download_button("📥 Tải File G-Code Layer 4 (.nc)", gcode_l4, file_name="Layer4_KhacChiTiet.nc", mime="text/plain")

        # ---------------------------------------------------------
        # LAYER 5: CẮT BIÊN
        # ---------------------------------------------------------
        with layer_tabs[4]:
            st.markdown("### ✂️ Layer 5: Cắt Biên Ngoại Tác (Profile Cutout)")
            c1, c2 = st.columns(2)
            with c1:
                l5_tool_dia = st.number_input("Đường kính dao cắt (mm):", value=6.0, key="l5_dia")
                l5_spindle = st.number_input("Tốc độ Spindle (RPM):", value=18000, key="l5_rpm")
                l5_feed = st.number_input("Tốc độ cắt F (mm/min):", value=2000, key="l5_f")
                l5_stepdown = st.number_input("Độ sâu cắt mỗi pass (mm):", value=4.0, key="l5_sd")
            
            with c2:
                st.write("🤖 **AI Thật Tư Vấn Cho Layer 5:**")
                if st.button("🧠 Hỏi AI tư vấn cho Layer 5", key="btn_ai_l5"):
                    with st.spinner("AI đang phân tích thông số Layer 5..."):
                        advice = call_real_ai_advisor(
                            api_key, "Cắt Biên Profile", material, width_mm, height_mm, max_depth_mm,
                            f"Endmill Ø{l5_tool_dia}mm", l5_spindle, l5_feed, 100, l5_stepdown
                        )
                        st.info(advice)
            
            st.markdown("---")
            if st.button("💾 Xuất Mã G-Code (Layer 5 - Cắt Biên)", key="btn_gc_l5"):
                gcode_l5 = f"(--- LAYER 5: PROFILE CUT ---\n(G54 Zero: {work_zero_option})\nG21\nG90\nM3 S{l5_spindle}\nG0 Z{safe_z}\nG0 X0 Y0\nG1 Z-{max_depth_mm} F{l5_feed}\nG1 X{width_mm}\nG1 Y{height_mm}\nG1 X0\nG1 Y0\nG0 Z{safe_z}\nM5\nM30"
                st.text_area("G-Code Layer 5 preview:", gcode_l5, height=120)
                st.download_button("📥 Tải File G-Code Layer 5 (.nc)", gcode_l5, file_name="Layer5_CatBien.nc", mime="text/plain")
