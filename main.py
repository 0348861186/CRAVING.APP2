import streamlit as st

# ---------------------------------------------------------
# CẤU HÌNH TRANG STREAMLIT
# ---------------------------------------------------------
try:
    st.set_page_config(
        page_title="CAM 3D Relief Generator for CNC",
        page_layout="wide",
        initial_sidebar_state="expanded"
    )
except Exception:
    pass

# ---------------------------------------------------------
# IMPORT THƯ VIỆN
# ---------------------------------------------------------
import numpy as np
import cv2
from PIL import Image
import trimesh
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# HÀM BỔ TRỢ (HELPER FUNCTIONS)
# ---------------------------------------------------------
def depth_map_to_mesh(depth_img, width_mm, height_mm, max_depth_mm, work_zero="Góc Dưới Trái (Bottom-Left)"):
    """Chuyển đổi Depth Map 2D thành Mesh 3D tính theo Gốc Tọa Độ chọn trước"""
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
st.title("🖼️ Hệ Thống CAM 3D Relief & Tư Vấn AI Theo Layer CNC")
st.caption("Tự động phân lớp gia công, cấu hình Work Zero, tư vấn thông số AI & Xuất G-Code riêng")

# ---------------------------------------------------------
# SIDEBAR: CẤU HÌNH VẬT LÝ & WORK ZERO
# ---------------------------------------------------------
st.sidebar.header("⚙️ 1. Thấu Số Phôi & Gốc Tọa Độ")
width_mm = st.sidebar.number_input("Chiều rộng (X - mm)", value=300, step=10)
height_mm = st.sidebar.number_input("Chiều cao (Y - mm)", value=400, step=10)
max_depth_mm = st.sidebar.slider("Độ sâu phù điêu (Z - mm)", min_value=3.0, max_value=50.0, value=15.0, step=0.5)
safe_z = st.sidebar.number_input("Chiều cao an toàn (Safe Z - mm)", value=10.0)

st.sidebar.markdown("---")
st.sidebar.header("📍 2. Định Vị Gốc Tọa Độ (Work Zero)")
work_zero_option = st.sidebar.selectbox(
    "Chọn gốc 0 (G54/Work Zero):",
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
material = st.sidebar.selectbox("Loại phôi gỗ/vật liệu:", ["Gỗ Tự Nhiên Chắc (Hương, Gụ, Trắc)", "Gỗ Mềm/Gỗ Công Nghiệp (MDF)", "Nhôm/Đồng", "Nhựa Acrylic/Formex"])

# ---------------------------------------------------------
# MAIN INTERFACE
# ---------------------------------------------------------
uploaded_img = st.file_uploader("📥 Tải lên ảnh mẫu (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_img:
    img_raw = Image.open(uploaded_img).convert("L")
    st.image(img_raw, caption="Ảnh mẫu gốc đã tải lên", width=300)

    # NÚT KÍCH HOẠT XỬ LÝ AI THEO YÊU CẦU 3
    st.markdown("---")
    if st.button("🤖 KÍCH HOẠT AI TƯ VẤN & TẠO LAYER G-CODE", type="primary", use_container_width=True):
        with st.spinner("AI đang xử lý ảnh, tạo Depth Map, phân tích Layer và tính toán mã G-Code..."):
            # 1. Xử lý ảnh
            img_np = np.array(img_raw)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced_img = clahe.apply(img_np)
            final_depth = cv2.GaussianBlur(enhanced_img, (3, 3), 0)
            
            # 2. Tạo Mesh 3D theo Work Zero
            mesh, x_g, y_g, z_g = depth_map_to_mesh(final_depth, width_mm, height_mm, max_depth_mm, work_zero_option)
            
            # Lưu dữ liệu vào session state
            st.session_state['processed'] = True
            st.session_state['final_depth'] = final_depth
            st.session_state['mesh'] = mesh
            st.session_state['x_g'] = x_g
            st.session_state['y_g'] = y_g
            st.session_state['z_g'] = z_g

if st.session_state.get('processed', False):
    st.success(f"✅ Đã xử lý xong! Gốc Work Zero được chọn: **{work_zero_option}**")
    
    tab_view, tab_layers, tab_post = st.tabs([
        "👁️ Xem Phối Cảnh 3D",
        "🪓 Lập Trình & AI Tư Vấn Từng Layer",
        "🪵 Quy Trình Hậu Kỳ"
    ])
    
    with tab_view:
        col_m1, col_m2 = st.columns([1, 2])
        with col_m1:
            st.write("**Thông số Mesh:**")
            st.metric("Số đỉnh (Vertices)", f"{len(st.session_state['mesh'].vertices):,}")
            st.metric("Số mặt (Faces)", f"{len(st.session_state['mesh'].faces):,}")
            st.info(f"Gốc tọa độ [0,0] đặt tại: **{work_zero_option}**")
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
        st.subheader("Cấu Hình Riêng & AI Tư Vấn Cho Từng Layer Gia Công")
        
        layer_tabs = st.tabs([
            "Layer 1: Phá Thô (Roughing)",
            "Layer 2: Phay Bán Tinh (Semi-Finish)",
            "Layer 3: Phay Tinh 3D (Finishing)",
            "Layer 4: Khắc Chi Tiết (V-Bit Detail)",
            "Layer 5: Cắt Biên (Profile Cut)"
        ])
        
        # --- LAYER 1: PHÁ THÔ ---
        with layer_tabs[0]:
            st.markdown("### 🔨 Layer 1: Phá Thô (Roughing Pass)")
            c1, c2 = st.columns(2)
            with c1:
                l1_tool_type = st.selectbox("Loại dao:", ["Endmill Trục Thẳng (Flat)", "Bullnose Mill (Mũi Bo Corners)"], key="l1_type")
                l1_tool_dia = st.number_input("Đường kính dao (mm):", value=6.0, key="l1_dia")
                l1_spindle = st.number_input("Tốc độ Spindle (RPM):", value=18000, key="l1_rpm")
                l1_feed = st.number_input("Tốc độ cắt F (mm/min):", value=3000, key="l1_f")
                l1_stepdown = st.number_input("Mỗi lát cắt sâu Z (Stepdown - mm):", value=3.0, key="l1_sd")
                l1_stepover = st.slider("Độ dịch dao (%)", 30, 70, 50, key="l1_so")
            
            with c2:
                st.write("🤖 **AI Tư Vấn Cho Layer Phá Thô:**")
                st.info(f"""
                - **Khuyên dùng:** Với vật liệu **{material}**, dao **{l1_tool_type} Ø{l1_tool_dia}mm** thích hợp để ăn lượng dư lớn.
                - **Spindle đề xuất:** {l1_spindle} RPM là tối ưu để tránh cháy gỗ.
                - **Lưu ý:** Giữ lượng dư (Stock Allowance) khoảng **0.5mm - 1.0mm** để phục vụ bước phay tinh tiếp theo.
                """)
                
            if st.button("🚀 Xuất Mã G-Code (Layer 1 - Phá Thô)"):
                so_mm = l1_tool_dia * (l1_stepover / 100.0)
                gcode_l1 = generate_layer_gcode("Roughing", st.session_state['x_g'], st.session_state['y_g'], st.session_state['z_g'], so_mm, l1_feed, l1_spindle, safe_z, f"{l1_tool_type} Ø{l1_tool_dia}mm")
                st.text_area("Mã G-Code xem trước:", gcode_l1[:400] + "\n...", height=150)
                st.download_button("💾 Tải G-Code Layer 1 (.nc)", gcode_l1, file_name="Layer1_Phathô.nc", mime="text/plain")

        # --- LAYER 2: BÁN TINH ---
        with layer_tabs[1]:
            st.markdown("### 🪓 Layer 2: Phay Bán Tinh (Semi-Finishing)")
            c1, c2 = st.columns(2)
            with c1:
                l2_tool_type = st.selectbox("Loại dao:", ["Ballnose (Dao Cầu)", "Bullnose (Mũi Bo)"], key="l2_type")
                l2_tool_dia = st.number_input("Đường kính dao (mm):", value=4.0, key="l2_dia")
                l2_spindle = st.number_input("Tốc độ Spindle (RPM):", value=20000, key="l2_rpm")
                l2_feed = st.number_input("Tốc độ cắt F (mm/min):", value=2500, key="l2_f")
                l2_stepover = st.slider("Độ dịch dao (%)", 10, 30, 20, key="l2_so")
            
            with c2:
                st.write("🤖 **AI Tư Vấn Cho Layer Bán Tinh:**")
                st.info(f"""
                - **Vai trò:** Giảm bớt các bậc thang gồ ghề do dao phá thô tạo ra.
                - **Khuyên dùng:** Dao cầu **Ø{l2_tool_dia}mm** giúp giảm tải cho dao phay tinh siêu nhỏ ở Layer 3, hạn chế gãy dao.
                """)
                
            if st.button("🚀 Xuất Mã G-Code (Layer 2 - Bán Tinh)"):
                so_mm = l2_tool_dia * (l2_stepover / 100.0)
                gcode_l2 = generate_layer_gcode("Semi-Finishing", st.session_state['x_g'], st.session_state['y_g'], st.session_state['z_g'], so_mm, l2_feed, l2_spindle, safe_z, f"{l2_tool_type} Ø{l2_tool_dia}mm")
                st.text_area("Mã G-Code xem trước:", gcode_l2[:400] + "\n...", height=150)
                st.download_button("💾 Tải G-Code Layer 2 (.nc)", gcode_l2, file_name="Layer2_BanTinh.nc", mime="text/plain")

        # --- LAYER 3: PHAY TINH 3D ---
        with layer_tabs[2]:
            st.markdown("### ✨ Layer 3: Phay Tinh 3D (3D Finishing)")
            c1, c2 = st.columns(2)
            with c1:
                l3_tool_type = st.selectbox("Loại dao:", ["Tapered Ballnose (Dao Cầu Nón)", "Ballnose (Dao Cầu Thẳng)"], key="l3_type")
                l3_tool_dia = st.number_input("Bán kính/Đường kính đầu dao (mm):", value=1.0, key="l3_dia")
                l3_spindle = st.number_input("Tốc độ Spindle (RPM):", value=22000, key="l3_rpm")
                l3_feed = st.number_input("Tốc độ cắt F (mm/min):", value=2000, key="l3_f")
                l3_stepover = st.slider("Độ dịch dao (%)", 5, 15, 8, key="l3_so")
            
            with c2:
                st.write("🤖 **AI Tư Vấn Cho Layer Phay Tinh:**")
                st.info(f"""
                - **Tối ưu bề mặt:** Sử dụng **Tapered Ballnose R{l3_tool_dia}mm** độ cứng cao, chống rung lắc tốt khi khắc mặt gỗ sâu.
                - **Stepover:** Đặt ở mức **8%** giúp bề mặt mịn láng mà không mất quá nhiều thời gian phay.
                """)
                
            if st.button("🚀 Xuất Mã G-Code (Layer 3 - Phay Tinh)"):
                so_mm = l3_tool_dia * (l3_stepover / 100.0)
                gcode_l3 = generate_layer_gcode("Finishing", st.session_state['x_g'], st.session_state['y_g'], st.session_state['z_g'], so_mm, l3_feed, l3_spindle, safe_z, f"{l3_tool_type} R{l3_tool_dia}mm")
                st.text_area("Mã G-Code xem trước:", gcode_l3[:400] + "\n...", height=150)
                st.download_button("💾 Tải G-Code Layer 3 (.nc)", gcode_l3, file_name="Layer3_PhayTinh.nc", mime="text/plain")

        # --- LAYER 4: KHẮC CHI TIẾT ---
        with layer_tabs[3]:
            st.markdown("### 🔍 Layer 4: Khắc Chi Tiết Sắc Nhọn (V-Bit Engraving)")
            c1, c2 = st.columns(2)
            with c1:
                l4_angle = st.selectbox("Góc Dao V-Bit (°):", [15, 20, 30, 60, 90], index=2, key="l4_angle")
                l4_spindle = st.number_input("Tốc độ Spindle (RPM):", value=24000, key="l4_rpm")
                l4_feed = st.number_input("Tốc độ cắt F (mm/min):", value=1500, key="l4_f")
            
            with c2:
                st.write("🤖 **AI Tư Vấn Cho Layer Khắc Chi Tiết:**")
                st.info(f"""
                - **Mục đích:** Nhấn các góc nhọn, nét chữ, rãnh hẹp mà dao cầu không chui vào được.
                - **Khuyên dùng:** Dao V-Bit **{l4_angle}°** chạy tốc độ Spindle cao **{l4_spindle} RPM** để đường cắt sắc nét, không bị xơ xước gỗ.
                """)
                
            if st.button("🚀 Xuất Mã G-Code (Layer 4 - Khắc V-Bit)"):
                gcode_l4 = generate_layer_gcode("V-Bit Engraving", st.session_state['x_g'], st.session_state['y_g'], st.session_state['z_g'], 0.2, l4_feed, l4_spindle, safe_z, f"V-Bit {l4_angle} deg")
                st.text_area("Mã G-Code xem trước:", gcode_l4[:400] + "\n...", height=150)
                st.download_button("💾 Tải G-Code Layer 4 (.nc)", gcode_l4, file_name="Layer4_KhacChiTiet.nc", mime="text/plain")

        # --- LAYER 5: CẮT BIÊN ---
        with layer_tabs[4]:
            st.markdown("### ✂️ Layer 5: Cắt Biên Ngoại Tác (Profile Cutout)")
            c1, c2 = st.columns(2)
            with c1:
                l5_tool_dia = st.number_input("Đường kính dao cắt (mm):", value=6.0, key="l5_dia")
                l5_spindle = st.number_input("Tốc độ Spindle (RPM):", value=18000, key="l5_rpm")
                l5_feed = st.number_input("Tốc độ cắt F (mm/min):", value=2000, key="l5_f")
                l5_stepdown = st.number_input("Độ sâu cắt mỗi pass (mm):", value=4.0, key="l5_sd")
            
            with c2:
                st.write("🤖 **AI Tư Vấn Cắt Biên:**")
                st.info(f"""
                - **Khuyên dùng:** Sử dụng dao Endmill **Ø{l5_tool_dia}mm** nén (Compression Bit) khi cắt gỗ ghép/MDF hoặc dao Xoắn Phay (Up-Cut) đối với gỗ thịt.
                - **Cảnh báo an toàn:** Cần bổ sung **Tabs/Cầu giữ phôi** để tránh khi cắt đứt phôi bị văng làm gãy dao.
                """)
                
            if st.button("🚀 Xuất Mã G-Code (Layer 5 - Cắt Biên)"):
                # Cắt profile vòng quanh biên ngoài
                gcode_l5 = f"(--- LAYER 5: PROFILE CUT ---)\nG21\nG90\nM3 S{l5_spindle}\nG0 Z{safe_z}\nG0 X0 Y0\nG1 Z-{max_depth_mm} F{l5_feed}\nG1 X{width_mm}\nG1 Y{height_mm}\nG1 X0\nG1 Y0\nG0 Z{safe_z}\nM5\nM30"
                st.text_area("Mã G-Code xem trước:", gcode_l5, height=150)
                st.download_button("💾 Tải G-Code Layer 5 (.nc)", gcode_l5, file_name="Layer5_CatBien.nc", mime="text/plain")

    with tab_post:
        st.markdown("### 🪵 Quy Trình Hậu Kỳ Xử Lý Sau Gia Công CNC")
        st.success("""
        1. **Làm sạch phôi:** Dùng khí nén xịt sạch bụi gỗ trong các khe rãnh nhỏ.
        2. **Chà nhám (Sanding):** Dùng chổi nhám hoặc giấy nhám dẻo P240/P320 chà nhẹ các sợi xơ gỗ sinh ra do dao khắc.
        3. **Sơn lót (Sealer):** Phun 1 lớp sơn lót lau khô để chống ẩm và giữ nét cho tranh.
        4. **Lau màu (Stain):** Sử dụng lau màu tối (như màu óc chó/cánh gián) vào các vùng chìm sâu để tăng độ tương phản 3D nổi bật.
        """)
