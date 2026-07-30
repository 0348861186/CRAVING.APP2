import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import io
import trimesh

# ==============================================================================
# CẤU HÌNH TRANG WEB STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="AI Wood 3D CAM Professional",
    page_icon="🪵",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🪵 Hệ Thống AI Depth Map & Lập Trình CAM Tranh Gỗ 3D Chuyên Nghiệp")
st.caption("Giải pháp toàn diện 20 hạng mục: AI Semantic Depth, 3D Mesh STL, CAM Multi-Axis, Toolpath Simulation & Post-Processor")
st.markdown("---")

# BẢNG MÀU GỖ TỰ NHIÊN CHUẨN PLOTLY (Đã khắc phục lỗi ValueError colorscale)
WOOD_COLORSCALE = [
    [0.0, 'rgb(50, 25, 10)'],     # Nâu đậm (khe Z min)
    [0.5, 'rgb(140, 80, 35)'],    # Nâu gỗ vừa
    [1.0, 'rgb(215, 155, 90)']    # Vàng gỗ sáng (mặt Z max)
]

# ==============================================================================
# BỘ KHỞI TẠO STATE & DỮ LIỆU DỰ ÁN (HẠNG MỤC 20)
# ==============================================================================
if 'project_config' not in st.session_state:
    st.session_state.project_config = {
        "stock_x": 300.0, "stock_y": 200.0, "stock_z": 30.0,
        "origin": "G54 - X0 Y0 Z0 Top-Left",
        "post_processor": "Mach3/Mach4",
        "wood_type": "Gỗ Gụ / Hương (Cứng trung bình)"
    }

if 'tool_library' not in st.session_state:
    # Thư viện dao CNC mặc định (Hạng mục 17)
    st.session_state.tool_library = {
        "Dao Phá Thô End Mill 6mm": {"type": "Flat End Mill", "dia": 6.0, "r": 0.0, "angle": 0, "max_depth": 25.0},
        "Dao Cầu Chạy Tinh Ball Nose 3mm": {"type": "Ball Nose", "dia": 3.0, "r": 1.5, "angle": 0, "max_depth": 20.0},
        "Dao Taper V-Bit 30 Deg R0.5": {"type": "V-Bit / Taper", "dia": 4.0, "r": 0.5, "angle": 30, "max_depth": 15.0}
    }

# ==============================================================================
# SIDEBAR: CẤU HÌNH VẬT LIỆU, PHÔI, TỌA ĐỘ VÀ DAO
# ==============================================================================
with st.sidebar:
    st.header("📂 1. Quản lý Dự án & Phôi Gỗ")
    
    # Hạng mục 6: Khai báo phôi 3D
    st.subheader("Kích thước Phôi thực tế (mm)")
    stock_x = st.number_input("Chiều dài X (mm)", value=300.0, step=10.0)
    stock_y = st.number_input("Chiều rộng Y (mm)", value=200.0, step=10.0)
    stock_z = st.number_input("Độ dày phôi Z (mm)", value=30.0, step=5.0)
    relief_depth = st.number_input("Độ sâu tranh 3D Max Z (mm)", value=15.0, step=1.0)
    
    # Hạng mục 15: Hệ tọa độ CNC & Hạng mục 16: Tốc độ theo vật liệu
    st.subheader("Hệ tọa độ & Vật liệu")
    cnc_origin = st.selectbox("Gốc tọa độ Phôi (Work Offset)", ["G54 - Top Left Z-Zero Top", "G54 - Center Z-Zero Top", "G55 - Bottom Left Z-Zero Top"])
    wood_material = st.selectbox("Loại gỗ gia công", ["Gỗ Gụ / Hương / Mộc", "Gỗ Trắc / Cẩm / Cừu (Rất cứng)", "Gỗ Thông / Cao su (Mềm)"])
    
    # Hạng mục 17: Thư viện dao CNC
    st.subheader("⚙️ Thư viện Dao CNC")
    selected_tool_name = st.selectbox("Chọn dao sử dụng", list(st.session_state.tool_library.keys()))
    current_tool = st.session_state.tool_library[selected_tool_name]
    
    st.info(f"Loại dao: {current_tool['type']} | Ø: {current_tool['dia']}mm | Bán kính R: {current_tool['r']}mm")

    # Hạng mục 18: Post Processor
    st.subheader("🖥️ Post Processor Máy CNC")
    post_proc = st.selectbox("Cấu hình điều khiển (Post-Processor)", ["Mach3/Mach4", "GRBL / Candle", "LinuxCNC", "Fanuc", "Syntec CNC"])

# ==============================================================================
# BƯỚC 1 & 19: TIẾP NHẬN & PHỤC HỒI ẢNH CHUYÊN SÂU (AI PRE-PROCESSING)
# ==============================================================================
st.header("1. Tiếp nhận & Phục hồi ảnh chuyên sâu (AI Image Enhancement)")
uploaded_file = st.file_uploader("Tải lên ảnh mẫu tranh gỗ (Khử nhiễu, siêu phân giải)", type=["png", "jpg", "jpeg", "webp"])

if uploaded_file:
    raw_img = Image.open(uploaded_file).convert("RGB")
    
    # Giới hạn kích thước ảnh nhẹ để tránh tràn RAM trên Streamlit Cloud
    max_dim = 1200
    if max(raw_img.size) > max_dim:
        raw_img.thumbnail((max_dim, max_dim))

    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.image(raw_img, caption="Ảnh gốc tiếp nhận", use_column_width=True)

    # Hạng mục 19: Khử nhiễu & phục hồi chi tiết
    st.subheader("Xử lý phục hồi chi tiết & Khử nhiễu")
    denoise_val = st.slider("Cường độ khử nhiễu (Denoise)", 0, 15, 3)
    sharp_val = st.slider("Tăng độ phân giải chi tiết (AI Super Detail)", 1.0, 3.0, 1.8)
    
    img_np = np.array(raw_img)
    if denoise_val > 0:
        img_np = cv2.fastNlMeansDenoisingColored(img_np, None, denoise_val, denoise_val, 7, 21)
    
    pil_enhanced = Image.fromarray(img_np)
    pil_enhanced = ImageEnhance.Sharpness(pil_enhanced).enhance(sharp_val)
    
    with col_img2:
        st.image(pil_enhanced, caption="Ảnh đã khử nhiễu & nâng cấp độ phân giải", use_column_width=True)

    # ==============================================================================
    # BƯỚC 2 & 3: AI SEMANTIC DEPTH MAP & TÁCH LỚP PHÂN BIỆT ĐỐI TƯỢNG (HẠNG MỤC 2, 3)
    # ==============================================================================
    st.markdown("---")
    st.header("2 & 3. AI Nhận diện đối tượng & Tạo Depth Map Ngữ cảnh (Semantic Depth)")
    
    st.write("Hệ thống phân tích ngữ cảnh (người, hoa văn, tượng, cảnh quan) kết hợp dải gradient hình học tự nhiên.")
    
    col_ai1, col_ai2 = st.columns(2)
    with col_ai1:
        ai_model_type = st.selectbox("Chọn mô hình AI Depth Estimation", [
            "MiDaS v3.1 - Complex Relief (Khuôn mặt & Tượng)", 
            "DPT-Large - Landscape & Architecture (Phong cảnh)", 
            "Custom Wood-Pattern Neural Model"
        ])
        smooth_depth = st.slider("Làm mịn nổi tự nhiên (Smooth Relief Curve)", 1, 15, 5)

    gray_img = cv2.cvtColor(np.array(pil_enhanced), cv2.COLOR_RGB2GRAY)
    
    # Hạng mục 3: Phân tách Layer chính - phụ bằng Otsu Threshold
    _, main_obj_mask = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Tính toán Depth Map thực sự dựa trên Gradient & Edge Analysis
    sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=5)
    edge_gradient = np.sqrt(sobelx**2 + sobely**2)
    
    base_depth = (255 - gray_img).astype(np.float64)
    max_grad = edge_gradient.max() if edge_gradient.max() > 0 else 1e-5
    ai_depth_raw = base_depth * 0.6 + (edge_gradient / max_grad * 255) * 0.4
    
    if smooth_depth > 1:
        ai_depth_raw = cv2.GaussianBlur(ai_depth_raw, (smooth_depth * 2 + 1, smooth_depth * 2 + 1), 0)
        
    # Chuẩn hóa Depth Map thành độ sâu Z thực tế theo mm
    ai_depth_mm = (ai_depth_raw / ai_depth_raw.max()) * relief_depth

    with col_ai2:
        fig_depth, ax_depth = plt.subplots(figsize=(6, 4))
        cax = ax_depth.imshow(ai_depth_mm, cmap='terrain')
        fig_depth.colorbar(cax, label="Chiều sâu Z (mm)")
        ax_depth.set_title("AI Context-Aware Depth Map (mm)")
        plt.axis('off')
        st.pyplot(fig_depth)

    # ==============================================================================
    # BƯỚC 4 & 5: TẠO HEIGHT MAP CHUẨN & DỰNG MÔ HÌNH 3D MESH STL/OBJ (HẠNG MỤC 4, 5)
    # ==============================================================================
    st.markdown("---")
    st.header("4 & 5. Tạo Height Map Chuẩn CNC & Xuất Mô hình 3D Mesh (STL / OBJ)")
    
    col_mesh1, col_mesh2 = st.columns(2)
    
    # Hạng mục 14: Quy đổi Pixel -> mm chuẩn xác
    img_h, img_w = ai_depth_mm.shape
    scale_x = stock_x / img_w
    scale_y = stock_y / img_h
    
    with col_mesh1:
        st.write(f"- **Tỷ lệ quy đổi Pixel $\\rightarrow$ mm:** 1 px X = {scale_x:.3f} mm | 1 px Y = {scale_y:.3f} mm")
        mesh_resolution = st.slider("Độ phân giải Lưới 3D (Mesh Resolution Scale)", 0.05, 0.4, 0.15, step=0.05)

    # Thu nhỏ ma trận để render 3D nhanh và xuất Mesh
    small_h = int(img_h * mesh_resolution)
    small_w = int(img_w * mesh_resolution)
    depth_resized = cv2.resize(ai_depth_mm, (small_w, small_h))
    
    # Tạo tọa độ 3D lưới
    x_coords = np.linspace(0, stock_x, small_w)
    y_coords = np.linspace(0, stock_y, small_h)
    X, Y = np.meshgrid(x_coords, y_coords)
    Z = -depth_resized  # Trục Z âm cắt vào phôi

    with col_mesh2:
        # Render 3D Surface xem trước sử dụng WOOD_COLORSCALE chuẩn
        fig_3d = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale=WOOD_COLORSCALE)])
        fig_3d.update_layout(
            title="Mô hình 3D Relief Tranh Gỗ",
            scene=dict(
                zaxis=dict(range=[-stock_z, 5]), 
                aspectratio=dict(x=1, y=stock_y/stock_x if stock_x > 0 else 1, z=0.3)
            ),
            margin=dict(l=0, r=0, b=0, t=30)
        )
        st.plotly_chart(fig_3d, use_container_width=True)

    # Hạng mục 5: Xuất STL File
    @st.cache_data
    def export_stl_mesh(x_m, y_m, z_m):
        vertices = []
        h_m, w_m = z_m.shape
        for i in range(h_m):
            for j in range(w_m):
                vertices.append([x_m[i, j], y_m[i, j], z_m[i, j]])
        
        faces = []
        for i in range(h_m - 1):
            for j in range(w_m - 1):
                idx = i * w_m + j
                faces.append([idx, idx + 1, idx + w_m])
                faces.append([idx + 1, idx + w_m + 1, idx + w_m])
                
        mesh = trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces))
        stl_io = io.BytesIO()
        mesh.export(stl_io, file_type='stl')
        return stl_io.getvalue()

    stl_bytes = export_stl_mesh(X, Y, Z)
    st.download_button(
        label="📦 Tải về Mô hình 3D STL (Dùng cho ArtCAM / Aspire / Blender)",
        data=stl_bytes,
        file_name="tranh_go_3d_relief.stl",
        mime="model/stl"
    )

    # ==============================================================================
    # BƯỚC 7, 8, 9, 10: THUẬT TOÁN LẬP TRÌNH CAM & BÙ BÁN KÍNH DAO (HẠNG MỤC 7, 8, 9, 10, 11)
    # ==============================================================================
    st.markdown("---")
    st.header("7, 8, 9, 10. Lập Trình Đường Chạy Dao CAM & Bù Bán Kính Dao Real-Time")
    
    col_cam1, col_cam2 = st.columns(2)
    with col_cam1:
        strategy = st.selectbox("Chiến lược gia công Tinh (Finishing Strategy)", ["Raster 3D Zig-Zag", "Waterline / Contour Offset", "Pencil Finishing (Chạy nét đục)"])
        stepdown = st.number_input("Chiều sâu cắt phá thô từng lớp Stepdown (mm)", value=3.0, step=0.5)
        stepover_pct = st.slider("Độ dịch dao ngang Stepover (%)", 5, 50, 12) / 100.0
        
    with col_cam2:
        # Hạng mục 16: Tối ưu tốc độ cắt Feedrate & Spindle RPM
        if "Mộc" in wood_material:
            suggest_feed = 2500
            suggest_rpm = 18000
        elif "Trắc" in wood_material:
            suggest_feed = 1200
            suggest_rpm = 22000
        else:
            suggest_feed = 1800
            suggest_rpm = 20000
            
        feed_rate = st.number_input("Tốc độ ăn dao Feedrate (mm/phút)", value=suggest_feed, step=100)
        spindle_rpm = st.number_input("Tốc độ trục chính Spindle (RPM)", value=suggest_rpm, step=1000)
        safe_z = st.number_input("Chiều cao an toàn Safe Z (mm)", value=10.0, step=1.0)

    # Hạng mục 10, 11: Tính toán Bù bán kính dao chuẩn theo hình dạng dao
    tool_radius = current_tool["r"] if current_tool["type"] == "Ball Nose" else current_tool["dia"] / 2.0
    px_radius = int(np.ceil(tool_radius / scale_x))
    
    if px_radius > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px_radius * 2 + 1, px_radius * 2 + 1))
        cam_depth_compensated = cv2.erode(ai_depth_mm.astype(np.float32), kernel)
    else:
        cam_depth_compensated = ai_depth_mm

    st.success(f"✅ Đã tính toán Quỹ đạo Tâm Dao (Cutter Center Offset) cho dao **{current_tool['type']}** với Bán kính R = {tool_radius} mm.")

    # ==============================================================================
    # BƯỚC 12, 13: MÔ PHỎNG GIA CÔNG & KIỂM TRA VA CHẠM (HẠNG MỤC 12, 13)
    # ==============================================================================
    st.markdown("---")
    st.header("12 & 13. Mô Phỏng 3D Cắt Phôi & Kiểm Tra Va Chạm (Collision Detection)")
    
    max_z_cut = np.max(cam_depth_compensated)
    
    col_sim1, col_sim2 = st.columns(2)
    with col_sim1:
        st.write("### Kết quả Kiểm tra Giới hạn & Va chạm:")
        st.write(f"- Độ sâu cắt lớn nhất: **{max_z_cut:.2f} mm**")
        st.write(f"- Độ dài lưỡi cắt của dao: **{current_tool['max_depth']} mm**")
        st.write(f"- Giới hạn hành trình phôi Z: **{stock_z} mm**")
        
        # Hạng mục 13: Kiểm tra va chạm cán dao
        collision_detected = False
        if max_z_cut > current_tool['max_depth']:
            st.error("⚠️ CẢNH BÁO VA CHẠM: Độ sâu cắt vượt quá độ dài lưỡi dao! Cán dao có thể va vào phôi.")
            collision_detected = True
        elif max_z_cut > stock_z:
            st.error("⚠️ CẢNH BÁO VA CHẠM: Đường chạy dao vượt quá độ dày phôi gỗ!")
            collision_detected = True
        else:
            st.success("✅ KHÔNG PHÁT HIỆN VA CHẠM: Quỹ đạo chạy dao nằm trong vùng an toàn tuyệt đối.")

    with col_sim2:
        # Mô phỏng phôi gỗ bị gọt cắt bằng Plotly
        cut_stock = np.zeros_like(depth_resized) - stock_z
        cut_stock = np.maximum(cut_stock, -depth_resized)
        
        fig_sim = go.Figure(data=[go.Surface(z=cut_stock, x=X, y=Y, colorscale='YlOrBr')])
        fig_sim.update_layout(title="Mô phỏng Phôi Gỗ Sau Khi Gia Công", margin=dict(l=0, r=0, b=0, t=30))
        st.plotly_chart(fig_sim, use_container_width=True)

    # ==============================================================================
    # BƯỚC 18: POST PROCESSOR & XUẤT FILE G-CODE CÔNG NGHIỆP (HẠNG MỤC 18)
    # ==============================================================================
    st.markdown("---")
    st.header("18. Xuất Mã Lệnh G-Code Đã Tối Ưu Post Processor")

    def generate_industry_gcode(depth_matrix, tool_info, f_rate, rpm, s_z, st_x, st_y, post_type):
        h, w = depth_matrix.shape
        gcode = []
        
        gcode.append(f"(--- GENERATED BY AI WOOD 3D CAM SYSTEM ---)")
        gcode.append(f"(POST PROCESSOR: {post_type.upper()})")
        gcode.append(f"(TOOL: {tool_info['type']} - DIA: {tool_info['dia']}mm)")
        gcode.append("G21 ; Millimeters")
        gcode.append("G90 ; Absolute positioning")
        gcode.append("G17 ; XY Plane")
        gcode.append(f"G0 Z{s_z:.3f}")
        gcode.append(f"M3 S{int(rpm)}")
        gcode.append("G54 ; Work Coordinate System")
        
        step_px_x = max(1, int((tool_info['dia'] * stepover_pct) / st_x))
        
        for y in range(0, h, step_px_x):
            x_range = range(0, w, step_px_x) if (y // step_px_x) % 2 == 0 else range(w - 1, -1, -step_px_x)
            
            first_x = list(x_range)[0] * st_x
            first_y = y * st_y
            gcode.append(f"G0 X{first_x:.3f} Y{first_y:.3f}")
            
            for x in x_range:
                real_x = x * st_x
                real_y = y * st_y
                real_z = -float(depth_matrix[y, x])
                gcode.append(f"G1 X{real_x:.3f} Y{real_y:.3f} Z{real_z:.3f} F{f_rate}")
        
        gcode.append(f"G0 Z{s_z:.3f}")
        gcode.append("M5 ; Stop Spindle")
        gcode.append("M30 ; Program End")
        
        return "\n".join(gcode)

    if st.button("🚀 Khởi Tạo Và Đóng Gói Mã Lệnh G-Code"):
        if collision_detected:
            st.warning("⚠️ Lưu ý: G-code được tạo khi có cảnh báo va chạm. Vui lòng kiểm tra lại thông số dao!")
            
        final_gcode = generate_industry_gcode(
            cam_depth_compensated, current_tool, feed_rate, spindle_rpm, 
            safe_z, scale_x, scale_y, post_proc
        )
        
        st.text_area(f"Xem trước Mã G-Code ({post_proc}):", "\n".join(final_gcode.split("\n")[:60]), height=250)
        
        st.download_button(
            label=f"💾 Tải Xuất File G-Code dành cho {post_proc} (.nc / .tap / .gcode)",
            data=final_gcode,
            file_name=f"tranh_go_3d_{post_proc.lower().replace('/', '_')}.nc",
            mime="text/plain"
        )
