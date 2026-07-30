import streamlit as st

# ---------------------------------------------------------
# CẤU HÌNH TRANG STREAMLIT
# ---------------------------------------------------------
try:
    st.set_page_config(
        page_title="CAM 3D Relief Generator - Advanced Gemini AI",
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

# Tích hợp Gemini AI
try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# ---------------------------------------------------------
# HÀM BỔ TRỢ & GỌI GEMINI AI CHUYÊN SÂU
# ---------------------------------------------------------
def call_gemini_ai_advisor_expert(api_key, layer_name, material, width, height, depth, tool_info, spindle, feed, stepover, stepdown=None):
    """Hàm gọi Gemini API tư vấn sâu về Dao và Thông số CAM CNC"""
    if not api_key:
        return "⚠️ Chưa nhập Gemini API Key ở Sidebar."
    
    if not GEMINI_AVAILABLE:
        return "⚠️ Chưa cài đặt thư viện Google GenAI SDK (`pip install google-genai`)."

    try:
        client = genai.Client(api_key=api_key)
        
        # PROMPT CHUYÊN SÂU DÀNH CHO DÂN CAM CNC PRO
        prompt = f"""
        Bạn là một Chuyên Gia Lập Trình CAM CNC 3D Relief và Sư Phụ Vận Hành Máy Khắc Phù Điêu với 20 năm kinh nghiệm.
        Hãy phân tích thật CHUYÊN SÂU, CHI TIẾT và KHẮC NGHIỆT cho Layer gia công sau:

        📌 THÔNG TIN ĐẦU VÀO:
        - Lớp gia công: {layer_name}
        - Vật liệu phôi: {material}
        - Kích thước bức tranh: {width} x {height} mm | Độ sâu Z cực đại: {depth} mm
        - Dao đang chọn: {tool_info}
        - Tốc độ Trục chính (Spindle): {spindle} RPM
        - Tốc độ Tiến dao (Feedrate): {feed} mm/min
        - Độ dịch dao ngang (Stepover): {stepover}%
        {f"- Độ sâu lát cắt (Stepdown): {stepdown} mm" if stepdown else "- Bước cắt sâu (Stepdown): Cắt theo biên độ 3D/Toàn bộ độ sâu"}

        🔴 PHÂN TÍCH CHUYÊN SÂU & YÊU CẦU BẮT BUỘC TRẢ LỜI ĐỦ 3 PHẦN:

        1. 🗡️ ĐÁNH GIÁ VỀ DAO GIA CÔNG (Cực kỳ quan trọng):
           - Loại dao đang chọn ({tool_info}) đã chuẩn cho {layer_name} trên vật liệu {material} chưa?
           - Bán kính mũi dao (R) hoặc đường kính (D) này có lấy hết được chi tiết nhỏ/nét mỏng của bức tranh không? Có nguy cơ để lại vết sọc (scallop) không?
           - TƯ VẤN ĐỔI DAO CỤ THỂ: Nên dùng chính xác loại dao nào? (Ví dụ: Dao Cầu Nón Tapered Ballnose R0.25/R0.5/R1.0, Dao V-Bit 20°/30° mũi 0.1mm, hay Endmill Flat 2 lưỡi/3 lưỡi...). Chỉ rõ bán kính R và góc nón tối ưu nhất.

        2. ⚙️ PHÂN TÍCH TỐC ĐỘ & BƯỚC CẮT (Spindle, Feedrate, Stepover, Stepdown):
           - Tốc độ S={spindle} RPM và F={feed} mm/min có gây cháy gỗ/cháy dao hoặc làm xước bề mặt không?
           - Stepover {stepover}% có quá to làm mịn kém hay quá nhỏ gây tốn thời gian chạy máy không?
           - Lực tải dao (Chip load) có an toàn không? Nguy cơ gãy dao hay xơ gỗ ở đâu?

        3. 🛠️ BẢNG ĐIỀU CHỈNH THÔNG SỐ TỐI ƯU GỢI Ý:
           Đưa ra danh sách thông số chuẩn xác khuyến nghị để nhập lại vào máy (Tên dao đề xuất, Spindle RPM, Feedrate F, Stepover %, Stepdown mm).
        """

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"❌ Lỗi Gemini API: {str(e)}"

def depth_map_to_mesh(depth_img, width_mm, height_mm, max_depth_mm, work_zero):
    """Chuyển đổi Depth Map 2D thành Mesh 3D tính theo Work Zero"""
    h, w = depth_img.shape
    norm_depth = (depth_img - depth_img.min()) / (depth_img.max() - depth_img.min() + 1e-6)
    
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
        "G21", "G90",
        f"G0 Z{safe_z:.2f}",
        f"M3 S{spindle_speed}"
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

    gcode.append("M5")
    gcode.append("G0 Z50")
    gcode.append("M30")
    return "\n".join(gcode)

# ---------------------------------------------------------
# TIÊU ĐỀ
# ---------------------------------------------------------
st.title("🖼️ CAM 3D Relief & Gemini AI Tư Vấn Dao & Thông Số Chuyên Sâu")
st.caption("Chuyên gia Gemini AI sẽ soi kỹ từng thông số, chỉ rõ chính xác loại dao R/D/Góc và Feedrate/Spindle cho từng Layer")

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.header("✨ Gemini AI Key")
gemini_api_key = st.sidebar.text_input("Gemini API Key:", type="password")

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Thông Số Phôi & Zero")
width_mm = st.sidebar.number_input("Chiều rộng (X - mm)", value=300, step=10)
height_mm = st.sidebar.number_input("Chiều cao (Y - mm)", value=400, step=10)
max_depth_mm = st.sidebar.slider("Độ sâu Z (mm)", min_value=3.0, max_value=50.0, value=15.0, step=0.5)
safe_z = st.sidebar.number_input("Safe Z (mm)", value=10.0)

work_zero_option = st.sidebar.selectbox(
    "Gốc tọa độ Work Zero [0,0]:",
    ["Góc Dưới Trái (Bottom-Left)", "Góc Trên Trái (Top-Left)", "Góc Dưới Phải (Bottom-Right)", "Góc Trên Phải (Top-Right)", "Tâm Phôi (Center)"]
)

material = st.sidebar.selectbox("Loại phôi gia công:", ["Gỗ Trắc/Hương/Cẩm (Cực Cứng)", "Gỗ Gụ/Sồi/Tần Bì (Cứng Tr.Bình)", "Gỗ Mềm/MDF", "Nhôm Dẻo/Đồng Thau", "Mica/Acrylic"])

# ---------------------------------------------------------
# MAIN INTERFACE
# ---------------------------------------------------------
uploaded_img = st.file_uploader("📥 Tải lên ảnh mẫu (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_img:
    img_raw = Image.open(uploaded_img).convert("L")
    st.image(img_raw, caption="Ảnh mẫu đã chọn", width=200)

    st.markdown("---")
    if st.button("🚀 KÍCH HOẠT PHÂN TÍCH CHUYÊN SÂU & XỬ LÝ G-CODE", type="primary", use_container_width=True):
        with st.spinner("AI đang phân tích từng loại dao, lực cắt, chạy Mesh 3D và lập trình G-Code..."):
            # 1. Tiền xử lý ảnh
            img_np = np.array(img_raw)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced_img = clahe.apply(img_np)
            final_depth = cv2.GaussianBlur(enhanced_img, (3, 3), 0)
            
            # 2. Dựng Mesh 3D
            mesh, x_g, y_g, z_g = depth_map_to_mesh(final_depth, width_mm, height_mm, max_depth_mm, work_zero_option)
            
            # 3. GỌI GEMINI AI CHUYÊN SÂU TƯ VẤN DAO CHO TẤT CẢ LAYER
            ai_advices = {}
            gcode_files = {}

            # Khai báo thông số để Gemini AI soi
            layer_configs = {
                "Layer 1: Phá Thô": {"name": "Phay Phá Thô (Roughing)", "tool": "Endmill Flat D6mm (Dao Phay Mặt Cắt Bằng)", "spindle": 18000, "feed": 3200, "so": 50, "sd": 3.5, "so_mm": 3.0},
                "Layer 2: Bán Tinh": {"name": "Phay Bán Tinh (Semi-Finish)", "tool": "Ballnose D4mm (Dao Cầu Tròn D4)", "spindle": 20000, "feed": 2500, "so": 20, "sd": None, "so_mm": 0.8},
                "Layer 3: Phay Tinh 3D": {"name": "Phay Tinh 3D (Finishing)", "tool": "Tapered Ballnose R0.5mm Góc 5° (Dao Cầu Nón)", "spindle": 22000, "feed": 2000, "so": 8, "sd": None, "so_mm": 0.08},
                "Layer 4: Khắc V-Bit": {"name": "Khắc Chi Tiết Nét Mỏng", "tool": "Dao V-Bit Góc 30° Mũi 0.2mm", "spindle": 24000, "feed": 1400, "so": 5, "sd": None, "so_mm": 0.05},
                "Layer 5: Cắt Biên": {"name": "Cắt Khung Biên Bức Tranh", "tool": "Endmill Flat D6mm 2 Lưỡi Thẳng", "spindle": 18000, "feed": 1800, "so": 100, "sd": 4.0, "so_mm": 6.0}
            }

            for l_key, cfg in layer_configs.items():
                # Call AI Expert
                ai_advices[l_key] = call_gemini_ai_advisor_expert(
                    gemini_api_key, cfg["name"], material, width_mm, height_mm, max_depth_mm,
                    cfg["tool"], cfg["spindle"], cfg["feed"], cfg["so"], cfg["sd"]
                )
                # G-Code
                if "Cắt Biên" in cfg["name"]:
                    gcode_files[l_key] = f"(--- LAYER 5: PROFILE CUT ---\nG21\nG90\nM3 S{cfg['spindle']}\nG0 Z{safe_z}\nG0 X0 Y0\nG1 Z-{max_depth_mm} F{cfg['feed']}\nG1 X{width_mm}\nG1 Y{height_mm}\nG1 X0\nG1 Y0\nG0 Z{safe_z}\nM5\nM30"
                else:
                    gcode_files[l_key] = generate_layer_gcode(cfg["name"], x_g, y_g, z_g, cfg["so_mm"], cfg["feed"], cfg["spindle"], safe_z, cfg["tool"])

            st.session_state['processed'] = True
            st.session_state['mesh'] = mesh
            st.session_state['x_g'] = x_g
            st.session_state['y_g'] = y_g
            st.session_state['z_g'] = z_g
            st.session_state['ai_advices'] = ai_advices
            st.session_state['gcode_files'] = gcode_files

# ---------------------------------------------------------
# HIỂN THỊ KẾT QUẢ
# ---------------------------------------------------------
if st.session_state.get('processed', False):
    st.success(f"✅ ĐÃ HOÀN TẤT PHÂN TÍCH CHUYÊN SÂU TỪ GEMINI AI! (Work Zero: {work_zero_option})")
    
    tab_view, tab_layers = st.tabs([
        "👁️ Xem Mô Hình 3D Mesh",
        "🪓 Đánh Giá Dao & Thông Số Chi Tiết Từ AI"
    ])
    
    with tab_view:
        fig = plt.figure(figsize=(7, 3.5))
        ax = fig.add_subplot(111, projection='3d')
        x_g, y_g, z_g = st.session_state['x_g'], st.session_state['y_g'], st.session_state['z_g']
        stride = max(1, int(x_g.shape[0] / 80))
        surf = ax.plot_surface(x_g[::stride, ::stride], y_g[::stride, ::stride], z_g[::stride, ::stride], cmap='gist_earth', linewidth=0)
        st.pyplot(fig)

    with tab_layers:
        layer_tabs = st.tabs([
            "Layer 1: Phá Thô",
            "Layer 2: Bán Tinh",
            "Layer 3: Phay Tinh 3D",
            "Layer 4: Khắc V-Bit",
            "Layer 5: Cắt Biên"
        ])

        tab_mapping = [
            ("Layer 1: Phá Thô", layer_tabs[0], "Layer1_PhaTho.nc"),
            ("Layer 2: Bán Tinh", layer_tabs[1], "Layer2_BanTinh.nc"),
            ("Layer 3: Phay Tinh 3D", layer_tabs[2], "Layer3_PhayTinh.nc"),
            ("Layer 4: Khắc V-Bit", layer_tabs[3], "Layer4_KhacChiTiet.nc"),
            ("Layer 5: Cắt Biên", layer_tabs[4], "Layer5_CatBien.nc"),
        ]

        for key, tab_obj, file_name in tab_mapping:
            with tab_obj:
                st.markdown(f"### 📌 Phân Tích Chuyên Sâu - {key}")
                
                # HIỂN THỊ TƯ VẤN SÂU
                st.markdown(st.session_state['ai_advices'][key])
                
                st.markdown("---")
                st.write("💾 **Mã G-Code Tương Ứng:**")
                st.text_area("Xem trước mã G-Code:", st.session_state['gcode_files'][key][:300] + "\n...", height=120, key=f"txt_{key}")
                st.download_button(
                    label=f"📥 Tải File G-Code ({file_name})",
                    data=st.session_state['gcode_files'][key],
                    file_name=file_name,
                    mime="text/plain",
                    key=f"dl_{key}"
                )
