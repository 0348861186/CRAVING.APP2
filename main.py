import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import io

# ==========================================
# CẤU HÌNH TRANG DASHBOARD CHUYÊN NGHIỆP
# ==========================================
st.set_page_config(
    page_title="Wood CNC 3D Relief Generator",
    page_icon="🪵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS cho giao diện Dashboard
st.markdown("""
    <style>
    .main-header {
        font-size: 28px;
        font-weight: bold;
        color: #4A2E16;
        text-align: center;
        padding: 10px;
        background-color: #F8F1E7;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    .step-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-left: 5px solid #8B5A2B;
        margin-bottom: 15px;
    }
    .stButton>button {
        background-color: #8B5A2B;
        color: white;
        border-radius: 6px;
        font-weight: bold;
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🪵 MÁY TẠO MÃ G-CODE PHAY TRANH GỖ 3D (AI & CAM)</div>', unsafe_allow_html=True)

# ==========================================
# SIDEBAR: CẤU HÌNH THÔNG SỐ CNC & CAM
# ==========================================
st.sidebar.header("⚙️ Cấu Hình Máy CNC & Dao")

# Kích thước phôi gỗ
st.sidebar.subheader("1. Kích thước phôi (mm)")
width_mm = st.sidebar.number_input("Chiều rộng (X)", value=200, step=10)
height_mm = st.sidebar.number_input("Chiều cao (Y)", value=200, step=10)
max_depth_mm = st.sidebar.number_input("Độ sâu khắc tối đa (Z)", value=10.0, step=0.5, help="Chiều sâu phay âm xuống gỗ")

# Thông số dao & cắt
st.sidebar.subheader("2. Thông số Dao & Cắt")
tool_diameter = st.sidebar.number_input("Đường kính dao (mm)", value=3.175, step=0.1)
stepover_ratio = st.sidebar.slider("Độ dịch dao ngang (% đường kính dao)", 10, 100, 40) / 100.0
feed_rate = st.sidebar.number_input("Tốc độ cắt F (mm/phút)", value=1200, step=100)
spindle_speed = st.sidebar.number_input("Tốc độ trục chính S (RPM)", value=12000, step=1000)
safe_z = st.sidebar.number_input("Độ cao an toàn Safe Z (mm)", value=5.0, step=1.0)

# ==========================================
# CÁC HÀM XỬ LÝ THEO QUY TRÌNH (ENGINE)
# ==========================================

def ai_restore_image(img):
    """Bước 2: AI / Phục hồi & Tối ưu ảnh đầu vào"""
    img_np = np.array(img)
    # Khử nhiễu
    denoised = cv2.fastNlMeansDenoisingColored(img_np, None, 10, 10, 7, 21)
    # Tăng cường tương phản (CLAHE)
    lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    restored = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return restored

def ai_segment_object(img_np):
    """Bước 3: AI phân tích / Tách đối tượng & Làm rõ biên dạng"""
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    # Làm nổi bật cạnh nét bằng Unsharp Masking
    gaussian = cv2.GaussianBlur(gray, (0, 0), 2.0)
    unsharp = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)
    return unsharp

def generate_depth_map(gray_img):
    """Bước 4 & 5: AI Tạo chiều sâu 3D & 3D Relief Engine"""
    # Chuẩn hóa giá trị xám từ 0.0 đến 1.0 (0: Đen/Sâu nhất, 1: Trắng/Nông nhất)
    depth_map = gray_img.astype(float) / 255.0
    # Làm mịn nhẹ chiều sâu để dao di chuyển mượt mà
    depth_map = cv2.GaussianBlur(depth_map, (3, 3), 0)
    return depth_map

def generate_gcode(depth_map, w_mm, h_mm, max_z, step_ratio, tool_d, f_rate, s_speed, z_safe):
    """Bước 6, 7, 8: CAM Engine, Toolpath Generator & G-Code Export (GRBL/UGS)"""
    rows, cols = depth_map.shape
    step_x = w_mm / cols
    step_y = tool_d * step_ratio
    num_passes_y = int(np.ceil(h_mm / step_y))

    gcode = []
    gcode.append("(=== CREATED BY WOOD CNC RELIEF GENERATOR ===)")
    gcode.append("(Dành cho máy GRBL / Universal Gcode Sender)")
    gcode.append("G90 G21 (Tuyet doi, Don vi mm)")
    gcode.append(f"M3 S{int(s_speed)} (Bat truc chinh)")
    gcode.append(f"G0 Z{z_safe:.3f} (Dua dao len cao an toan)")

    # Lập trình đường dao phay đan lưới (Raster Strategy)
    for i in range(num_passes_y):
        y_pos = i * step_y
        if y_pos > h_mm:
            y_pos = h_mm
        
        # Lấy dòng tương ứng trong depth map
        row_idx = int((y_pos / h_mm) * (rows - 1))
        
        # Đi từ trái sang phải hoặc ngược lại để tối ưu hành trình
        x_indices = range(cols) if i % 2 == 0 else range(cols - 1, -1, -1)
        
        # Di chuyển nhanh đến đầu đường phay
        first_x = (x_indices[0] / (cols - 1)) * w_mm
        gcode.append(f"G0 X{first_x:.3f} Y{y_pos:.3f}")
        
        for j in x_indices:
            x_pos = (j / (cols - 1)) * w_mm
            # Độ sâu Z: Ảnh tối = Phay sâu (-max_z), Ảnh sáng = Phay nông (0)
            z_pos = -1.0 * (1.0 - depth_map[row_idx, j]) * max_z
            gcode.append(f"G1 X{x_pos:.3f} Y{y_pos:.3f} Z{z_pos:.3f} F{f_rate}")

    gcode.append(f"G0 Z{z_safe:.3f} (Rut dao)")
    gcode.append("G0 X0 Y0 (Ve gốc tọa độ)")
    gcode.append("M5 (Tat truc chinh)")
    gcode.append("M30 (Kiet thuc chuong trinh)")

    return "\n".join(gcode)

# ==========================================
# GIAO DIỆN CHÍNH (MAIN DASHBOARD BOARD)
# ==========================================

uploaded_file = st.file_uploader("📂 Bước 1: Tải ảnh khách gửi (JPG, PNG, WEBP)", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    raw_image = Image.open(uploaded_file).convert("RGB")
    
    st.subheader("🔄 Tiến trình xử lý tự động")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**1. Ảnh gốc khách gửi**")
        st.image(raw_image, use_container_width=True)

    # Thực hiện quy trình AI & CAM
    with st.spinner("Đang chạy AI Phục hồi & Phân tích..."):
        restored_img = ai_restore_image(raw_image)
        segmented_img = ai_segment_object(restored_img)
        depth_map = generate_depth_map(segmented_img)

    with col2:
        st.markdown("**2. AI Phục hồi & Nét hóa**")
        st.image(restored_img, use_container_width=True)

    with col3:
        st.markdown("**3. AI Phân tích đối tượng**")
        st.image(segmented_img, use_container_width=True)
    with col4:
        st.markdown("**4. AI 3D Depth Map**")
        # Hiển thị depth map dưới dạng heatmap relief
        fig, ax = plt.subplots()
        cax = ax.imshow(depth_map, cmap='magma')
        plt.axis('off')
        st.pyplot(fig, use_container_width=True)

    st.divider()

    # Tạo Mã G-Code
    st.subheader("🛠️ Kết xuất CAM Engine & G-Code")
    
    if st.button("🚀 BẮT ĐẦU TẠO ĐƯỜNG DAO & MÃ G-CODE GRBL"):
        with st.spinner("Đang tính toán Toolpath và tạo mã G-code..."):
            gcode_text = generate_gcode(
                depth_map, width_mm, height_mm, max_depth_mm, 
                stepover_ratio, tool_diameter, feed_rate, spindle_speed, safe_z
            )
            
            c_left, c_right = st.columns([2, 1])
            
            with c_left:
                st.markdown("**Xem trước mã G-code (50 dòng đầu):**")
                st.code("\n".join(gcode_text.split("\n")[:50]) + "\n...", language="gcode")
            
            with c_right:
                st.success("✅ Tạo đường dao CNC thành công!")
                st.info(f"📍 Kích thước: {width_mm}x{height_mm} mm\n\n📉 Độ sâu Z max: -{max_depth_mm} mm\n\n⏱️ Chuẩn mã: GRBL / UGS")
                
                # Nút Tải G-code về máy
                st.download_button(
                    label="📥 TẢI FILE .NC (G-CODE DÙNG CHO UGS)",
                    data=gcode_text,
                    file_name="tranh_go_3d.nc",
                    mime="text/plain"
                )

else:
    st.info("👋 Vui lòng tải lên ảnh tranh gỗ ở ô trên để bắt đầu quy trình.")
