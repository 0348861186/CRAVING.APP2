import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import io

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="Hệ thống Chuyển đổi Tranh Gỗ 3D Cloud",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🪵 Phần Mềm Phay Tranh Gỗ 3D & Lập Trình CAM (12 Bước)")
st.caption("Phiên bản triển khai trên Streamlit Community Cloud")
st.markdown("---")

# ==========================================
# SIDEBAR: CẤU HÌNH THÔNG SỐ VẬT LIỆU & DAO
# ==========================================
st.sidebar.header("⚙️ Thông số CAM & Dao CNC")
tool_dia = st.sidebar.number_input("Đường kính dao (mm)", value=6.0, step=0.5)
tool_radius = tool_dia / 2.0

rough_stepover = st.sidebar.slider("Dịch dao ngang Phá thô (%)", 10, 80, 50) / 100.0
rough_stepdown = st.sidebar.number_input("Chiều sâu cắt lớp Phá thô (mm)", value=3.0, step=0.5)

finish_stepover = st.sidebar.slider("Dịch dao ngang Chạy mịn (%)", 1, 30, 10) / 100.0
feed_rate = st.sidebar.number_input("Tốc độ cắt F (mm/min)", value=1500, step=100)
plunge_rate = st.sidebar.number_input("Tốc độ đâm dao Z (mm/min)", value=500, step=50)
safe_z = st.sidebar.number_input("Độ cao an toàn Safe Z (mm)", value=10.0, step=1.0)
target_depth = st.sidebar.number_input("Độ sâu điêu khắc Z max (mm)", value=15.0, step=1.0)

# ==========================================
# BƯỚC 1: TIẾP NHẬN HÌNH ẢNH TỪ KHÁCH HÀNG
# ==========================================
st.header("1. Tiếp nhận hình ảnh mẫu tranh")
uploaded_file = st.file_uploader("Tải lên tệp hình ảnh (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    raw_img = Image.open(uploaded_file).convert("RGB")
    
    # Resize nhẹ để tối ưu bộ nhớ RAM trên Streamlit Cloud
    max_dim = 1000
    if max(raw_img.size) > max_dim:
        raw_img.thumbnail((max_dim, max_dim))
        st.info("Ảnh đã được điều chỉnh kích thước nhẹ để tối ưu hiệu năng tính toán Cloud.")

    col1, col2 = st.columns(2)
    with col1:
        st.image(raw_img, caption="Hình ảnh gốc", use_column_width=True)
    
    img_np = np.array(raw_img)

    # ==========================================
    # BƯỚC 2: NÂNG CẤP VÀ PHỤC HỒI ĐỘ NÉAT
    # ==========================================
    st.header("2. Nâng cấp và phục hồi độ nét AI/Filter")
    col_a, col_b = st.columns(2)
    with col_a:
        sharpness = st.slider("Độ sắc nét (Sharpness)", 1.0, 3.0, 1.8, 0.1)
        contrast = st.slider("Độ tương phản (Contrast)", 1.0, 2.5, 1.3, 0.1)
    
    enhancer = ImageEnhance.Sharpness(raw_img)
    enhanced_img = enhancer.enhance(sharpness)
    enhancer_c = ImageEnhance.Contrast(enhanced_img)
    enhanced_img = enhancer_c.enhance(contrast)
    
    with col_b:
        st.image(enhanced_img, caption="Ảnh nâng cấp độ nét", use_column_width=True)

    # ==========================================
    # BƯỚC 3: TÁCH PHÂN LỚP ĐỐI TƯỢNG CHÍNH - PHỤ
    # ==========================================
    st.header("3. Tách phân lớp các đối tượng")
    gray = cv2.cvtColor(np.array(enhanced_img), cv2.COLOR_RGB2GRAY)
    
    thresh_val = st.slider("Ngưỡng tách đối tượng", 0, 255, 127)
    _, mask = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.image(mask, caption="Mặt nạ phân lớp (Layer Mask)", use_column_width=True)
    with col_m2:
        foreground = cv2.bitwise_and(np.array(enhanced_img), np.array(enhanced_img), mask=mask)
        st.image(foreground, caption="Đối tượng chính", use_column_width=True)

    # ==========================================
    # BƯỚC 4 & 5: NHẬN DIỆN KHỐI VÀ BẢN ĐỒ ĐỘ SÂU (DEPTH MAP)
    # ==========================================
    st.header("4 & 5. Nhận diện khối & Khởi tạo Depth Map")
    blur_ksize = st.slider("Độ làm mịn khối (Blur Smooth)", 3, 31, 9, step=2)
    blurred_gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    
    depth_map = (255 - blurred_gray) / 255.0 * target_depth

    fig, ax = plt.subplots(figsize=(6, 4))
    cax = ax.imshow(depth_map, cmap='terrain')
    fig.colorbar(cax, label="Độ sâu Z (mm)")
    plt.axis('off')
    st.pyplot(fig)

    # ==========================================
    # BƯỚC 6 & 7: 3D RELIEF & SCULPTING DETAIL
    # ==========================================
    st.header("6 & 7. Chuyển đổi 3D Relief & Điêu khắc bề mặt")
    sculpt_factor = st.slider("Cường độ chi tiết điêu khắc", 0.5, 2.0, 1.0, 0.1)
    
    sobelx = cv2.Sobel(blurred_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred_gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_detail = np.sqrt(sobelx**2 + sobely**2)
    
    if sobel_detail.max() > 0:
        sobel_detail = (sobel_detail / sobel_detail.max()) * sculpt_factor

    final_depth_map = np.clip(depth_map + sobel_detail, 0, target_depth)
    st.success("Đã hoàn tất tính toán chi tiết điêu khắc 3D!")

    # ==========================================
    # BƯỚC 8, 9, 10: LẬP TRÌNH CAM & BÙ BÁN KÍNH DAO
    # ==========================================
    st.header("8, 9 & 10. Lập trình CAM (Phá thô, Chạy mịn, Bù bán kính dao)")
    
    kernel_size = int(np.ceil(tool_radius)) * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    compensated_depth_map = cv2.erode(final_depth_map.astype(np.float32), kernel)

    st.info(f"Áp dụng bù bán kính dao {tool_radius:.1f} mm thành công.")

    # ==========================================
    # BƯỚC 11: MÔ PHỎNG VÀ KIỂM TRA VA CHẠM
    # ==========================================
    st.header("11. Mô phỏng & Kiểm tra va chạm (Collision Check)")
    
    max_z_engrave = np.max(compensated_depth_map)
    min_z_engrave = np.min(compensated_depth_map)
    
    st.write(f"- **Độ sâu cắt Z lớn nhất:** {max_z_engrave:.2f} mm")
    st.write(f"- **Độ sâu cắt Z nhỏ nhất:** {min_z_engrave:.2f} mm")
    
    if max_z_engrave > target_depth:
        st.error("⚠️ CẢNH BÁO: Phát hiện độ sâu vượt quá giới hạn an toàn!")
    else:
        st.success("✅ Kiểm tra an toàn: Không phát hiện xung đột chiều sâu cắt.")

    # ==========================================
    # BƯỚC 12: ĐÓNG GÓI VÀ XUẤT FILE G-CODE
    # ==========================================
    st.header("12. Đóng gói và xuất file mã lệnh G-code")

    def generate_gcode(depth_matrix, f_rate, p_rate, s_z, step_x):
        h, w = depth_matrix.shape
        gcode = []
        gcode.append("(--- G-CODE TRANH GO 3D RELIEF - STREAMLIT CLOUD ---)")
        gcode.append("G21 ; Measure mm")
        gcode.append("G90 ; Absolute coordinates")
        gcode.append(f"G0 Z{s_z:.3f}")
        gcode.append("M3 S12000")
        
        step_pixel = max(1, int(step_x))
        
        for y in range(0, h, step_pixel):
            x_range = range(0, w, step_pixel) if (y // step_pixel) % 2 == 0 else range(w - 1, -1, -step_pixel)
            for x in x_range:
                z_val = -float(depth_matrix[y, x])
                gcode.append(f"G1 X{x:.3f} Y{y:.3f} Z{z_val:.3f} F{f_rate}")
        
        gcode.append(f"G0 Z{s_z:.3f}")
        gcode.append("M5")
        gcode.append("M30")
        return "\n".join(gcode)

    if st.button("🚀 Khởi tạo & Xuất G-Code"):
        step_pixel = max(1, int(tool_dia * finish_stepover))
        gcode_data = generate_gcode(compensated_depth_map, feed_rate, plunge_rate, safe_z, step_pixel)
        
        st.text_area("Xem trước G-Code (50 dòng đầu):", "\n".join(gcode_data.split("\n")[:50]), height=200)
        
        st.download_button(
            label="💾 Tải về File G-Code (.nc)",
            data=gcode_data,
            file_name="tranh_go_3d_relief.nc",
            mime="text/plain"
        )
