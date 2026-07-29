import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io

# ==========================================
# 1. CẤU HÌNH TRANG & GIAO DIỆN STREAMLIT
# ==========================================
st.set_page_config(
    page_title="AI CNC Wood Carving Studio",
    page_layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🪚 AI CNC Wood Carving Studio")
st.caption("Ứng dụng tạo & mô phỏng G-code đục gỗ 3D/Relief từ hình ảnh")

# ==========================================
# 2. THANH BÊN (SIDEBAR) - THÔNG SỐ VẬN HÀNH
# ==========================================
st.sidebar.header("⚙️ Thông Số Kỹ Thuật CNC")

# Kích thước phôi
st.sidebar.subheader("1. Kích Thước Phôi Gỗ (mm)")
width = st.sidebar.number_input("Chiều rộng phôi (X)", min_value=10.0, value=400.0, step=10.0)
height = st.sidebar.number_input("Chiều dài phôi (Y)", min_value=10.0, value=600.0, step=10.0)
thickness = st.sidebar.number_input("Độ dày phôi (Z)", min_value=1.0, value=20.0, step=1.0)
max_depth = st.sidebar.slider("Độ sâu đục tối đa (mm)", min_value=1.0, max_value=float(thickness), value=10.0)

# ------------------------------------------
# BỔ SUNG: TỌA ĐỘ MỐC (WORK ZERO ORIGIN) - DROPDOWN
# ------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Tọa Độ Mốc (Work Zero Origin)")

zero_position = st.sidebar.selectbox(
    "Vị trí mốc (X=0, Y=0):",
    options=[
        "Góc dưới - Bên trái (Bottom-Left)",
        "Góc dưới - Bên phải (Bottom-Right)",
        "Góc trên - Bên trái (Top-Left)",
        "Góc trên - Bên phải (Top-Right)",
        "Chính giữa phôi (Center)"
    ],
    index=0
)

z_zero_position = st.sidebar.selectbox(
    "Vị trí mốc chiều cao (Z=0):",
    options=[
        "Mặt trên phôi (Material Top)",
        "Mặt bàn máy (Material Bottom / Bed)"
    ],
    index=0
)

# Tính toán Offset dựa trên dropdown chọn Work Zero
if zero_position == "Góc dưới - Bên trái (Bottom-Left)":
    offset_x, offset_y = 0.0, 0.0
elif zero_position == "Góc dưới - Bên phải (Bottom-Right)":
    offset_x, offset_y = float(width), 0.0
elif zero_position == "Góc trên - Bên trái (Top-Left)":
    offset_x, offset_y = 0.0, float(height)
elif zero_position == "Góc trên - Bên phải (Top-Right)":
    offset_x, offset_y = float(width), float(height)
elif zero_position == "Chính giữa phôi (Center)":
    offset_x, offset_y = width / 2.0, height / 2.0

# Thông số Dao & Tốc độ
st.sidebar.markdown("---")
st.sidebar.subheader("2. Thông Số Dao & Tốc Độ")
tool_diameter = st.sidebar.number_input("Đường kính mũi dao (mm)", min_value=0.1, value=3.175, step=0.1)
stepover_percent = st.sidebar.slider("Bước dịch dao (%)", min_value=10, max_value=90, value=40)
feed_rate = st.sidebar.number_input("Tốc độ cắt F (mm/phút)", min_value=100, value=2500, step=100)
plunge_rate = st.sidebar.number_input("Tốc độ lao dao Z (mm/phút)", min_value=50, value=800, step=50)
safe_z = st.sidebar.number_input("Chiều cao Z an toàn (mm)", min_value=1.0, value=5.0, step=1.0)
spindle_speed = st.sidebar.number_input("Tốc độ trục chính (RPM)", min_value=1000, value=18000, step=1000)

# ==========================================
# 3. NỘI DUNG CHÍNH (MAIN DASHBOARD)
# ==========================================

# Hiển thị tóm tắt thiết lập gốc
st.info(
    f"📍 **Thiết lập Mốc Tọa Độ hiện tại:** "
    f"X0, Y0 ở **{zero_position}** (Offset: X={offset_x:.1f}mm, Y={offset_y:.1f}mm) | "
    f"Z0 ở **{z_zero_position}**"
)

# Upload hình ảnh mẫu đục
uploaded_file = st.file_uploader("📥 Tải lên ảnh thiết kế (JPG, PNG, BMP)", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)

    # Đọc và xử lý ảnh sang Grayscale Heightmap (Logic giữ nguyên)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    with col1:
        st.subheader("🖼️ Ảnh Gốc Tải Lên")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)

    with col2:
        st.subheader("🗺️ Bản Đồ Độ Sâu (Heightmap)")
        st.image(gray, use_container_width=True, caption="Vùng trắng = Nông / Vùng đen = Sâu")

    st.markdown("---")

    # Nút bấm phát sinh G-Code
    if st.button("🚀 Khởi Tạo G-Code Đục Gỗ 3D", type="primary"):
        with st.spinner("Đang tính toán đường chạy dao CNC..."):
            
            # Tính bước dịch dao (Stepover in mm)
            stepover = tool_diameter * (stepover_percent / 100.0)

            # Resize ảnh theo kích thước mm và độ phân giải đường chạy dao (Logic gốc)
            rows = int(height / stepover)
            cols = int(width / stepover)
            resized_gray = cv2.resize(gray, (cols, rows), interpolation=cv2.INTER_AREA)

            # Khởi tạo chuỗi G-Code
            gcode_lines = []
            
            # Header
            gcode_lines.append("(--- DỰ ÁN CNC WOOD CARVING ---)")
            gcode_lines.append(f"(Kich thuoc phoi: X={width}mm, Y={height}mm, Z={thickness}mm)")
            gcode_lines.append(f"(Moc Work Zero X,Y: {zero_position})")
            gcode_lines.append(f"(Moc Work Zero Z: {z_zero_position})")
            gcode_lines.append("G21 (Don vi: mm)")
            gcode_lines.append("G90 (Toa do tuyet doi)")
            gcode_lines.append("G54 (Chon he toa do G54)")
            gcode_lines.append(f"M03 S{spindle_speed} (Bat truc chinh)")
            gcode_lines.append(f"G00 Z{safe_z:.3f} (Dua dao len chieu cao an toan)")

            # Thuật toán Raster zic-zac (Logic gốc + Trừ Offset Work Zero)
            for r in range(rows):
                y_coord = (r * stepover) - offset_y
                
                # Đi zic-zac để tối ưu đường chạy dao
                col_range = range(cols) if r % 2 == 0 else range(cols - 1, -1, -1)
                
                for c in col_range:
                    x_coord = (c * stepover) - offset_x
                    
                    # Tính độ sâu Z dựa vào giá trị pixel
                    pixel_val = resized_gray[r, c]
                    depth = (1.0 - (pixel_val / 255.0)) * max_depth
                    
                    if z_zero_position == "Mặt trên phôi (Material Top)":
                        z_coord = -depth
                    else:
                        z_coord = thickness - depth

                    if r == 0 and c == 0:
                        gcode_lines.append(f"G00 X{x_coord:.3f} Y{y_coord:.3f}")
                        gcode_lines.append(f"G01 Z{z_coord:.3f} F{plunge_rate}")
                    else:
                        gcode_lines.append(f"G01 X{x_coord:.3f} Y{y_coord:.3f} Z{z_coord:.3f} F{feed_rate}")

            # Footer
            gcode_lines.append(f"G00 Z{safe_z:.3f} (Rut dao an toan)")
            gcode_lines.append("G00 X0 Y0 (Ve toa do goc)")
            gcode_lines.append("M05 (Tat truc chinh)")
            gcode_lines.append("M30 (Ket thuc chuong trinh)")

            full_gcode = "\n".join(gcode_lines)

        st.success("✅ Đã xuất file G-code thành công!")

        # Xem trước & Tải về
        st.subheader("📜 Xem Trước G-Code")
        st.text_area("Mẫu G-Code phát sinh:", value="\n".join(gcode_lines[:30]) + "\n\n... (Còn tiếp) ...", height=200)

        st.download_button(
            label="💾 Tải File G-Code (.nc)",
            data=full_gcode,
            file_name="CNC_Carving_Model.nc",
            mime="text/plain"
        )
else:
    st.warning("👈 Vui lòng tải lên một bức ảnh mẫu đục ở thanh công cụ bên trái để bắt đầu!")
