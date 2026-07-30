import io
import zipfile
import cv2
import numpy as np
import streamlit as st
from PIL import Image

# ==============================================================================
# CẤU HÌNH TRANG WEB
# ==============================================================================
st.set_page_config(
    page_title="AI Wood CAM GRBL/UGS Professional",
    page_icon="🪵",
    layout="wide",
)

st.title("🪵 Hệ Thống AI CAM Tranh Gỗ 3D Phân Lớp (Chuẩn GRBL / UGS)")
st.caption(
    "Sửa lỗi KeyError, Tự động Khởi tạo Session State An toàn & Cắt Biên Đa Tầng"
)
st.markdown("---")

# ==============================================================================
# KHỞI TẠO SESSION STATE (SỬA LỖI KEYERROR)
# ==============================================================================
if "gcode_layers" not in st.session_state:
    st.session_state["gcode_layers"] = {}
if "processed_images" not in st.session_state:
    st.session_state["processed_images"] = {}

# ==============================================================================
# SIDEBAR: CẤU HÌNH PHÔI & WORK ZERO
# ==============================================================================
with st.sidebar:
    st.header("📂 1. Kích Thước Phôi & Work Zero")
    stock_x = st.number_input("Chiều dài phôi X (mm)", value=300.0, step=10.0)
    stock_y = st.number_input("Chiều rộng phôi Y (mm)", value=200.0, step=10.0)
    stock_z = st.number_input("Chiều dày phôi Z (mm)", value=30.0, step=5.0)
    relief_depth = st.number_input(
        "Độ sâu tranh 3D Z (mm)", value=15.0, step=1.0
    )
    safe_z = st.number_input(
        "Chiều cao an toàn Safe Z (mm)", value=10.0, step=1.0
    )

    st.subheader("🎯 Đặt Gốc Phôi (Work Zero X0 Y0 Z0)")
    work_zero = st.selectbox(
        "Vị trí lấy gốc dao:",
        [
            "Góc dưới bên trái (Bottom-Left - Chuẩn UGS)",
            "Tâm phôi (Center)",
            "Góc trên bên trái (Top-Left)",
            "Góc trên bên phải (Top-Right)",
            "Góc dưới bên phải (Bottom-Right)",
        ],
    )

    st.markdown("---")
    st.header("🪵 2. Chọn Loại Gỗ Gia Công")
    wood_type = st.selectbox(
        "Vật liệu gỗ phôi:",
        [
            "Gỗ Gụ / Hương / Mộc (Cứng vừa)",
            "Gỗ Trắc / Cẩm / Cừu (Rất cứng)",
            "Gỗ Thông / Cao Su (Mềm)",
        ],
    )

    # Tự động gợi ý Feedrate/Spindle dựa theo loại gỗ
    if "Rất cứng" in wood_type:
        default_feed = 1200
        default_stepdown = 1.0
    elif "Mềm" in wood_type:
        default_feed = 2500
        default_stepdown = 3.0
    else:
        default_feed = 1800
        default_stepdown = 2.0

    st.markdown("---")
    st.header("⚙️ 3. Thông Số Dao & Chế Độ Cắt")
    tool_dia = st.number_input(
        "Đường kính dao cắt (mm)", value=6.0, step=1.0
    )
    feed_rate = st.number_input(
        "Tốc độ tiến dao F (mm/min)", value=default_feed, step=100
    )
    plunge_rate = st.number_input(
        "Tốc độ đâm dao F_Plunge (mm/min)", value=500, step=50
    )
    spindle_speed = st.number_input(
        "Tốc độ trục chính S (RPM)", value=18000, step=1000
    )
    step_down = st.number_input(
        "Độ sâu mỗi lớp cắt ΔZ (mm)", value=default_stepdown, step=0.5
    )

    st.markdown("---")
    st.info(
        "🤖 **Bộ điều khiển:** Chuẩn **GRBL / UGS (Universal Gcode Sender)**"
    )

# ==============================================================================
# GIAO DIỆN CHÍNH: TẢI ẢNH ĐỘ SÂU (DEPTH MAP)
# ==============================================================================
col_ui_1, col_ui_2 = st.columns([1, 1])

with col_ui_1:
    st.subheader("📸 Tải lên ảnh thiết kế (Grayscale Depth Map)")
    uploaded_file = st.file_uploader(
        "Chọn ảnh PNG/JPG...", type=["png", "jpg", "jpeg"]
    )

# ==============================================================================
# HÀM XỬ LÝ LỚP CẮT & SINH G-CODE CHUẨN GRBL
# ==============================================================================
def generate_layer_gcode(
    contours,
    current_z,
    tool_r,
    origin_offset,
    f_xy,
    f_z,
    s_rpm,
    safe_z_val,
):
    """Sinh mã G-code cho một tầng cắt dựa trên đường biên contour"""
    offset_x, offset_y = origin_offset
    gcode = []

    # G-code Header cho mỗi tầng (Đơn vị mm, Tọa độ tuyệt đối)
    gcode.append("G21 ; Set units to mm")
    gcode.append("G90 ; Absolute positioning")
    gcode.append(f"M3 S{s_rpm} ; Turn spindle CW")
    gcode.append(f"G0 Z{safe_z_val} ; Move to safe Z")

    for contour in contours:
        if len(contour) < 2:
            continue

        # Điểm bắt đầu của đường biên
        start_pt = contour[0][0]
        start_x = start_pt[0] - offset_x
        start_y = offset_y - start_pt[1]  # Đảo trục Y cho đúng hệ tọa độ máy CNC

        # Di chuyển nhanh đến vị trí bắt đầu trên cao, rồi đâm dao xuống độ sâu hiện tại
        gcode.append(f"G0 X{start_x:.3f} Y{start_y:.3f}")
        gcode.append(f"G1 Z{current_z:.3f} F{f_z}")

        # Chạy dao dọc theo các điểm của đường biên
        for pt in contour[1:]:
            x = pt[0][0] - offset_x
            y = offset_y - pt[0][1]
            gcode.append(f"G1 X{x:.3f} Y{y:.3f} F{f_xy}")

        # Nhấc dao an toàn sau khi cắt xong một hốc/biên riêng lẻ
        gcode.append(f"G0 Z{safe_z_val}")

    gcode.append("M5 ; Turn off spindle")
    return "\n".join(gcode)

# ==============================================================================
# XỬ LÝ LOGIC GIA CÔNG KHI CÓ ẢNH
# ==============================================================================
if uploaded_file is not None:
    # Đọc ảnh đầu vào
    image = Image.open(uploaded_file).convert("L")
    img_np = np.array(image)

    # Resize ảnh đồng bộ với kích thước phôi thực tế (Độ phân giải 1mm = 2 pixel để mượt và nhanh)
    scale_px = 2
    img_w = int(stock_x * scale_px)
    img_h = int(stock_y * scale_px)
    img_resized = cv2.resize(img_np, (img_w, img_h))

    # Tính toán tọa độ Gốc (Work Zero Offset) dựa trên lựa chọn của user
    if "Bottom-Left" in work_zero:
        offset_x, offset_y = 0, img_h
    elif "Center" in work_zero:
        offset_x, offset_y = img_w / 2, img_h / 2
    elif "Top-Left" in work_zero:
        offset_x, offset_y = 0, 0
    elif "Top-Right" in work_zero:
        offset_x, offset_y = img_w, 0
    else:  # Bottom-Right
        offset_x, offset_y = img_w, img_h

    # Chuyển đổi tọa độ pixel sang mm thực tế để tính toán offset
    origin_offset = (offset_x / scale_px, offset_y / scale_px)

    with col_ui_2:
        st.subheader("⚙️ Cấu hình Phân Lớp Đa Tầng")
        num_layers = st.slider(
            "Số lượng tầng cắt (Layers):", min_value=2, max_value=10, value=5
        )
        btn_generate = st.button(
            "🚀 Bắt Đầu Lập Trình AI CAM", type="primary"
        )

    if btn_generate:
        st.session_state["gcode_layers"] = {}
        st.session_state["processed_images"] = {}

        # Tính toán phân mảnh độ sâu theo số tầng
        threshold_steps = np.linspace(255, 0, num_layers + 1)
        z_steps = np.linspace(0, -relief_depth, num_layers + 1)

        progress_bar = st.progress(0.0)

        for i in range(num_layers):
            upper_thresh = int(threshold_steps[i])
            lower_thresh = int(threshold_steps[i + 1])
            current_z = z_steps[i + 1]

            # Tạo mặt nạ nhị phân tách lớp đường biên dựa vào sắc độ xám
            mask = cv2.inRange(img_resized, lower_thresh, upper_thresh)

            # Tìm đường biên vector hóa (Contours) phục vụ chạy CNC đường biên
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Tỷ lệ hóa contour từ pixel về kích thước mm thực tế
            scaled_contours = []
            for cnt in contours:
                scaled_contours.append(cnt / scale_px)

            # Sinh mã G-code cho tầng hiện tại
            layer_gcode = generate_layer_gcode(
                contours=scaled_contours,
                current_z=current_z,
                tool_r=tool_dia / 2.0,
                origin_offset=origin_offset,
                f_xy=feed_rate,
                f_z=plunge_rate,
                s_rpm=spindle_speed,
                safe_z_val=safe_z,
            )

            # Lưu trữ vào session_state để hiển thị và tải về
            layer_name = f"Layer_{i+1}_Z{current_z:.1f}mm"
            st.session_state["gcode_layers"][layer_name] = layer_gcode

            # Vẽ hình minh họa đường dao chạy của tầng
            preview_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(preview_img, contours, -1, (0, 255, 0), 2)
            st.session_state["processed_images"][layer_name] = preview_img

            progress_bar.progress((i + 1) / num_layers)

        st.success(
            f"🎉 Đã biên dịch xong {num_layers} tầng cắt chuẩn GRBL hoàn chỉnh!"
        )

    # ==============================================================================
    # KHU VỰC HIỂN THỊ KẾT QUẢ VÀ TẢI FILE G-CODE
    # ==============================================================================
    if st.session_state["gcode_layers"]:
        st.markdown("---")
        st.header("📦 Kết Quả Lập Trình & Tải Xuống G-Code")

        # Tạo file nén ZIP chứa tất cả các tầng G-Code trong bộ nhớ RAM
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for layer_name, gcode_content in st.session_state[
                "gcode_layers"
            ].items():
                zip_file.writestr(f"{layer_name}.nc", gcode_content)

        st.download_button(
            label="💾 Tải Toàn Bộ Lớp Cắt (File .ZIP)",
            data=zip_buffer.getvalue(),
            file_name="AI_CAM_Wood_Layers.zip",
            mime="application/zip",
            type="secondary",
        )

        # Xem trước từng tầng trực quan
        tabs = st.tabs(list(st.session_state["gcode_layers"].keys()))
        for idx, (layer_name, gcode_content) in enumerate(
            st.session_state["gcode_layers"].items()
        ):
            with tabs[idx]:
                col_tab_1, col_tab_2 = st.columns([1, 1])
                with col_tab_1:
                    st.image(
                        st.session_state["processed_images"][layer_name],
                        caption=f"Đường biên chạy dao của {layer_name}",
                        use_container_width=True,
                    )
                with col_tab_2:
                    st.text_area(
                        f"Xem trước mã lệnh G-code ({layer_name})",
                        gcode_content,
                        height=300,
                    )

else:
    st.info(
        "💡 Vui lòng tải một hình ảnh Depth Map (Tranh thang xám) lên để bắt đầu phân lớp tự động."
    )
