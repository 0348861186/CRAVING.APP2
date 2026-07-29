import streamlit as st
import cv2
import numpy as np
from PIL import Image
import math
import io

# =============================================================================
# 1. CẤU HÌNH TRANG STREAMLIT
# =============================================================================
st.set_page_config(
    page_title="AI 3D Relief CNC Generator",
    page_icon="🪵",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🪵 Hệ Thống Tạo G-Code Khắc Gỗ 3D Bằng AI")
st.caption("Tối ưu hóa thời gian xử lý AI (1-2s) & Tự động bù bán kính dao cắt biên")

# =============================================================================
# 2. KHỞI TẠO MÔ HÌNH AI MIDAS (CÓ CACHE)
# =============================================================================
@st.cache_resource(show_spinner=False)
def load_midas_model():
    import torch
    # Sử dụng mô hình MiDaS_small nhẹ nhất, tối ưu cho CPU
    model_type = "MiDaS_small"
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    device = torch.device("cpu")
    midas.to(device)
    midas.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform

    return midas, transform, device

# =============================================================================
# 3. XỬ LÝ ẢNH AI 2 TẦNG (TỐI ƯU SIÊU TỐC 1-2 GIÂY)
# =============================================================================
@st.cache_data(show_spinner=False)
def ai_stage_1_processing(img_np, sharpness=2.0, contrast=1.5, denoise=True):
    """ Tầng 1: Làm mịn nhẹ, tăng nét & tương phản (dùng GaussianBlur siêu tốc thay fastNlMeans) """
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # GaussianBlur chỉ tốn ~0.01 giây
    if denoise:
        img_bgr = cv2.GaussianBlur(img_bgr, (3, 3), 0)
    
    # Unsharp Mask làm nét
    gaussian = cv2.GaussianBlur(img_bgr, (0, 0), 2.0)
    sharpened = cv2.addWeighted(img_bgr, 1.0 + (sharpness * 0.5), gaussian, -(sharpness * 0.5), 0)
    
    # CLAHE Cân bằng độ tương phản
    lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=contrast * 2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl, a, b))
    enhanced_bgr = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    return cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)


@st.cache_data(show_spinner=False)
def ai_stage_2_depth_map(enhanced_np):
    """ Tầng 2: Trích xuất Depth Map bằng AI MiDaS (Scale về 256x256 để quẹt cực nhanh) """
    import torch
    
    # Ép 1 thread để không nghẽn CPU của ứng dụng
    torch.set_num_threads(1)
    
    midas, transform, device = load_midas_model()
    
    h_orig, w_orig = enhanced_np.shape[:2]
    img_pil = Image.fromarray(enhanced_np)
    
    # Resize ảnh về 256x256 giúp AI xử lý chỉ mất ~0.8 giây trên CPU
    img_resized = img_pil.resize((256, 256), Image.Resampling.BILINEAR)
    img_array = np.array(img_resized)
    
    input_batch = transform(img_array).to(device)
    
    with torch.no_grad():
        prediction = midas(input_batch)
        depth_np = prediction.squeeze().cpu().numpy()
        
    # Chuẩn hóa về dải 0 - 255
    depth_normalized = cv2.normalize(depth_np, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Nội suy phóng lớn bản đồ độ sâu trở lại kích thước ảnh gốc
    depth_full = cv2.resize(depth_normalized, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
    
    # Đảo ngược màu: Vùng sáng = Khắc sâu, Vùng tối = Nổi cao (Phù hợp đục khắc CNC)
    return 255 - depth_full

# =============================================================================
# 4. TẠO CODE CAM / G-CODE LỚP CẮT VÀ ĐỤC 3D (CÓ BÙ BÁN KÍNH DAO)
# =============================================================================
@st.cache_data(show_spinner=False)
def generate_rough_gcode(stock_w, stock_h, max_depth, tool_dia, stepover, stepdown, feedrate, spindle_rpm, z_safe):
    """ Layer 1: Phay phá thô (Roughing) """
    lines = [
        "(--- LAYER 1: PHAY PHA THO / ROUGHING ---)",
        "G21 ; Don vi mm", "G90 ; Toa do tuyet doi",
        f"M03 S{int(spindle_rpm)}", f"G00 Z{z_safe:.2f}"
    ]
    
    step_xy = tool_dia * (stepover / 100.0)
    num_passes = math.ceil(max_depth / stepdown)
    
    for p in range(1, num_passes + 1):
        z_curr = -min(p * stepdown, max_depth)
        lines.append(f"(; Pass {p}/{num_passes} - Z = {z_curr:.2f}mm)")
        
        y = 0.0
        direction = 1
        while y <= stock_h:
            x_start = 0.0 if direction == 1 else stock_w
            x_end = stock_w if direction == 1 else 0.0
            
            lines.append(f"G00 X{x_start:.2f} Y{y:.2f}")
            lines.append(f"G01 Z{z_curr:.2f} F{int(feedrate/2)}")
            lines.append(f"G01 X{x_end:.2f} F{int(feedrate)}")
            
            y += step_xy
            direction *= -1
            
        lines.append(f"G00 Z{z_safe:.2f}")
        
    lines.extend(["M05", "G00 X0 Y0", "M30"])
    return "\n".join(lines)


@st.cache_data(show_spinner=False)
def generate_finish_3d_gcode(depth_map, stock_w, stock_h, max_depth, tool_dia, stepover, feedrate, spindle_rpm, z_safe):
    """ Layer 2: Phay tinh 3D Relief dựa trên Depth Map """
    lines = [
        "(--- LAYER 2: PHAY TINH 3D / FINISHING ---)",
        "G21 ; Don vi mm", "G90 ; Toa do tuyet doi",
        f"M03 S{int(spindle_rpm)}", f"G00 Z{z_safe:.2f}"
    ]
    
    h_img, w_img = depth_map.shape
    step_xy = tool_dia * (stepover / 100.0)
    
    x_coords = np.arange(0, stock_w, step_xy)
    y_coords = np.arange(0, stock_h, step_xy)
    
    lines.append(f"G00 X0 Y0")
    lines.append(f"G01 Z0.00 F{int(feedrate/2)}")
    
    direction = 1
    for y in y_coords:
        pix_y = int(min(max((y / stock_h) * h_img, 0), h_img - 1))
        
        x_list = x_coords if direction == 1 else x_coords[::-1]
        for x in x_list:
            pix_x = int(min(max((x / stock_w) * w_img, 0), w_img - 1))
            
            # Tính độ sâu Z dựa trên pixel Depth Map
            depth_val = depth_map[pix_y, pix_x] / 255.0
            z_val = - (depth_val * max_depth)
            
            lines.append(f"G01 X{x:.2f} Y{y:.2f} Z{z_val:.2f} F{int(feedrate)}")
            
        direction *= -1
        
    lines.extend([f"G00 Z{z_safe:.2f}", "M05", "G00 X0 Y0", "M30"])
    return "\n".join(lines)


@st.cache_data(show_spinner=False)
def generate_cutout_gcode(stock_w, stock_h, stock_thickness, tool_dia, stepdown, feedrate, spindle_rpm, z_safe, tab_width, tab_height):
    """ Layer 3: Cắt biên có TỰ ĐỘNG BÙ BÁN KÍNH DAO & Tạo Tab giữ phôi """
    lines = [
        "(--- LAYER 3: CAT BIEN & TAO TAB / CUTOUT CONTOUR ---)",
        "G21 ; Don vi mm", "G90 ; Toa do tuyet doi",
        f"M03 S{int(spindle_rpm)}", f"G00 Z{z_safe:.2f}"
    ]
    
    # -------------------------------------------------------------------------
    # BÙ BÁN KÍNH DAO (TOOL RADIUS COMPENSATION)
    # Tọa độ tâm dao lùi ra ngoài biên phôi một khoảng r = tool_dia / 2
    # -------------------------------------------------------------------------
    r = tool_dia / 2.0
    x0, y0 = -r, -r
    x1, y1 = stock_w + r, stock_h + r
    
    num_passes = math.ceil(stock_thickness / stepdown)
    
    for p in range(1, num_passes + 1):
        current_z = -min(p * stepdown, stock_thickness)
        lines.append(f"(; Pass {p}/{num_passes} - Z = {current_z:.2f}mm)")
        
        # 4 Cạnh đường bao ngoài phôi
        path_segments = [
            ((x0, y0), (x1, y0)), # Bottom
            ((x1, y0), (x1, y1)), # Right
            ((x1, y1), (x0, y1)), # Top
            ((x0, y1), (x0, y0))  # Left
        ]
        
        lines.append(f"G00 X{x0:.2f} Y{y0:.2f}")
        lines.append(f"G01 Z{current_z:.2f} F{int(feedrate/2)}")
        
        for p_start, p_end in path_segments:
            # Chỉ tạo Tab ở pass cuối hoặc pass chạm sát đáy
            is_final_pass = (p == num_passes) or (abs(current_z) >= (stock_thickness - tab_height))
            
            if is_final_pass:
                mid_x = (p_start[0] + p_end[0]) / 2.0
                mid_y = (p_start[1] + p_end[1]) / 2.0
                tab_z = min(current_z + tab_height, 0.0)
                
                if p_start[1] == p_end[1]: # Cạnh ngang
                    lines.append(f"G01 X{mid_x - tab_width/2:.2f} Y{p_start[1]:.2f} Z{current_z:.2f} F{int(feedrate)}")
                    lines.append(f"G01 Z{tab_z:.2f} F{int(feedrate/2)}")
                    lines.append(f"G01 X{mid_x + tab_width/2:.2f} F{int(feedrate)}")
                    lines.append(f"G01 Z{current_z:.2f} F{int(feedrate/2)}")
                    lines.append(f"G01 X{p_end[0]:.2f} F{int(feedrate)}")
                else: # Cạnh dọc
                    lines.append(f"G01 X{p_start[0]:.2f} Y{mid_y - tab_width/2:.2f} Z{current_z:.2f} F{int(feedrate)}")
                    lines.append(f"G01 Z{tab_z:.2f} F{int(feedrate/2)}")
                    lines.append(f"G01 Y{mid_y + tab_width/2:.2f} F{int(feedrate)}")
                    lines.append(f"G01 Z{current_z:.2f} F{int(feedrate/2)}")
                    lines.append(f"G01 Y{p_end[1]:.2f} F{int(feedrate)}")
            else:
                lines.append(f"G01 X{p_end[0]:.2f} Y{p_end[1]:.2f} F{int(feedrate)}")
                
    lines.extend([f"G00 Z{z_safe:.2f}", "M05", "G00 X0 Y0", "M30"])
    return "\n".join(lines)

# =============================================================================
# 5. GIAO DIỆN CHÍNH (SIDEBAR & MAIN PANELS)
# =============================================================================

# --- SIDEBAR CẤU HÌNH ---
st.sidebar.header("⚙️ 1. Kích Thước Phôi & An Toàn")
stock_w = st.sidebar.number_input("Chiều rộng phôi (X) [mm]", value=200.0, step=10.0)
stock_h = st.sidebar.number_input("Chiều cao phôi (Y) [mm]", value=300.0, step=10.0)
stock_thickness = st.sidebar.number_input("Độ dày phôi gỗ [mm]", value=20.0, step=1.0)
max_depth = st.sidebar.number_input("Độ sâu đục 3D tối đa [mm]", value=10.0, step=0.5)
z_safe = st.sidebar.number_input("Cao độ an toàn Z-Safe [mm]", value=5.0, step=1.0)

st.sidebar.header("🛠️ 2. Thông Số Dao & Vận Hành")
spindle_rpm = st.sidebar.number_input("Tốc độ trục chính (RPM)", value=18000, step=1000)
feedrate = st.sidebar.number_input("Tốc độ cắt Feedrate [mm/phút]", value=1500, step=100)

st.sidebar.subheader("Dao Phá Thô & Dao Cắt Biên")
rough_tool_dia = st.sidebar.number_input("Đường kính dao phá/cắt (mm)", value=6.0, step=0.5)
rough_stepdown = st.sidebar.number_input("Độ sâu mỗi pass Z-Stepdown (mm)", value=3.0, step=0.5)

st.sidebar.subheader("Dao Cầu Phay Tinh 3D")
finish_tool_dia = st.sidebar.number_input("Đường kính dao cầu (mm)", value=3.175, step=0.175)
finish_stepover = st.sidebar.slider("Độ đè bước Stepover (%)", 5, 50, 15)

st.sidebar.subheader("Cấu Hình Tab Giữ Phôi (Cắt biên)")
tab_width = st.sidebar.number_input("Rộng Tab [mm]", value=10.0, step=1.0)
tab_height = st.sidebar.number_input("Cao Tab [mm]", value=3.0, step=0.5)

# --- KHU VỰC TẢI ẢNH VÀ XỬ LÝ ---
uploaded_file = st.file_uploader("🖼️ Chọn ảnh mẫu cần làm bức phù điêu 3D (JPG/PNG):", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Đọc ảnh
    image_bytes = uploaded_file.read()
    img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_np = np.array(img_pil)

    # Hiển thị thanh chỉnh sửa ảnh Tầng 1
    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        sharpness = st.slider("⚡ Mức độ làm nét (Sharpen)", 0.0, 5.0, 2.0, 0.5)
    with col_ctrl2:
        contrast = st.slider("☯️ Mức độ tương phản (Contrast)", 0.5, 3.0, 1.5, 0.1)

    # Chạy AI
    with st.spinner("🤖 AI đang tối ưu hóa hình ảnh & trích xuất bản đồ 3D (Siêu tốc 1-2s)..."):
        enhanced_img = ai_stage_1_processing(img_np, sharpness=sharpness, contrast=contrast)
        depth_map = ai_stage_2_depth_map(enhanced_img)

    # Hiển thị kết quả 3 Tầng ảnh
    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(img_np, caption="1. Ảnh Gốc Tải Lên", use_container_width=True)
    with c2:
        st.image(enhanced_img, caption="2. Tầng 1: Đã Khử Nhiễu & Tăng Nét", use_container_width=True)
    with c3:
        st.image(depth_map, caption="3. Tầng 2: Bản Đồ Độ Sâu 3D (AI Depth)", use_container_width=True)

    st.markdown("---")
    st.subheader("⚙️ Xuất Mã G-Code Cho Máy CNC")

    tab_l1, tab_l2, tab_l3 = st.tabs(["🔨 Lớp 1: Phá Thô (Roughing)", "✨ Lớp 2: Phay Tinh 3D (Finishing)", "✂️ Lớp 3: Cắt Biên (Cutout)"])

    # XUẤT LỚP 1
    with tab_l1:
        gcode_l1 = generate_rough_gcode(
            stock_w, stock_h, max_depth, rough_tool_dia, 
            stepover=40.0, stepdown=rough_stepdown, feedrate=feedrate, 
            spindle_rpm=spindle_rpm, z_safe=z_safe
        )
        st.text_area("Xem trước G-Code Lớp 1:", gcode_l1[:1000] + "\n...\n(Còn tiếp)", height=200)
        st.download_button("💾 Tải G-Code Lớp 1 (.nc)", gcode_l1, file_name="Layer1_Roughing.nc", mime="text/plain")

    # XUẤT LỚP 2
    with tab_l2:
        gcode_l2 = generate_finish_3d_gcode(
            depth_map, stock_w, stock_h, max_depth, 
            finish_tool_dia, finish_stepover, feedrate, 
            spindle_rpm, z_safe
        )
        st.text_area("Xem trước G-Code Lớp 2:", gcode_l2[:1000] + "\n...\n(Còn tiếp)", height=200)
        st.download_button("💾 Tải G-Code Lớp 2 (.nc)", gcode_l2, file_name="Layer2_Finishing3D.nc", mime="text/plain")

    # XUẤT LỚP 3
    with tab_l3:
        st.info(f"💡 Đã áp dụng bù bán kính dao: Tâm dao sẽ dịch ra ngoài **r = {rough_tool_dia/2:.2f}mm** để bảo toàn chính xác kích thước phôi ({stock_w}x{stock_h}mm).")
        gcode_l3 = generate_cutout_gcode(
            stock_w, stock_h, stock_thickness, rough_tool_dia, 
            stepdown=rough_stepdown, feedrate=feedrate, spindle_rpm=spindle_rpm, 
            z_safe=z_safe, tab_width=tab_width, tab_height=tab_height
        )
        st.text_area("Xem trước G-Code Lớp 3:", gcode_l3, height=200)
        st.download_button("💾 Tải G-Code Lớp 3 (.nc)", gcode_l3, file_name="Layer3_Cutout.nc", mime="text/plain")

else:
    st.info("👆 Vui lòng tải lên một bức ảnh ở ô phía trên để bắt đầu tạo G-Code.")
