import streamlit as st
import numpy as np
import cv2
from PIL import Image
import math
import torch

# -----------------------------------------------------------------------------
# TỐI ƯU HÓA HỆ THỐNG
# -----------------------------------------------------------------------------
torch.set_num_threads(2)

st.set_page_config(
    page_title="AI CNC Wood Carving Studio (GRBL / UGS)",
    page_icon="🪵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App Custom Styling
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #5A3E2B; margin-bottom: 0px; }
    .sub-title { font-size: 1.0rem; color: #8C6D53; margin-bottom: 20px; }
    .ai-badge { background-color: #E0F2FE; color: #0369A1; font-weight: bold; padding: 4px 8px; border-radius: 4px; font-size: 0.85rem; }
    .warning-badge { background-color: #FEF3C7; color: #B45309; font-weight: bold; padding: 4px 8px; border-radius: 4px; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🪵 AI CNC Wood Carving Studio</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Hệ thống xử lý ảnh AI & Tự động sinh G-code Chuyển đổi Tranh Gỗ 2D/3D (Chuẩn GRBL & Universal Gcode Sender - UGS)</p>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
if 'processed_img' not in st.session_state:
    st.session_state.processed_img = None
if 'original_img' not in st.session_state:
    st.session_state.original_img = None
if 'depth_map' not in st.session_state:
    st.session_state.depth_map = None

# -----------------------------------------------------------------------------
# SIDEBAR - BOARD & STOCK DIMENSIONS & WORK PIECE SETTINGS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ 1. Thông Số Phôi & Khổ Ván")
    
    st.subheader("📋 Tấm Ván Tổng (Sheet)")
    board_w = st.number_input("Chiều rộng ván X (mm)", value=1200.0, step=50.0, min_value=100.0)
    board_h = st.number_input("Chiều dài ván Y (mm)", value=800.0, step=50.0, min_value=100.0)
    board_z = st.number_input("Độ dày ván Z (mm)", value=18.0, step=1.0, min_value=1.0)
    
    st.subheader("🪵 Phôi Gia Công (Workpiece)")
    stock_w = st.number_input("Rộng phôi X (mm)", value=300.0, step=10.0, min_value=10.0, max_value=board_w)
    stock_h = st.number_input("Dài phôi Y (mm)", value=400.0, step=10.0, min_value=10.0, max_value=board_h)
    target_depth = st.number_input("Độ sâu khắc tối đa Z (mm)", value=10.0, step=0.5, min_value=0.5, max_value=board_z)
    
    st.subheader("📍 Tọa Độ Mốc (Zero Origin)")
    offset_x = st.number_input("Vị trí X trên ván (mm)", value=50.0, step=5.0, max_value=max(board_w-stock_w, 0.0))
    offset_y = st.number_input("Vị trí Y trên ván (mm)", value=50.0, step=5.0, max_value=max(board_h-stock_h, 0.0))
    z_safe = st.number_input("Mặt phẳng an toàn Z-Safe (mm)", value=5.0, step=1.0, min_value=1.0)
    
    st.markdown("---")
    st.info("💡 **Ghi chú GRBL/UGS:** G-code sinh ra sử dụng hệ tọa độ tương đối/tuyệt đối chuẩn `G90`, đơn vị `G21` (mm) tương thích hoàn toàn với UGS, Candle và Mach3.")

# -----------------------------------------------------------------------------
# CACHED AI & PIPELINE FUNCTIONS
# -----------------------------------------------------------------------------
@st.cache_resource
def load_midas_depth_model():
    model_type = "MiDaS_small"
    midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
    device = torch.device("cpu")
    midas.to(device)
    midas.eval()
    
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    transform = midas_transforms.small_transform
    return midas, transform, device

@st.cache_data(show_spinner=False)
def ai_stage_1_processing(img_bytes, sharpness=2.0, contrast=1.5, denoise=True):
    image_pil = Image.open(img_bytes)
    img_work = image_pil.copy()
    img_work.thumbnail((800, 800))  # Tối ưu size vừa đủ nét nhưng nhẹ RAM
    
    img_array = np.array(img_work.convert('RGB'))
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    if denoise:
        # Giảm hLuminance xuống 5 để tăng tốc độ xử lý OpenCV
        img_bgr = cv2.fastNlMeansDenoisingColored(img_bgr, None, 5, 5, 7, 21)
    
    gaussian = cv2.GaussianBlur(img_bgr, (0, 0), 3.0)
    sharpened = cv2.addWeighted(img_bgr, 1.0 + (sharpness * 0.5), gaussian, -(sharpness * 0.5), 0)
    
    lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=contrast * 2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl, a, b))
    enhanced_bgr = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    return Image.fromarray(cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB))

@st.cache_data(show_spinner=False)
def ai_stage_2_depth_map(enhanced_pil_img):
    midas, transform, device = load_midas_depth_model()
    
    img_resized = enhanced_pil_img.copy()
    img_resized.thumbnail((256, 256))  # 256px cho phản hồi AI tức thì
    
    img_array = np.array(img_resized.convert('RGB'))
    input_batch = transform(img_array).to(device)
    
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(enhanced_pil_img.height, enhanced_pil_img.width),
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        
    depth_np = prediction.cpu().numpy()
    depth_normalized = cv2.normalize(depth_np, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return 255 - depth_normalized

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS: G-CODE GENERATOR (CACHED TỐI ƯU SPEED)
# -----------------------------------------------------------------------------
def optimize_depth_map_for_gcode(depth_map, max_dim=200):
    h, w = depth_map.shape
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(depth_map, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return depth_map

@st.cache_data(show_spinner=False)
def generate_roughing_gcode(depth_map, stock_w, stock_h, target_depth, tool_dia, stepover_pct, stepdown, feedrate, spindle_rpm, z_safe):
    dmap = optimize_depth_map_for_gcode(depth_map, max_dim=150)
    h, w = dmap.shape
    
    lines = [
        "(--- LAYER 1: PHA THO 3D / ROUGHING CARVING ---)",
        "(G-code sinh cho GRBL + UGS)",
        "G21 ; Don vi mm",
        "G90 ; Toa do tuyet doi",
        f"M03 S{int(spindle_rpm)} ; Bat truc chinh RPM",
        f"G00 Z{z_safe:.3f} ; Nac dao an toan"
    ]
    
    step_x = max(tool_dia * (stepover_pct / 100.0), 0.5)
    step_y = max(tool_dia * (stepover_pct / 100.0), 0.5)
    cols = max(1, int(stock_w / step_x))
    rows = max(1, int(stock_h / step_y))
    num_passes = math.ceil(target_depth / stepdown)
    
    for current_pass in range(1, num_passes + 1):
        pass_z = min(current_pass * stepdown, target_depth)
        lines.append(f"(; --- Luot pha tho depth = -{pass_z:.2f} mm ---)")
        for r in range(0, rows):
            y_pos = r * step_y
            py = min(int((y_pos / stock_h) * (h - 1)), h - 1)
            x_range = range(0, cols) if r % 2 == 0 else range(cols - 1, -1, -1)
            
            for c in x_range:
                x_pos = c * step_x
                px = min(int((x_pos / stock_w) * (w - 1)), w - 1)
                normalized_depth = (dmap[py, px] / 255.0) * pass_z
                z_pos = -normalized_depth
                
                if c == x_range[0]:
                    lines.append(f"G00 X{x_pos:.3f} Y{y_pos:.3f}")
                    lines.append(f"G01 Z{z_pos:.3f} F{int(feedrate/2)}")
                else:
                    lines.append(f"G01 X{x_pos:.3f} Y{y_pos:.3f} Z{z_pos:.3f} F{int(feedrate)}")
        lines.append(f"G00 Z{z_safe:.3f}")

    lines.extend(["G00 Z%.3f" % z_safe, "M05", "G00 X0 Y0", "M30"])
    return "\n".join(lines)

@st.cache_data(show_spinner=False)
def generate_finishing_gcode(depth_map, stock_w, stock_h, target_depth, tool_dia, stepover_pct, feedrate, spindle_rpm, z_safe):
    dmap = optimize_depth_map_for_gcode(depth_map, max_dim=250)
    h, w = dmap.shape
    
    lines = [
        "(--- LAYER 2: KHAC TINH 3D / FINISHING CARVING ---)",
        "(G-code sinh cho GRBL + UGS)",
        "G21", "G90",
        f"M03 S{int(spindle_rpm)}",
        f"G00 Z{z_safe:.3f}"
    ]
    
    step_x = max(tool_dia * (stepover_pct / 100.0), 0.2)
    step_y = max(tool_dia * (stepover_pct / 100.0), 0.2)
    cols = max(1, int(stock_w / step_x))
    rows = max(1, int(stock_h / step_y))
    
    for r in range(0, rows):
        y_pos = r * step_y
        py = min(int((y_pos / stock_h) * (h - 1)), h - 1)
        x_range = range(0, cols) if r % 2 == 0 else range(cols - 1, -1, -1)
        
        for c in x_range:
            x_pos = c * step_x
            px = min(int((x_pos / stock_w) * (w - 1)), w - 1)
            z_pos = -((dmap[py, px] / 255.0) * target_depth)
            
            if c == x_range[0]:
                lines.append(f"G00 X{x_pos:.3f} Y{y_pos:.3f}")
                lines.append(f"G01 Z{z_pos:.3f} F{int(feedrate/2)}")
            else:
                lines.append(f"G01 X{x_pos:.3f} Y{y_pos:.3f} Z{z_pos:.3f} F{int(feedrate)}")
                
    lines.extend(["G00 Z%.3f" % z_safe, "M05", "G00 X0 Y0", "M30"])
    return "\n".join(lines)

@st.cache_data(show_spinner=False)
def generate_cutout_gcode(stock_w, stock_h, stock_thickness, tool_dia, stepdown, feedrate, spindle_rpm, z_safe, tab_width, tab_height, tab_count):
    lines = [
        "(--- LAYER 3: CAT BIEN & TAO TAB / CUTOUT CONTOUR ---)",
        "G21", "G90",
        f"M03 S{int(spindle_rpm)}",
        f"G00 Z{z_safe:.3f}"
    ]
    
    r = tool_dia / 2.0
    x0, y0 = -r, -r
    x1, y1 = stock_w + r, stock_h + r
    num_passes = math.ceil(stock_thickness / stepdown)
    
    for p in range(1, num_passes + 1):
        current_z = -min(p * stepdown, stock_thickness)
        lines.append(f"(; --- Luot cat depth = {current_z:.2f} mm ---)")
        path_segments = [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))]
        
        lines.append(f"G00 X{x0:.3f} Y{y0:.3f}")
        lines.append(f"G01 Z{current_z:.3f} F{int(feedrate/2)}")
        
        for p_start, p_end in path_segments:
            is_final_pass = (p == num_passes) or (abs(current_z) >= (stock_thickness - tab_height))
            if is_final_pass:
                mid_x = (p_start[0] + p_end[0]) / 2.0
                mid_y = (p_start[1] + p_end[1]) / 2.0
                tab_z = min(current_z + tab_height, 0.0)
                
                lines.append(f"G01 X{mid_x - tab_width/2:.3f} Y{mid_y:.3f} Z{current_z:.3f} F{int(feedrate)}")
                lines.append(f"G01 Z{tab_z:.3f} F{int(feedrate/2)}")
                lines.append(f"G01 X{mid_x + tab_width/2:.3f} Y{mid_y:.3f} F{int(feedrate)}")
                lines.append(f"G01 Z{current_z:.3f} F{int(feedrate/2)}")
                lines.append(f"G01 X{p_end[0]:.3f} Y{p_end[1]:.3f} F{int(feedrate)}")
            else:
                lines.append(f"G01 X{p_end[0]:.3f} Y{p_end[1]:.3f} F{int(feedrate)}")
                
    lines.extend(["G00 Z%.3f" % z_safe, "M05", "G00 X0 Y0", "M30"])
    return "\n".join(lines)

# -----------------------------------------------------------------------------
# MAIN INTERFACE & WORKFLOW
# -----------------------------------------------------------------------------
tab_upload, tab_layers, tab_3d = st.tabs([
    "🖼️ 1. Upload & AI Xử Lý Ảnh",
    "🔲 2. Phân Layer Gia Công & Sinh G-Code",
    "🖥️ 3. Dashboard Mô Phỏng Trực Quan 3D Ván"
])

with tab_upload:
    st.subheader("1. Tải Lên Ảnh Tranh Gỗ Mẫu & Xử Lý AI Siêu Nét")
    uploaded_file = st.file_uploader("Chọn ảnh bức tranh (JPG, PNG, WEBP)", type=["jpg", "jpeg", "png", "webp"])
    
    if uploaded_file:
        st.session_state.original_img = Image.open(uploaded_file)
        
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
        with col_ctrl1:
            sharp_val = st.slider("Độ sắc nét AI (Sharpness)", 0.5, 4.0, 2.0, 0.1)
        with col_ctrl2:
            contrast_val = st.slider("Độ tương phản (Contrast)", 0.8, 2.5, 1.4, 0.1)
        with col_ctrl3:
            denoise_chk = st.checkbox("Khử nhiễu ảnh (Denoise)", value=True)
            
        if st.button("🚀 Kích Hoạt AI Xử Lý Ảnh (Tốc Độ Siêu Tốc)", type="primary"):
            with st.spinner("AI đang xử lý Tầng 1 & Tầng 2 Depth Map..."):
                uploaded_file.seek(0)
                bytes_data = uploaded_file.read()
                
                enhanced_img = ai_stage_1_processing(
                    bytes_data, 
                    sharpness=sharp_val, 
                    contrast=contrast_val, 
                    denoise=denoise_chk
                )
                depth_map = ai_stage_2_depth_map(enhanced_img)
                
                st.session_state.processed_img = enhanced_img
                st.session_state.depth_map = depth_map
                st.success("Xử lý ảnh bằng AI Deep Learning hoàn tất!")
                
        if st.session_state.original_img is not None:
            st.markdown("---")
            st.markdown("#### 🔍 Đối Chiếu So Sánh Ảnh Gốc vs Ảnh AI Đã Xử Lý")
            col_img1, col_img2 = st.columns(2)
            
            with col_img1:
                st.image(st.session_state.original_img, caption="📷 Ảnh Gốc Tải Lên", use_container_width=True)
                
            with col_img2:
                if st.session_state.processed_img is not None:
                    st.image(st.session_state.processed_img, caption="✨ Ảnh AI Tầng 1 (Enhanced & Contrast)", use_container_width=True)
                else:
                    st.info("Nhấn 'Kích Hoạt AI Xử Lý Ảnh' để xem kết quả.")
                    
            if st.session_state.depth_map is not None:
                st.markdown("#### 🗺️ Bản Đồ Độ Sâu 3D Thực Sự (AI Neural Network MiDaS Depth Map)")
                st.image(st.session_state.depth_map, caption="Heightmap 3D trích xuất cho dao CNC", use_container_width=True)

with tab_layers:
    st.subheader("2. Quản Lý Layer Gia Công & Sinh G-Code Cho GRBL / UGS")
    
    if st.session_state.depth_map is None:
        st.warning("⚠️ Vui lòng tải ảnh và kích hoạt AI xử lý tại Tab 1 trước khi cấu hình Layer.")
    else:
        st.markdown("AI đã tự động phân tích ảnh và sinh **3 Layer Gia Công Chuẩn CNC**:")
        
        # LAYER 1: PHA THÔ 3D
        with st.expander("🔨 LAYER 1: PHA THÔ 3D (ROUGHING CARVING)", expanded=True):
            st.markdown('<span class="ai-badge">🤖 AI Advisor: Đề xuất sử dụng dao Endmill 6mm / Phá vạt nhanh vùng gỗ thừa.</span>', unsafe_allow_html=True)
            
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                l1_tool_type = st.selectbox("Loại dao", ["Endmill (Dao bằng)", "Bullnose", "Flycutter"], index=0, key="l1_type")
                l1_tool_dia = st.number_input("Đường kính dao (mm)", value=6.0, step=0.5, key="l1_dia")
            with c2:
                l1_stepdown = st.number_input("Độ sâu mỗi lượt Z (mm)", value=3.0, step=0.5, key="l1_sd")
                l1_stepover = st.slider("Dịch dao % (Stepover)", 10, 80, 40, key="l1_so")
            with c3:
                l1_feed = st.number_input("Tốc độ tiến dao F (mm/min)", value=1800, step=100, key="l1_feed")
                l1_rpm = st.number_input("Tốc độ trục S (RPM)", value=15000, step=1000, key="l1_rpm")
            with c4:
                st.markdown("**AI Consultation Safety:**")
                if l1_stepdown > l1_tool_dia / 2:
                    st.markdown('<span class="warning-badge">⚠️ Cảnh báo: Độ sâu Z quá lớn dễ gãy dao gỗ cứng!</span>', unsafe_allow_html=True)
                else:
                    st.success("✅ Thông số an toàn tối ưu cho gỗ MDF/Gụ/Hương.")
            
            if st.button("⚙️ Sinh G-Code Layer 1", key="btn_g1"):
                gcode_l1 = generate_roughing_gcode(
                    st.session_state.depth_map, stock_w, stock_h, target_depth,
                    l1_tool_dia, l1_stepover, l1_stepdown, l1_feed, l1_rpm, z_safe
                )
                st.download_button(
                    label="📥 Tải G-Code Layer 1 (Layer1_Roughing.nc)",
                    data=gcode_l1,
                    file_name="Layer1_Roughing.nc",
                    mime="text/plain"
                )

        # LAYER 2: KHẮC TINH 3D
        with st.expander("✨ LAYER 2: KHẮC TINH CHI TIẾT 3D (FINISHING CARVING)", expanded=False):
            st.markdown('<span class="ai-badge">🤖 AI Advisor: Đề xuất Dao Cầu / Tapered Ballnose 2mm R0.5 / Độ nét tinh xảo.</span>', unsafe_allow_html=True)
            
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                l2_tool_type = st.selectbox("Loại dao", ["Tapered Ballnose (Dao cầu nón)", "Ballnose (Dao cầu)", "V-Bit 15°"], index=0, key="l2_type")
                l2_tool_dia = st.number_input("Đường kính dao (mm)", value=2.0, step=0.1, key="l2_dia")
            with c2:
                l2_stepover = st.slider("Dịch dao % (Stepover tinh)", 5, 25, 10, key="l2_so")
                l2_feed = st.number_input("Tốc độ tiến dao F (mm/min)", value=2200, step=100, key="l2_feed")
            with c3:
                l2_rpm = st.number_input("Tốc độ trục S (RPM)", value=18000, step=1000, key="l2_rpm")
            with c4:
                st.markdown("**AI Consultation Safety:**")
                st.success("✅ Stepover 10% giúp bề mặt mịn không cần xả nhám.")
            
            if st.button("⚙️ Sinh G-Code Layer 2", key="btn_g2"):
                gcode_l2 = generate_finishing_gcode(
                    st.session_state.depth_map, stock_w, stock_h, target_depth,
                    l2_tool_dia, l2_stepover, l2_feed, l2_rpm, z_safe
                )
                st.download_button(
                    label="📥 Tải G-Code Layer 2 (Layer2_Finishing.nc)",
                    data=gcode_l2,
                    file_name="Layer2_Finishing.nc",
                    mime="text/plain"
                )

        # LAYER 3: CẮT BIÊN & TẠO TAB
        with st.expander("✂️ LAYER 3: CẮT BIÊN & TẠO CẦU GIỮ PHÔI (CUTOUT & TABS)", expanded=False):
            st.markdown('<span class="ai-badge">🤖 AI Advisor: Tự động tính toán Tab cầu giữ chống văng phôi khi đứt ván.</span>', unsafe_allow_html=True)
            
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                l3_tool_dia = st.number_input("Đường kính dao cắt (mm)", value=6.0, step=0.5, key="l3_dia")
                l3_stepdown = st.number_input("Độ sâu cắt mỗi lượt (mm)", value=3.0, step=0.5, key="l3_sd")
            with c2:
                tab_width = st.number_input("Chiều rộng Tab giữ (mm)", value=8.0, step=1.0, key="tab_w")
                tab_height = st.number_input("Chiều cao Tab giữ (mm)", value=4.0, step=0.5, key="tab_h")
            with c3:
                tab_count = st.number_input("Số lượng Tab quanh chu vi", value=4, min_value=2, max_value=12, key="tab_c")
                l3_feed = st.number_input("Tốc độ cắt F (mm/min)", value=1200, step=100, key="l3_feed")
            with c4:
                l3_rpm = st.number_input("Tốc độ S (RPM)", value=16000, step=1000, key="l3_rpm")
                st.info(f"Tổng độ sâu cắt biên: {board_z} mm ({math.ceil(board_z/l3_stepdown)} lượt cắt)")
            
            if st.button("⚙️ Sinh G-Code Layer 3", key="btn_g3"):
                gcode_l3 = generate_cutout_gcode(
                    stock_w, stock_h, board_z, l3_tool_dia, l3_stepdown,
                    l3_feed, l3_rpm, z_safe, tab_width, tab_height, tab_count
                )
                st.download_button(
                    label="📥 Tải G-Code Layer 3 (Layer3_Cutout_Tabs.nc)",
                    data=gcode_l3,
                    file_name="Layer3_Cutout_Tabs.nc",
                    mime="text/plain"
                )

with tab_3d:
    st.subheader("3. Mô Phỏng Chi Tiết Gia Công Nằm Trong Tấm Ván")
    st.write(f"**Khổ ván gỗ tổng:** {board_w} x {board_h} x {board_z} mm | **Phôi gia công:** {stock_w} x {stock_h} x {target_depth} mm")
    
    scale = 0.5
    svg_w = int(board_w * scale)
    svg_h = int(board_h * scale)
    
    sx = int(offset_x * scale)
    sy = int(offset_y * scale)
    sw = int(stock_w * scale)
    sh = int(stock_h * scale)
    
    svg_content = f"""
    <svg width="{svg_w}" height="{svg_h}" style="background-color: #D2B48C; border: 3px solid #8B4513; border-radius: 8px;">
        <defs>
            <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
                <path d="M 20 0 L 0 0 0 20" fill="none" stroke="#C19A6B" stroke-width="0.5"/>
            </pattern>
        </defs>
        <rect width="100%" height="100%" fill="url(#grid)" />
        <rect x="{sx}" y="{sy}" width="{sw}" height="{sh}" fill="#A0522D" stroke="#5C2C16" stroke-width="2" rx="4" opacity="0.85"/>
        <circle cx="{sx}" cy="{sy}" r="6" fill="#FF0000" />
        <line x1="{sx}" y1="{sy}" x2="{sx + 30}" y2="{sy}" stroke="#FF0000" stroke-width="2" />
        <line x1="{sx}" y1="{sy}" x2="{sx}" y2="{sy + 30}" stroke="#00FF00" stroke-width="2" />
        <text x="{sx + 8}" y="{sy - 8}" fill="#000000" font-weight="bold" font-size="12">G54 (X0, Y0)</text>
        <rect x="{sx + sw/2 - 10}" y="{sy - 2}" width="20" height="4" fill="#00FF00" />
        <rect x="{sx + sw/2 - 10}" y="{sy + sh - 2}" width="20" height="4" fill="#00FF00" />
        <rect x="{sx - 2}" y="{sy + sh/2 - 10}" width="4" height="20" fill="#00FF00" />
        <rect x="{sx + sw - 2}" y="{sy + sh/2 - 10}" width="4" height="20" fill="#00FF00" />
        <text x="{sx + 10}" y="{sy + 25}" fill="#FFFFFF" font-size="14" font-weight="bold">Tranh Khắc CNC ({stock_w}x{stock_h}mm)</text>
        <text x="10" y="{svg_h - 10}" fill="#3D2314" font-size="12">Tấm Ván Tổng: {board_w} x {board_h} mm</text>
    </svg>
    """
    
    st.components.v1.html(svg_content, height=svg_h + 30)
    
    st.markdown("""
    **💡 Chú thích trực quan Dashboard:**
    - 🟫 **Vùng màu nâu vàng ngoài:** Tấm ván nguyên khổ ($1200 \\times 800$ mm).
    - 🟧 **Khu vực phôi khắc 3D:** Vị trí bức tranh đặt trong tấm ván ($300 \\times 400$ mm).
    - 🔴 **Điểm đỏ (G54):** Mốc tọa độ X0, Y0, Z0 cài đặt trên Universal Gcode Sender (UGS).
    - 🟢 **Vạch xanh lá:** Vị trí các cầu giữ phôi (Tabs) chống rơi/văng phôi sau khi cắt đứt.
    """)
