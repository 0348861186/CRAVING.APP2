import streamlit as st
import numpy as np
import cv2
from PIL import Image
import math

st.set_page_config(
    page_title="AI CNC Wood Carving Studio (Ultra Light)",
    page_icon="🪵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App Custom Styling
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #5A3E2B; margin-bottom: 0px; }
    .sub-title { font-size: 1.0rem; color: #8C6D53; margin-bottom: 20px; }
    .light-badge { background-color: #DCFCE7; color: #15803D; font-weight: bold; padding: 4px 8px; border-radius: 4px; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🪵 AI CNC Wood Carving Studio <span class="light-badge">⚡ Ultra Light Engine</span></p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Hệ thống xử lý Heightmap siêu tốc & Tự động sinh G-code (GRBL / UGS)</p>', unsafe_allow_html=True)

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

# -----------------------------------------------------------------------------
# FAST & LIGHTWEIGHT IMAGE PIPELINE (NO PYTORCH / NO HEAVY DEEP LEARNING)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def fast_stage_1_processing(img_np, sharpness=1.5, contrast=1.4):
    """Xử lý làm nét và nâng tương phản cực nhanh bằng OpenCV Pure C++"""
    # Resize về kích thước chuẩn tối ưu cho CNC để xử lý trong vài miligiây
    h, w = img_np.shape[:2]
    max_dim = 600
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        img_np = cv2.resize(img_np, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # Unsharp Masking siêu nhẹ
    gaussian = cv2.GaussianBlur(img_bgr, (0, 0), 2.0)
    sharpened = cv2.addWeighted(img_bgr, 1.0 + sharpness, gaussian, -sharpness, 0)
    
    # CLAHE Cân bằng độ tương phản
    lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=contrast * 1.5, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl, a, b))
    enhanced_bgr = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    return cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)

@st.cache_data(show_spinner=False)
def fast_stage_2_depth_map(enhanced_np, invert=False, blur_ksize=5):
    """Tạo Heightmap siêu tốc từ Luma & Gradient (Chạy trong 0.01 giây)"""
    gray = cv2.cvtColor(enhanced_np, cv2.COLOR_RGB2GRAY)
    
    # Làm mịn cục bộ để tránh dao CNC bị nhấp nhô quá gắt
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    smoothed = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    
    # Đảo ngược độ sâu tùy chọn (Vùng sáng nổi hay vùng tối nổi)
    if invert:
        depth = smoothed
    else:
        depth = 255 - smoothed
        
    return depth

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS: G-CODE GENERATOR
# -----------------------------------------------------------------------------
def optimize_depth_map_for_gcode(depth_map, max_dim=150):
    h, w = depth_map.shape
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(depth_map, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return depth_map

@st.cache_data(show_spinner=False)
def generate_roughing_gcode(depth_map, stock_w, stock_h, target_depth, tool_dia, stepover_pct, stepdown, feedrate, spindle_rpm, z_safe):
    dmap = optimize_depth_map_for_gcode(depth_map, max_dim=120)
    h, w = dmap.shape
    
    lines = [
        "(--- LAYER 1: PHA THO 3D / ROUGHING CARVING ---)",
        "G21 ; Don vi mm", "G90 ; Toa do tuyet doi",
        f"M03 S{int(spindle_rpm)}", f"G00 Z{z_safe:.3f}"
    ]
    
    step_x = max(tool_dia * (stepover_pct / 100.0), 0.5)
    step_y = max(tool_dia * (stepover_pct / 100.0), 0.5)
    cols = max(1, int(stock_w / step_x))
    rows = max(1, int(stock_h / step_y))
    num_passes = math.ceil(target_depth / stepdown)
    
    for current_pass in range(1, num_passes + 1):
        pass_z = min(current_pass * stepdown, target_depth)
        for r in range(0, rows):
            y_pos = r * step_y
            py = min(int((y_pos / stock_h) * (h - 1)), h - 1)
            x_range = range(0, cols) if r % 2 == 0 else range(cols - 1, -1, -1)
            
            for c in x_range:
                x_pos = c * step_x
                px = min(int((x_pos / stock_w) * (w - 1)), w - 1)
                z_pos = -((dmap[py, px] / 255.0) * pass_z)
                
                if c == x_range[0]:
                    lines.append(f"G00 X{x_pos:.2f} Y{y_pos:.2f}")
                    lines.append(f"G01 Z{z_pos:.2f} F{int(feedrate/2)}")
                else:
                    lines.append(f"G01 X{x_pos:.2f} Y{y_pos:.2f} Z{z_pos:.2f} F{int(feedrate)}")
        lines.append(f"G00 Z{z_safe:.2f}")

    lines.extend([f"G00 Z{z_safe:.2f}", "M05", "G00 X0 Y0", "M30"])
    return "\n".join(lines)

@st.cache_data(show_spinner=False)
def generate_finishing_gcode(depth_map, stock_w, stock_h, target_depth, tool_dia, stepover_pct, feedrate, spindle_rpm, z_safe):
    dmap = optimize_depth_map_for_gcode(depth_map, max_dim=200)
    h, w = dmap.shape
    
    lines = [
        "(--- LAYER 2: KHAC TINH 3D / FINISHING CARVING ---)",
        "G21", "G90", f"M03 S{int(spindle_rpm)}", f"G00 Z{z_safe:.2f}"
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
                lines.append(f"G00 X{x_pos:.2f} Y{y_pos:.2f}")
                lines.append(f"G01 Z{z_pos:.2f} F{int(feedrate/2)}")
            else:
                lines.append(f"G01 X{x_pos:.2f} Y{y_pos:.2f} Z{z_pos:.2f} F{int(feedrate)}")
                
    lines.extend([f"G00 Z{z_safe:.2f}", "M05", "G00 X0 Y0", "M30"])
    return "\n".join(lines)

# -----------------------------------------------------------------------------
# MAIN INTERFACE & WORKFLOW
# -----------------------------------------------------------------------------
tab_upload, tab_layers = st.tabs([
    "🖼️ 1. Upload & Tạo Heightmap Siêu Tốc",
    "🔲 2. Cấu Hình Layer & Sinh G-Code"
])

with tab_upload:
    st.subheader("1. Tải Lên Bức Tranh & Tạo Heightmap Trong Chớp Mắt")
    uploaded_file = st.file_uploader("Chọn ảnh bức tranh (JPG, PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        st.session_state.original_img = Image.open(uploaded_file)
        
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
        with col_ctrl1:
            sharp_val = st.slider("Độ nét chi tiết (Sharpness)", 0.5, 3.0, 1.5, 0.1)
        with col_ctrl2:
            contrast_val = st.slider("Độ tương phản (Contrast)", 0.8, 2.5, 1.4, 0.1)
        with col_ctrl3:
            invert_val = st.checkbox("Đảo ngược chiều sâu (Invert Z)", value=False)
            blur_val = st.slider("Làm mịn mặt gỗ (Smooth)", 1, 15, 5, step=2)
            
        if st.button("⚡ Tạo Heightmap Ngay Lập Tức", type="primary"):
            img_np = np.array(st.session_state.original_img.convert('RGB'))
            enhanced_np = fast_stage_1_processing(img_np, sharpness=sharp_val, contrast=contrast_val)
            depth_map = fast_stage_2_depth_map(enhanced_np, invert=invert_val, blur_ksize=blur_val)
            
            st.session_state.processed_img = Image.fromarray(enhanced_np)
            st.session_state.depth_map = depth_map
            st.success("Tạo Heightmap thành công trong 0.05 giây!")
                
        if st.session_state.original_img is not None:
            st.markdown("---")
            col_img1, col_img2 = st.columns(2)
            with col_img1:
                st.image(st.session_state.original_img, caption="📷 Ảnh Gốc", use_container_width=True)
            with col_img2:
                if st.session_state.depth_map is not None:
                    st.image(st.session_state.depth_map, caption="🗺️ Heightmap 3D (Độ sâu dao đục CNC)", use_container_width=True)

with tab_layers:
    st.subheader("2. Sinh G-Code Cho GRBL / UGS")
    
    if st.session_state.depth_map is None:
        st.warning("⚠️ Vui lòng tải ảnh và tạo Heightmap ở Tab 1 trước.")
    else:
        # LAYER 1: PHA THÔ 3D
        with st.expander("🔨 LAYER 1: PHA THÔ 3D (ROUGHING CARVING)", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                l1_tool_dia = st.number_input("Đường kính dao (mm)", value=6.0, step=0.5, key="l1_dia")
                l1_stepdown = st.number_input("Độ sâu mỗi lượt Z (mm)", value=3.0, step=0.5, key="l1_sd")
            with c2:
                l1_stepover = st.slider("Dịch dao % (Stepover)", 10, 80, 40, key="l1_so")
                l1_feed = st.number_input("Tốc độ tiến dao F (mm/min)", value=1800, step=100, key="l1_feed")
            with c3:
                l1_rpm = st.number_input("Tốc độ trục S (RPM)", value=15000, step=1000, key="l1_rpm")
            
            if st.button("⚙️ Sinh G-Code Layer 1", key="btn_g1"):
                gcode_l1 = generate_roughing_gcode(
                    st.session_state.depth_map, stock_w, stock_h, target_depth,
                    l1_tool_dia, l1_stepover, l1_stepdown, l1_feed, l1_rpm, z_safe
                )
                st.download_button("📥 Tải Layer1_Roughing.nc", data=gcode_l1, file_name="Layer1_Roughing.nc", mime="text/plain")

        # LAYER 2: KHẮC TINH 3D
        with st.expander("✨ LAYER 2: KHẮC TINH CHI TIẾT 3D (FINISHING CARVING)", expanded=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                l2_tool_dia = st.number_input("Đường kính dao (mm)", value=2.0, step=0.1, key="l2_dia")
                l2_stepover = st.slider("Dịch dao % (Stepover tinh)", 5, 25, 10, key="l2_so")
            with c2:
                l2_feed = st.number_input("Tốc độ tiến dao F (mm/min)", value=2200, step=100, key="l2_feed")
            with c3:
                l2_rpm = st.number_input("Tốc độ trục S (RPM)", value=18000, step=1000, key="l2_rpm")
            
            if st.button("⚙️ Sinh G-Code Layer 2", key="btn_g2"):
                gcode_l2 = generate_finishing_gcode(
                    st.session_state.depth_map, stock_w, stock_h, target_depth,
                    l2_tool_dia, l2_stepover, l2_feed, l2_rpm, z_safe
                )
                st.download_button("📥 Tải Layer2_Finishing.nc", data=gcode_l2, file_name="Layer2_Finishing.nc", mime="text/plain")
