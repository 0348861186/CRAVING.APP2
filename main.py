import streamlit as st
import numpy as np
import cv2
from PIL import Image
import math
import io

# Set Page Config
st.set_page_config(
    page_title="AI CNC Wood Carving Studio",
    page_icon="🪵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #5A3E2B; margin-bottom: 0px; }
    .sub-title { font-size: 1.0rem; color: #8C6D53; margin-bottom: 20px; }
    .ai-badge { background-color: #E0F2FE; color: #0369A1; font-weight: bold; padding: 4px 8px; border-radius: 4px; font-size: 0.85rem; }
    .warning-badge { background-color: #FEF3C7; color: #B45309; font-weight: bold; padding: 4px 8px; border-radius: 4px; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🪵 AI CNC Wood Carving Studio</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Hệ thống xử lý ảnh AI MiDaS 3D & Tự động sinh G-code Chuyển đổi Tranh Gỗ (Chuẩn GRBL & UGS)</p>', unsafe_allow_html=True)

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
# SIDEBAR - BOARD & STOCK DIMENSIONS
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
# LAZY-LOADING OPTIMIZED AI PIPELINE (GIỮ NGUYÊN MIDAS AI)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_midas_model():
    """Tải mô hình MiDaS một lần duy nhất vào bộ nhớ cache"""
    import torch
    torch.set_num_threads(1)  # Giới hạn 1 thread để không đơ CPU/Streamlit
    model_type = "MiDaS_small"
    midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
    device = torch.device("cpu")
    midas.to(device)
    midas.eval()
    
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    transform = midas_transforms.small_transform
    return midas, transform, device

@st.cache_data(show_spinner=False)
def ai_stage_1_processing(img_np, sharpness=2.0, contrast=1.5, denoise=True):
    """Tầng 1: Xử lý làm nét và tương phản OpenCV"""
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    if denoise:
        # Dùng Gaussian thay cho fastNlMeans để nhanh hơn 10 lần
        img_bgr = cv2.GaussianBlur(img_bgr, (3, 3), 0)
    
    gaussian = cv2.GaussianBlur(img_bgr, (0, 0), 2.0)
    sharpened = cv2.addWeighted(img_bgr, 1.0 + (sharpness * 0.5), gaussian, -(sharpness * 0.5), 0)
    
    lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=contrast * 2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl, a, b))
    enhanced_bgr = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    return cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)

@st.cache_data(show_spinner=False)
def ai_stage_2_depth_map(enhanced_np):
    """Tầng 2: Dùng Neural Network MiDaS tính toán Depth Map 3D thực sự"""
    import torch
    midas, transform, device = load_midas_model()
    
    # Resize tạm để AI chạy siêu nhanh trong 1 giây
    h_orig, w_orig = enhanced_np.shape[:2]
    img_pil = Image.fromarray(enhanced_np)
    img_resized = img_pil.copy()
    img_resized.thumbnail((256, 256))
    
    img_array = np.array(img_resized.convert('RGB'))
    input_batch = transform(img_array).to(device)
    
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(h_orig, w_orig),
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        
    depth_np = prediction.cpu().numpy()
    depth_normalized = cv2.normalize(depth_np, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return 255 - depth_normalized

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS: G-CODE GENERATOR (3 LAYERS)
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
        "G21", "G90", f"M03 S{int(spindle_rpm)}", f"G00 Z{z_safe:.2f}"
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

@st.cache_data(show_spinner=False)
def generate_cutout_gcode(stock_w, stock_h, stock_thickness, tool_dia, stepdown, feedrate, spindle_rpm, z_safe, tab_width, tab_height, tab_count):
    lines = [
        "(--- LAYER 3: CAT BIEN & TAO TAB / CUTOUT CONTOUR ---)",
        "G21", "G90", f"M03 S{int(spindle_rpm)}", f"G00 Z{z_safe:.2f}"
    ]
    
    r = tool_dia / 2.0
    x0, y0 = -r, -r
    x1, y1 = stock_w + r, stock_h + r
    num_passes = math.ceil(stock_thickness / stepdown)
    
    for p in range(1, num_passes + 1):
        current_z = -min(p * stepdown, stock_thickness)
        path_segments = [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))]
        
        lines.append(f"G00 X{x0:.2f} Y{y0:.2f}")
        lines.append(f"G01 Z{current_z:.2f} F{int(feedrate/2)}")
        
        for p_start, p_end in path_segments:
            is_final_pass = (p == num_passes) or (abs(current_z) >= (stock_thickness - tab_height))
            if is_final_pass:
                mid_x = (p_start[0] + p_end[0]) / 2.0
                mid_y = (p_start[1] + p_end[1]) / 2.0
                tab_z = min(current_z + tab_height, 0.0)
                
                lines.append(f"G01 X{mid_x - tab_width/2:.2f} Y{mid_y:.2f} Z{current_z:.2f} F{int(feedrate)}")
                lines.append(f"G01 Z{tab_z:.2f} F{int(feedrate/2)}")
                lines.append(f"G01 X{mid_x + tab_width/2:.2f} Y{mid_y:.2f} F{int(feedrate)}")
                lines.append(f"G01 Z{current_z:.2f} F{int(feedrate/2)}")
                lines.append(f"G01 X{p_end[0]:.2f} Y{p_end[1]:.2f} F{int(feedrate)}")
            else:
                lines.append(f"G01 X{p_end[0]:.2f} Y{p_end[1]:.2f} F{int(feedrate)}")
                
    lines.extend([f"G00 Z{z_safe:.2f}", "M05", "G00 X0 Y0", "M30"])
    return "\n".join(lines)

# -----------------------------------------------------------------------------
# MAIN INTERFACE & WORKFLOW
# -----------------------------------------------------------------------------
tab_upload, tab_layers, tab_3d = st.tabs([
    "🖼️ 1. Upload & AI Xử Lý Ảnh (MiDaS 3D)",
    "🔲 2. Phân Layer Gia Công & Sinh G-Code",
    "🖥️ 3. Dashboard Mô Phỏng Trực Quan 3D Ván"
])

with tab_upload:
    st.subheader("1. Tải Lên Ảnh Tranh Gỗ Mẫu & Xử Lý AI Depth Map 3D")
    uploaded_file = st.file_uploader("Chọn ảnh bức tranh (JPG, PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        st.session_state.original_img = Image.open(uploaded_file)
        
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
        with col_ctrl1:
            sharp_val = st.slider("Độ sắc nét AI (Sharpness)", 0.5, 4.0, 2.0, 0.1)
        with col_ctrl2:
            contrast_val = st.slider("Độ tương phản (Contrast)", 0.8, 2.5, 1.4, 0.1)
        with col_ctrl3:
            denoise_chk = st.checkbox("Mịn bề mặt (Denoise)", value=True)
            
        if st.button("🚀 Kích Hoạt AI MiDaS Dựng Depth Map 3D", type="primary"):
            with st.spinner("AI MiDaS đang trích xuất độ sâu 3D..."):
                img_pil = st.session_state.original_img.convert('RGB')
                img_pil.thumbnail((800, 800))
                img_np = np.array(img_pil)
                
                enhanced_np = ai_stage_1_processing(img_np, sharpness=sharp_val, contrast=contrast_val, denoise=denoise_chk)
                depth_map = ai_stage_2_depth_map(enhanced_np)
                
                st.session_state.processed_img = Image.fromarray(enhanced_np)
                st.session_state.depth_map = depth_map
                st.success("Tạo Depth Map 3D thành công!")
                
        if st.session_state.original_img is not None:
            st.markdown("---")
            col_img1, col_img2 = st.columns(2)
            with col_img1:
                st.image(st.session_state.original_img, caption="📷 Ảnh Gốc Tải Lên", use_container_width=True)
            with col_img2:
                if st.session_state.processed_img is not None:
                    st.image(st.session_state.processed_img, caption="✨ Ảnh AI Tầng 1 (Enhanced)", use_container_width=True)
                else:
                    st.info("Nhấn nút kích hoạt AI ở trên để bắt đầu xử lý.")
                    
            if st.session_state.depth_map is not None:
                st.markdown("#### 🗺️ Bản Đồ Độ Sâu 3D Thực Sự (AI MiDaS Neural Network)")
                st.image(st.session_state.depth_map, caption="Heightmap 3D trích xuất cho dao CNC đục", use_container_width=True)

with tab_layers:
    st.subheader("2. Quản Lý Layer Gia Công & Sinh G-Code Cho GRBL / UGS")
    
    if st.session_state.depth_map is None:
        st.warning("⚠️ Vui lòng tải ảnh và kích hoạt AI xử lý tại Tab 1 trước.")
    else:
        # LAYER 1: PHA THÔ 3D
        with st.expander("🔨 LAYER 1: PHA THÔ 3D (ROUGHING CARVING)", expanded=True):
            st.markdown('<span class="ai-badge">🤖 AI Advisor: Dao Endmill 6mm / Phá thô gỗ nhanh.</span>', unsafe_allow_html=True)
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
            st.markdown('<span class="ai-badge">🤖 AI Advisor: Dao Tapered Ballnose 2mm / Đục tinh xảo.</span>', unsafe_allow_html=True)
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

        # LAYER 3: CẮT BIÊN & TẠO TAB
        with st.expander("✂️ LAYER 3: CẮT BIÊN & TẠO CẦU GIỮ PHÔI (CUTOUT & TABS)", expanded=False):
            st.markdown('<span class="ai-badge">🤖 AI Advisor: Tự động tính Tab giữ phôi chống văng khi đứt ván.</span>', unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                l3_tool_dia = st.number_input("Đường kính dao cắt (mm)", value=6.0, step=0.5, key="l3_dia")
                l3_stepdown = st.number_input("Độ sâu cắt mỗi lượt (mm)", value=3.0, step=0.5, key="l3_sd")
            with c2:
                tab_width = st.number_input("Chiều rộng Tab (mm)", value=8.0, step=1.0, key="tab_w")
                tab_height = st.number_input("Chiều cao Tab (mm)", value=4.0, step=0.5, key="tab_h")
            with c3:
                tab_count = st.number_input("Số lượng Tab", value=4, min_value=2, max_value=12, key="tab_c")
                l3_feed = st.number_input("Tốc độ cắt F (mm/min)", value=1200, step=100, key="l3_feed")
            with c4:
                l3_rpm = st.number_input("Tốc độ S (RPM)", value=16000, step=1000, key="l3_rpm")
            
            if st.button("⚙️ Sinh G-Code Layer 3", key="btn_g3"):
                gcode_l3 = generate_cutout_gcode(
                    stock_w, stock_h, board_z, l3_tool_dia, l3_stepdown,
                    l3_feed, l3_rpm, z_safe, tab_width, tab_height, tab_count
                )
                st.download_button("📥 Tải Layer3_Cutout_Tabs.nc", data=gcode_l3, file_name="Layer3_Cutout_Tabs.nc", mime="text/plain")

with tab_3d:
    st.subheader("3. Mô Phỏng Chi Tiết Gia Công Nằm Trong Tấm Ván")
    st.write(f"**Khổ ván gỗ tổng:** {board_w} x {board_h} x {board_z} mm | **Phôi gia công:** {stock_w} x {stock_h} x {target_depth} mm")
    
    scale = 0.5
    svg_w, svg_h = int(board_w * scale), int(board_h * scale)
    sx, sy = int(offset_x * scale), int(offset_y * scale)
    sw, sh = int(stock_w * scale), int(stock_h * scale)
    
    svg_content = f"""
    <svg width="{svg_w}" height="{svg_h}" style="background-color: #D2B48C; border: 3px solid #8B4513; border-radius: 8px;">
        <rect x="{sx}" y="{sy}" width="{sw}" height="{sh}" fill="#A0522D" stroke="#5C2C16" stroke-width="2" opacity="0.85"/>
        <circle cx="{sx}" cy="{sy}" r="6" fill="#FF0000" />
        <text x="{sx + 8}" y="{sy - 8}" fill="#000000" font-weight="bold" font-size="12">G54 (X0, Y0)</text>
        <text x="{sx + 10}" y="{sy + 25}" fill="#FFFFFF" font-size="14" font-weight="bold">Tranh Khắc CNC ({stock_w}x{stock_h}mm)</text>
    </svg>
    """
    st.components.v1.html(svg_content, height=svg_h + 30)
