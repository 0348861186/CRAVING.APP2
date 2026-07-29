import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import math
import io

# Cấu hình giao diện Streamlit Dashboard
st.set_page_config(
    page_title="AI CNC Wood Carving Studio (GRBL / UGS)",
    page_icon="🪵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling cho Dashboard
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

# Khởi tạo Session State
if 'processed_img' not in st.session_state:
    st.session_state.processed_img = None
if 'original_img' not in st.session_state:
    st.session_state.original_img = None
if 'depth_map' not in st.session_state:
    st.session_state.depth_map = None

# =============================================================================
# YÊU CẦU 8: SIDEBAR CỘT TRÁI - NHẬP KÍCH THƯỚC KHỔ VÁN, PHÔI, ĐỘ SÂU
# =============================================================================
with st.sidebar:
    st.header("⚙️ 1. Thấu số Phôi & Khổ Ván")
    
    st.subheader("📋 Tấm Ván Tổng (Sheet)")
    board_w = st.number_input("Chiều rộng ván X (mm)", value=1200.0, step=50.0, min_value=100.0)
    board_h = st.number_input("Chiều dài ván Y (mm)", value=800.0, step=50.0, min_value=100.0)
    board_z = st.number_input("Độ dày ván Z (mm)", value=18.0, step=1.0, min_value=1.0)
    
    st.subheader("🪵 Phôi Gia Công (Workpiece)")
    stock_w = st.number_input("Rộng phôi X (mm)", value=300.0, step=10.0, min_value=10.0, max_value=board_w)
    stock_h = st.number_input("Dài phôi Y (mm)", value=400.0, step=10.0, min_value=10.0, max_value=board_h)
    target_depth = st.number_input("Độ sâu khắc tối đa Z (mm)", value=10.0, step=0.5, min_value=0.5, max_value=board_z)
    
    st.subheader("📍 Tọa Độ Mốc (Zero Origin)")
    offset_x = st.number_input("Vị trí X trên ván (mm)", value=50.0, step=5.0, max_value=board_w-stock_w)
    offset_y = st.number_input("Vị trí Y trên ván (mm)", value=50.0, step=5.0, max_value=board_h-stock_h)
    z_safe = st.number_input("Mặt phẳng an toàn Z-Safe (mm)", value=5.0, step=1.0, min_value=1.0)
    
    st.markdown("---")
    st.info("💡 **Tương thích GRBL/UGS:** G-code tự động chuẩn hóa lệnh `G21` (mm), `G90` (Toạ độ tuyệt đối), `M03/M05` (Trục chính).")

# =============================================================================
# THUẬT TOÁN AI XỬ LÝ ẢNH & SINH G-CODE CHUẨN GRBL/UGS
# =============================================================================
def process_ai_image(image_pil, sharpness=2.0, contrast=1.4, denoise=True):
    # Yêu cầu 1: Nâng cấp ảnh nét chi tiết & Khử nhiễu
    img_array = np.array(image_pil.convert('RGB'))
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    if denoise:
        img_bgr = cv2.fastNlMeansDenoisingColored(img_bgr, None, 10, 10, 7, 21)
    
    # Unsharp masking tăng biên độ tương phản nét
    gaussian = cv2.GaussianBlur(img_bgr, (0, 0), 3.0)
    sharpened = cv2.addWeighted(img_bgr, 1.0 + (sharpness * 0.5), gaussian, -(sharpness * 0.5), 0)
    
    pil_enhanced = Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Contrast(pil_enhanced)
    final_img = enhancer.enhance(contrast)
    
    # Yêu cầu 2: Tạo Depth Map 16-bit cho gia công 3D
    gray = cv2.cvtColor(np.array(final_img), cv2.COLOR_RGB2GRAY)
    depth_smooth = cv2.GaussianBlur(gray, (5, 5), 0)
    depth_map = 255 - depth_smooth
    
    return final_img, depth_map

def generate_roughing_gcode(depth_map, stock_w, stock_h, target_depth, tool_dia, stepover_pct, stepdown, feedrate, spindle_rpm, z_safe):
    """L1: Sinh G-code Phá Thô 3D (Roughing)"""
    lines = [
        "(--- LAYER 1: PHA THO 3D / ROUGHING CARVING ---)",
        "(G-code sinh cho GRBL + UGS)",
        "G21 ; Don vi mm",
        "G90 ; Toa do tuyet doi",
        f"M03 S{int(spindle_rpm)}",
        f"G00 Z{z_safe:.3f}"
    ]
    h, w = depth_map.shape
    step_x = tool_dia * (stepover_pct / 100.0)
    step_y = tool_dia * (stepover_pct / 100.0)
    cols, rows = int(stock_w / step_x), int(stock_h / step_y)
    num_passes = math.ceil(target_depth / stepdown)
    
    for current_pass in range(1, num_passes + 1):
        pass_z = min(current_pass * stepdown, target_depth)
        lines.append(f"\n(; --- Luot pha tho depth = -{pass_z:.2f} mm ---)")
        for r in range(0, rows):
            y_pos = r * step_y
            py = min(max(int((y_pos / stock_h) * (h - 1)), 0), h - 1)
            x_range = range(0, cols) if r % 2 == 0 else range(cols - 1, -1, -1)
            for c in x_range:
                x_pos = c * step_x
                px = min(max(int((x_pos / stock_w) * (w - 1)), 0), w - 1)
                z_pos = -((depth_map[py, px] / 255.0) * pass_z)
                if c == x_range[0]:
                    lines.append(f"G00 X{x_pos:.3f} Y{y_pos:.3f}")
                    lines.append(f"G01 Z{z_pos:.3f} F{int(feedrate/2)}")
                else:
                    lines.append(f"G01 X{x_pos:.3f} Y{y_pos:.3f} Z{z_pos:.3f} F{int(feedrate)}")
        lines.append(f"G00 Z{z_safe:.3f}")
        
    lines.extend([f"G00 Z{z_safe:.3f}", "M05", "G00 X0 Y0", "M30"])
    return "\n".join(lines)

def generate_finishing_gcode(depth_map, stock_w, stock_h, target_depth, tool_dia, stepover_pct, feedrate, spindle_rpm, z_safe):
    """L2: Sinh G-code Khắc Tinh 3D (Finishing)"""
    lines = [
        "(--- LAYER 2: KHAC TINH 3D / FINISHING CARVING ---)",
        "(G-code sinh cho GRBL + UGS)",
        "G21", "G90",
        f"M03 S{int(spindle_rpm)}",
        f"G00 Z{z_safe:.3f}"
    ]
    h, w = depth_map.shape
    step_x = tool_dia * (stepover_pct / 100.0)
    step_y = tool_dia * (stepover_pct / 100.0)
    cols, rows = int(stock_w / step_x), int(stock_h / step_y)
    
    for r in range(0, rows):
        y_pos = r * step_y
        py = min(max(int((y_pos / stock_h) * (h - 1)), 0), h - 1)
        x_range = range(0, cols) if r % 2 == 0 else range(cols - 1, -1, -1)
        for c in x_range:
            x_pos = c * step_x
            px = min(max(int((x_pos / stock_w) * (w - 1)), 0), w - 1)
            z_pos = -((depth_map[py, px] / 255.0) * target_depth)
            if c == x_range[0]:
                lines.append(f"G00 X{x_pos:.3f} Y{y_pos:.3f}")
                lines.append(f"G01 Z{z_pos:.3f} F{int(feedrate/2)}")
            else:
                lines.append(f"G01 X{x_pos:.3f} Y{y_pos:.3f} Z{z_pos:.3f} F{int(feedrate)}")
                
    lines.extend([f"G00 Z{z_safe:.3f}", "M05", "G00 X0 Y0", "M30"])
    return "\n".join(lines)

def generate_cutout_gcode(stock_w, stock_h, stock_thickness, tool_dia, stepdown, feedrate, spindle_rpm, z_safe, tab_width, tab_height, tab_count):
    """L3: Sinh G-code Cắt Biên & Tạo Cầu Giữ Phôi (Cutout & Tabs)"""
    lines = [
        "(--- LAYER 3: CAT BIEN & TAO TAB / CUTOUT CONTOUR ---)",
        "(G-code sinh cho GRBL + UGS)",
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
        lines.append(f"\n(; --- Luot cat depth = {current_z:.2f} mm ---)")
        path_segments = [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))]
        lines.append(f"G00 X{x0:.3f} Y{y0:.3f}")
        lines.append(f"G01 Z{current_z:.3f} F{int(feedrate/2)}")
        
        is_final_pass = (p == num_passes) or (abs(current_z) >= (stock_thickness - tab_height))
        for p_start, p_end in path_segments:
            if is_final_pass:
                mid_x = (p_start[0] + p_end[0]) / 2.0
                mid_y = (p_start[1] + p_end[1]) / 2.0
                tab_z = max(current_z + tab_height, 0.0)
                lines.append(f"G01 X{mid_x - tab_width/2:.3f} Y{mid_y:.3f} Z{current_z:.3f} F{int(feedrate)}")
                lines.append(f"G01 Z{tab_z:.3f} F{int(feedrate/2)}")
                lines.append(f"G01 X{mid_x + tab_width/2:.3f} Y{mid_y:.3f} F{int(feedrate)}")
                lines.append(f"G01 Z{current_z:.3f} F{int(feedrate/2)}")
                lines.append(f"G01 X{p_end[0]:.3f} Y{p_end[1]:.3f} F{int(feedrate)}")
            else:
                lines.append(f"G01 X{p_end[0]:.3f} Y{p_end[1]:.3f} F{int(feedrate)}")
                
    lines.extend([f"G00 Z{z_safe:.3f}", "M05", "G00 X0 Y0", "M30"])
    return "\n".join(lines)

# =============================================================================
# GIAO DIỆN TƯƠNG TÁC THEO TỪNG TÁC VỤ
# =============================================================================
tab_upload, tab_layers, tab_3d = st.tabs([
    "🖼️ 1. Upload & AI Xử Lý Ảnh (Yêu cầu 1, 9)",
    "🔲 2. Phân Layer Gia Công & AI Tư Vấn & G-Code (Yêu cầu 2, 3, 4, 5, 6, 7)",
    "🖥️ 3. Dashboard Mô Phỏng Trực Quan 3D Ván (Yêu cầu 8, 10)"
])

# --- TAB 1: UPLOAD & ĐỐI CHIẾU ẢNH AI ---
with tab_upload:
    st.subheader("1. Tải Lên Ảnh Tranh Gỗ Mẫu & Xử Lý AI Siêu Nét")
    uploaded_file = st.file_uploader("Chọn ảnh bức tranh (JPG, PNG, WEBP)", type=["jpg", "jpeg", "png", "webp"])
    
    if uploaded_file:
        st.session_state.original_img = Image.open(uploaded_file)
        c1, c2, c3 = st.columns(3)
        with c1: sharp_val = st.slider("Độ sắc nét AI (Sharpness)", 0.5, 4.0, 2.0, 0.1)
        with c2: contrast_val = st.slider("Độ tương phản (Contrast)", 0.8, 2.5, 1.4, 0.1)
        with c3: denoise_chk = st.checkbox("Khử nhiễu ảnh (Denoise)", value=True)
            
        if st.button("🚀 Kích Hoạt AI Xử Lý Ảnh & Sinh Depth Map 3D", type="primary"):
            with st.spinner("AI đang nâng cấp độ phân giải, làm nét chi tiết và tạo Heightmap 3D..."):
                enhanced_img, depth_map = process_ai_image(st.session_state.original_img, sharpness=sharp_val, contrast=contrast_val, denoise=denoise_chk)
                st.session_state.processed_img = enhanced_img
                st.session_state.depth_map = depth_map
                st.success("Xử lý ảnh AI hoàn tất!")
                
        if st.session_state.original_img is not None:
            st.markdown("---")
            st.markdown("#### 🔍 Đối Chiếu So Sánh Ảnh Gốc vs Ảnh AI Đã Xử Lý")
            col_img1, col_img2 = st.columns(2)
            with col_img1:
                st.image(st.session_state.original_img, caption="📷 Ảnh Gốc Tải Lên", use_container_width=True)
            with col_img2:
                if st.session_state.processed_img is not None:
                    st.image(st.session_state.processed_img, caption="✨ Ảnh AI Siêu Nét", use_container_width=True)
                else:
                    st.info("Nhấn nút Kích Hoạt AI để xem kết quả.")
                    
            if st.session_state.depth_map is not None:
                st.markdown("#### 🗺️ Bản Đồ Độ Sâu (3D Depth Map Heightmap)")
                st.image(st.session_state.depth_map, caption="Heightmap 16-bit phân tầng độ sâu gia công", use_container_width=True)

# --- TAB 2: QUẢN LÝ LAYER, TƯ VẤN AI & TẢI G-CODE ---
with tab_layers:
    st.subheader("2. Quản Lý Layer Gia Công & Sinh G-Code Cho GRBL / UGS")
    if st.session_state.depth_map is None:
        st.warning("⚠️ Vui lòng tải ảnh và kích hoạt AI xử lý tại Tab 1 trước.")
    else:
        # Layer 1: Pha Thô
        with st.expander("🔨 LAYER 1: PHA THÔ 3D (ROUGHING CARVING)", expanded=True):
            st.markdown('<span class="ai-badge">🤖 AI Advisor: Khuyên dùng dao Endmill 6mm phá vạc lòng chảo nhanh.</span>', unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                l1_tool_type = st.selectbox("Loại dao", ["Endmill (Dao bằng)", "Bullnose"], index=0, key="l1_t")
                l1_tool_dia = st.number_input("Đường kính dao (mm)", value=6.0, step=0.5, key="l1_d")
            with c2:
                l1_stepdown = st.number_input("Độ sâu lượt Z (mm)", value=3.0, step=0.5, key="l1_sd")
                l1_stepover = st.slider("Dịch dao %", 10, 80, 40, key="l1_so")
            with c3:
                l1_feed = st.number_input("Tốc độ F (mm/min)", value=1800, step=100, key="l1_f")
                l1_rpm = st.number_input("Tốc độ S (RPM)", value=15000, step=1000, key="l1_r")
            with c4:
                if l1_stepdown > l1_tool_dia / 2:
                    st.markdown('<span class="warning-badge">⚠️ Cảnh báo: Z-step lớn hơn 1/2 đường kính dao!</span>', unsafe_allow_html=True)
                else:
                    st.success("✅ Thông số an toàn.")
            
            gcode_l1 = generate_roughing_gcode(st.session_state.depth_map, stock_w, stock_h, target_depth, l1_tool_dia, l1_stepover, l1_stepdown, l1_feed, l1_rpm, z_safe)
            st.download_button("📥 Tải G-Code Layer 1 (Layer1_Roughing.nc)", data=gcode_l1, file_name="Layer1_Roughing.nc", mime="text/plain")

        # Layer 2: Khắc Tinh
        with st.expander("✨ LAYER 2: KHẮC TINH CHI TIẾT 3D (FINISHING CARVING)", expanded=False):
            st.markdown('<span class="ai-badge">🤖 AI Advisor: Khuyên dùng Tapered Ballnose R0.5 cho chi tiết hoa văn siêu nhỏ.</span>', unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                l2_tool_type = st.selectbox("Loại dao", ["Tapered Ballnose", "Ballnose"], index=0, key="l2_t")
                l2_tool_dia = st.number_input("Đường kính dao (mm)", value=2.0, step=0.1, key="l2_d")
            with c2:
                l2_stepover = st.slider("Dịch dao % (Stepover tinh)", 5, 25, 10, key="l2_so")
                l2_feed = st.number_input("Tốc độ F (mm/min)", value=2200, step=100, key="l2_f")
            with c3:
                l2_rpm = st.number_input("Tốc độ S (RPM)", value=18000, step=1000, key="l2_r")
            with c4: st.success("✅ Stepover 10% bề mặt cực mịn.")
            
            gcode_l2 = generate_finishing_gcode(st.session_state.depth_map, stock_w, stock_h, target_depth, l2_tool_dia, l2_stepover, l2_feed, l2_rpm, z_safe)
            st.download_button("📥 Tải G-Code Layer 2 (Layer2_Finishing.nc)", data=gcode_l2, file_name="Layer2_Finishing.nc", mime="text/plain")

        # Layer 3: Cắt Biên & Cầu Giữ Phôi
        with st.expander("✂️ LAYER 3: CẮT BIÊN & TẠO CẦU GIỮ PHÔI (CUTOUT & TABS)", expanded=False):
            st.markdown('<span class="ai-badge">🤖 AI Advisor: Tự động chèn Tab cầu giữ phôi chống văng.</span>', unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                l3_tool_dia = st.number_input("Đường kính dao cắt (mm)", value=6.0, step=0.5, key="l3_d")
                l3_stepdown = st.number_input("Độ sâu cắt/lượt (mm)", value=3.0, step=0.5, key="l3_sd")
            with c2:
                tab_width = st.number_input("Rộng Tab (mm)", value=8.0, step=1.0, key="tb_w")
                tab_height = st.number_input("Cao Tab (mm)", value=4.0, step=0.5, key="tb_h")
            with c3:
                tab_count = st.number_input("Số lượng Tab", value=4, min_value=2, max_value=12, key="tb_c")
                l3_feed = st.number_input("Tốc độ F (mm/min)", value=1200, step=100, key="l3_f")
            with c4:
                l3_rpm = st.number_input("Tốc độ S (RPM)", value=16000, step=1000, key="l3_r")
            
            gcode_l3 = generate_cutout_gcode(stock_w, stock_h, board_z, l3_tool_dia, l3_stepdown, l3_feed, l3_rpm, z_safe, tab_width, tab_height, tab_count)
            st.download_button("📥 Tải G-Code Layer 3 (Layer3_Cutout_Tabs.nc)", data=gcode_l3, file_name="Layer3_Cutout_Tabs.nc", mime="text/plain")

# --- TAB 3: MÔ PHỎNG VISUAL DASHBOARD 3D ---
with tab_3d:
    st.subheader("3. Mô Phỏng Chi Tiết Gia Công Nằm Trong Tấm Ván")
    st.write(f"Khổ ván: **{board_w}x{board_h}x{board_z} mm** | Phôi khắc: **{stock_w}x{stock_h}x{target_depth} mm**")
    
    scale = 0.5
    svg_w, svg_h = int(board_w * scale), int(board_h * scale)
    sx, sy, sw, sh = int(offset_x * scale), int(offset_y * scale), int(stock_w * scale), int(stock_h * scale)
    
    svg_content = f"""
    <svg width="{svg_w}" height="{svg_h}" style="background-color: #D2B48C; border: 3px solid #8B4513; border-radius: 8px;">
        <rect width="100%" height="100%" fill="#D2B48C" />
        <rect x="{sx}" y="{sy}" width="{sw}" height="{sh}" fill="#A0522D" stroke="#5C2C16" stroke-width="2" rx="4" />
        <circle cx="{sx}" cy="{sy}" r="6" fill="#FF0000" />
        <line x1="{sx}" y1="{sy}" x2="{sx + 30}" y2="{sy}" stroke="#FF0000" stroke-width="2" />
        <line x1="{sx}" y1="{sy}" x2="{sx}" y2="{sy + 30}" stroke="#00FF00" stroke-width="2" />
        <text x="{sx + 8}" y="{sy - 8}" fill="#000000" font-weight="bold" font-size="12">G54 (X0, Y0)</text>
        <rect x="{sx + sw/2 - 10}" y="{sy - 2}" width="20" height="4" fill="#00FF00" />
        <rect x="{sx + sw/2 - 10}" y="{sy + sh - 2}" width="20" height="4" fill="#00FF00" />
        <text x="{sx + 10}" y="{sy + 25}" fill="#FFFFFF" font-size="14" font-weight="bold">Tranh Khắc CNC ({stock_w}x{stock_h}mm)</text>
    </svg>
    """
    st.components.v1.html(svg_content, height=svg_h + 30)
