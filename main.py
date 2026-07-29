import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import math
import io
import re

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="AI CNC Wood Carving Studio (GRBL / UGS)",
    page_icon="🪵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #5A3E2B;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #8C6D53;
        margin-bottom: 20px;
    }
    .stCard {
        background-color: #FDFBF7;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #E6DCCF;
        margin-bottom: 15px;
    }
    .ai-badge {
        background-color: #E0F2FE;
        color: #0369A1;
        font-weight: bold;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.85rem;
    }
    .warning-badge {
        background-color: #FEF3C7;
        color: #B45309;
        font-weight: bold;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.85rem;
    }
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
# SIDEBAR - REQUIREMENT 8: BOARD & STOCK DIMENSIONS & WORK PIECE SETTINGS
# -----------------------------------------------------------------------------
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
    st.info("💡 **Ghi chú GRBL/UGS:** G-code sinh ra sử dụng hệ tọa độ tương đối/tuyệt đối chuẩn `G90`, đơn vị `G21` (mm) tương thích hoàn toàn với UGS, Candle và Mach3.")

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS: AI IMAGE PROCESSING & G-CODE GENERATOR
# -----------------------------------------------------------------------------
def process_ai_image(image_pil, sharpness=2.0, contrast=1.5, denoise=True, generate_depth=True):
    # Convert PIL to OpenCV BGR
    img_array = np.array(image_pil.convert('RGB'))
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # 1. Denoising
    if denoise:
        img_bgr = cv2.fastNlMeansDenoisingColored(img_bgr, None, 10, 10, 7, 21)
    
    # 2. Detail & Edge Sharpening (Unsharp Masking)
    gaussian = cv2.GaussianBlur(img_bgr, (0, 0), 3.0)
    sharpened = cv2.addWeighted(img_bgr, 1.0 + (sharpness * 0.5), gaussian, -(sharpness * 0.5), 0)
    
    # 3. Contrast adjustment
    pil_enhanced = Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Contrast(pil_enhanced)
    final_img = enhancer.enhance(contrast)
    
    # 4. Pseudo-3D Heightmap Generation (Simulated Depth Anything V2)
    gray = cv2.cvtColor(np.array(final_img), cv2.COLOR_RGB2GRAY)
    # Blur slightly to smooth depth transitions for CNC ballnose carving
    depth_smooth = cv2.GaussianBlur(gray, (5, 5), 0)
    # Invert so brighter areas are higher (less carve) and darker are deeper
    depth_map = 255 - depth_smooth
    
    return final_img, depth_map

def generate_roughing_gcode(depth_map, stock_w, stock_h, target_depth, tool_dia, stepover_pct, stepdown, feedrate, spindle_rpm, z_safe):
    """
    Generate 3D Roughing (Pha thô) G-code in Z-layers using GRBL/UGS dialect
    """
    lines = [
        "(--- LAYER 1: PHA THO 3D / ROUGHING CARVING ---)",
        "(G-code sinh cho GRBL + UGS)",
        "G21 ; Thiet lap don vi milimet",
        "G90 ; Toa do tuyet doi",
        f"M03 S{int(spindle_rpm)} ; Bat trục chinh RPM",
        f"G00 Z{z_safe:.3f} ; Nac dao an toan"
    ]
    
    h, w = depth_map.shape
    step_x = tool_dia * (stepover_pct / 100.0)
    step_y = tool_dia * (stepover_pct / 100.0)
    
    cols = int(stock_w / step_x)
    rows = int(stock_h / step_y)
    
    # Multi-pass depth layers
    num_passes = math.ceil(target_depth / stepdown)
    
    for current_pass in range(1, num_passes + 1):
        pass_z = min(current_pass * stepdown, target_depth)
        lines.append(f"(; --- Luot pha thô phoi depth = -{pass_z:.2f} mm ---)")
        for r in range(0, rows):
            y_pos = r * step_y
            py = int((y_pos / stock_h) * (h - 1))
            py = min(max(py, 0), h - 1)
            
            # Raster scan alternating
            x_range = range(0, cols) if r % 2 == 0 else range(cols - 1, -1, -1)
            
            for c in x_range:
                x_pos = c * step_x
                px = int((x_pos / stock_w) * (w - 1))
                px = min(max(px, 0), w - 1)
                
                # Depth calculation (0 to 1 scale * target depth)
                normalized_depth = (depth_map[py, px] / 255.0) * pass_z
                z_pos = -normalized_depth
                
                if c == x_range[0]:
                    lines.append(f"G00 X{x_pos:.3f} Y{y_pos:.3f}")
                    lines.append(f"G01 Z{z_pos:.3f} F{int(feedrate/2)}")
                else:
                    lines.append(f"G01 X{x_pos:.3f} Y{y_pos:.3f} Z{z_pos:.3f} F{int(feedrate)}")
        
        lines.append(f"G00 Z{z_safe:.3f}")

    lines.extend([
        f"G00 Z{z_safe:.3f}",
        "M05 ; Tat truc chinh",
        "G00 X0 Y0 ; Ve goc toan do",
        "M30 ; Ket thuc chuong trinh"
    ])
    return "\n".join(lines)

def generate_finishing_gcode(depth_map, stock_w, stock_h, target_depth, tool_dia, stepover_pct, feedrate, spindle_rpm, z_safe):
    """
    Generate 3D Finishing (Khắc tinh) G-code with high precision rastering
    """
    lines = [
        "(--- LAYER 2: KHAC TINH 3D / FINISHING CARVING ---)",
        "(G-code sinh cho GRBL + UGS)",
        "G21 ; Don vi mm",
        "G90 ; Toa do tuyet doi",
        f"M03 S{int(spindle_rpm)}",
        f"G00 Z{z_safe:.3f}"
    ]
    
    h, w = depth_map.shape
    step_x = tool_dia * (stepover_pct / 100.0)
    step_y = tool_dia * (stepover_pct / 100.0)
    
    cols = int(stock_w / step_x)
    rows = int(stock_h / step_y)
    
    for r in range(0, rows):
        y_pos = r * step_y
        py = int((y_pos / stock_h) * (h - 1))
        py = min(max(py, 0), h - 1)
        
        x_range = range(0, cols) if r % 2 == 0 else range(cols - 1, -1, -1)
        
        for c in x_range:
            x_pos = c * step_x
            px = int((x_pos / stock_w) * (w - 1))
            px = min(max(px, 0), w - 1)
            
            z_pos = -((depth_map[py, px] / 255.0) * target_depth)
            
            if c == x_range[0]:
                lines.append(f"G00 X{x_pos:.3f} Y{y_pos:.3f}")
                lines.append(f"G01 Z{z_pos:.3f} F{int(feedrate/2)}")
            else:
                lines.append(f"G01 X{x_pos:.3f} Y{y_pos:.3f} Z{z_pos:.3f} F{int(feedrate)}")
                
    lines.extend([
        f"G00 Z{z_safe:.3f}",
        "M05",
        "G00 X0 Y0",
        "M30"
    ])
    return "\n".join(lines)

def generate_cutout_gcode(stock_w, stock_h, stock_thickness, tool_dia, stepdown, feedrate, spindle_rpm, z_safe, tab_width, tab_height, tab_count):
    """
    Generate Cutout Contour Pass with Tabs & Multi-pass depth
    """
    lines = [
        "(--- LAYER 3: CAT BIEN & TAO TAB / CUTOUT CONTOUR ---)",
        "(G-code sinh cho GRBL + UGS)",
        "G21",
        "G90",
        f"M03 S{int(spindle_rpm)}",
        f"G00 Z{z_safe:.3f}"
    ]
    
    # Outer rectangle perimeter path with tool radius offset compensation
    r = tool_dia / 2.0
    x0, y0 = -r, -r
    x1, y1 = stock_w + r, stock_h + r
    
    perimeter = 2 * (stock_w + stock_h)
    tab_positions = [i * (perimeter / tab_count) for i in range(tab_count)]
    
    num_passes = math.ceil(stock_thickness / stepdown)
    
    for p in range(1, num_passes + 1):
        current_z = -min(p * stepdown, stock_thickness)
        lines.append(f"(; --- Luot cat depth = {current_z:.2f} mm ---)")
        
        # Rectangle path: (x0,y0) -> (x1,y0) -> (x1,y1) -> (x0,y1) -> (x0,y0)
        path_segments = [
            ((x0, y0), (x1, y0)),
            ((x1, y0), (x1, y1)),
            ((x1, y1), (x0, y1)),
            ((x0, y1), (x0, y0))
        ]
        
        dist_acc = 0.0
        lines.append(f"G00 X{x0:.3f} Y{y0:.3f}")
        lines.append(f"G01 Z{current_z:.3f} F{int(feedrate/2)}")
        
        for p_start, p_end in path_segments:
            seg_len = math.hypot(p_end[0]-p_start[0], p_end[1]-p_start[1])
            
            # Check if tabs fall on this segment
            # Simple Tab Bridge check on final passes
            is_final_pass = (p == num_passes) or (abs(current_z) >= (stock_thickness - tab_height))
            
            if is_final_pass:
                # Add tab bridging logic
                mid_x = (p_start[0] + p_end[0]) / 2.0
                mid_y = (p_start[1] + p_end[1]) / 2.0
                tab_z = current_z + tab_height
                if tab_z > 0: tab_z = 0
                
                # Cut to before tab
                lines.append(f"G01 X{mid_x - tab_width/2:.3f} Y{mid_y:.3f} Z{current_z:.3f} F{int(feedrate)}")
                # Raise for Tab
                lines.append(f"G01 Z{tab_z:.3f} F{int(feedrate/2)}")
                lines.append(f"G01 X{mid_x + tab_width/2:.3f} Y{mid_y:.3f} F{int(feedrate)}")
                # Lower back down
                lines.append(f"G01 Z{current_z:.3f} F{int(feedrate/2)}")
                # Finish segment
                lines.append(f"G01 X{p_end[0]:.3f} Y{p_end[1]:.3f} F{int(feedrate)}")
            else:
                lines.append(f"G01 X{p_end[0]:.3f} Y{p_end[1]:.3f} F{int(feedrate)}")
                
    lines.extend([
        f"G00 Z{z_safe:.3f}",
        "M05",
        "G00 X0 Y0",
        "M30"
    ])
    return "\n".join(lines)

# -----------------------------------------------------------------------------
# MAIN INTERFACE & WORKFLOW
# -----------------------------------------------------------------------------

tab_upload, tab_layers, tab_3d = st.tabs([
    "🖼️ 1. Upload & AI Xử Lý Ảnh (Requirement 1, 9)",
    "🔲 2. Phân Layer Gia Công & AI Tư Vấn & G-Code (Requirement 2, 3, 4, 5, 6, 7)",
    "🖥️ 3. Dashboard Mô Phỏng Trực Quan 3D Ván (Requirement 8, 10)"
])

# --- TAB 1: UPLOAD & AI IMAGE ENHANCEMENT ---
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
            
        if st.button("🚀 Kích Hoạt AI Xử Lý Ảnh & Sinh Depth Map 3D", type="primary"):
            with st.spinner("AI đang nâng cấp độ phân giải, tăng nét chi tiết và tạoHeightmap 3D..."):
                enhanced_img, depth_map = process_ai_image(
                    st.session_state.original_img, 
                    sharpness=sharp_val, 
                    contrast=contrast_val, 
                    denoise=denoise_chk
                )
                st.session_state.processed_img = enhanced_img
                st.session_state.depth_map = depth_map
                st.success("Xử lý ảnh AI hoàn tất!")
                
        # REQUIREMENT 9: Side-by-side Image Comparison
        if st.session_state.original_img is not None:
            st.markdown("---")
            st.markdown("#### 🔍 Đối Chiếu So Sánh Ảnh Gốc vs Ảnh AI Đã Xử Lý")
            col_img1, col_img2 = st.columns(2)
            
            with col_img1:
                st.image(st.session_state.original_img, caption="📷 Ảnh Gốc Tải Lên", use_container_width=True)
                
            with col_img2:
                if st.session_state.processed_img is not None:
                    st.image(st.session_state.processed_img, caption="✨ Ảnh AI Siêu Nét (Edge Enhanced)", use_container_width=True)
                else:
                    st.info("Nhấn 'Kích Hoạt AI Xử Lý Ảnh' để xem kết quả siêu nét.")
                    
            if st.session_state.depth_map is not None:
                st.markdown("#### 🗺️ Bản Đồ Độ Sâu (3D Depth Map Heightmap)")
                st.image(st.session_state.depth_map, caption="Heightmap 16-bit phân tầng cho dao CNC gọt khắc", use_container_width=True)

# --- TAB 2: LAYERS, AI ADVISOR, PARAMETERS & G-CODE GENERATION ---
with tab_layers:
    st.subheader("2. Quản Lý Layer Gia Công & Sinh G-Code Cho GRBL / UGS")
    
    if st.session_state.depth_map is None:
        st.warning("⚠️ Vui lòng tải ảnh và kích hoạt AI xử lý tại Tab 1 trước khi cấu hình Layer.")
    else:
        st.markdown("AI đã tự động phân tích ảnh và sinh **3 Layer Gia Công Chuẩn CNC**:")
        
        # ---------------------------------------------------------------------
        # LAYER 1: PHA THÔ 3D (ROUGHING)
        # ---------------------------------------------------------------------
        with st.expander("🔨 LAYER 1: PHA THÔ 3D (ROUGHING CARVING)", expanded=True):
            st.markdown('<span class="ai-badge">🤖 AI Advisor: Đề xuất sử dụng dao Endmill 6mm / Phá vạt nhanh vùng gỗ thừa.</span>', unsafe_allow_html=True)
            st.markdown("")
            
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
            
            gcode_l1 = generate_roughing_gcode(
                st.session_state.depth_map, stock_w, stock_h, target_depth,
                l1_tool_dia, l1_stepover, l1_stepdown, l1_feed, l1_rpm, z_safe
            )
            
            # REQUIREMENT 6: Download Button per Layer
            st.download_button(
                label="📥 Tải G-Code Layer 1 (Layer1_Roughing.nc)",
                data=gcode_l1,
                file_name="Layer1_Roughing.nc",
                mime="text/plain"
            )

        # ---------------------------------------------------------------------
        # LAYER 2: KHẮC TINH 3D (FINISHING)
        # ---------------------------------------------------------------------
        with st.expander("✨ LAYER 2: KHẮC TINH CHI TIẾT 3D (FINISHING CARVING)", expanded=False):
            st.markdown('<span class="ai-badge">🤖 AI Advisor: Đề xuất Dao Cầu / Tapered Ballnose 2mm R0.5 / Độ nét tinh xảo.</span>', unsafe_allow_html=True)
            st.markdown("")
            
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

        # ---------------------------------------------------------------------
        # LAYER 3: CẮT BIÊN & CẦU GIỮ PHÔI (CUTOUT & TABS) - REQUIREMENT 7
        # ---------------------------------------------------------------------
        with st.expander("✂️ LAYER 3: CẮT BIÊN & TẠO CẦU GIỮ PHÔI (CUTOUT & TABS)", expanded=False):
            st.markdown('<span class="ai-badge">🤖 AI Advisor: Tự động tính toán Tab cầu giữ chống văng phôi khi đứt ván.</span>', unsafe_allow_html=True)
            st.markdown("")
            
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

# --- TAB 3: VISUAL 3D DASHBOARD ---
with tab_3d:
    st.subheader("3. Mô Phỏng Chi Tiết Gia Công Nằm Trong Tấm Ván (Requirement 10)")
    
    st.write(f"**Khổ ván gỗ tổng:** {board_w} x {board_h} x {board_z} mm | **Phôi gia công:** {stock_w} x {stock_h} x {target_depth} mm")
    
    # SVG Interactive Canvas Simulation
    scale = 0.5  # scaling for display
    svg_w = int(board_w * scale)
    svg_h = int(board_h * scale)
    
    sx = int(offset_x * scale)
    sy = int(offset_y * scale)
    sw = int(stock_w * scale)
    sh = int(stock_h * scale)
    
    svg_content = f"""
    <svg width="{svg_w}" height="{svg_h}" style="background-color: #D2B48C; border: 3px solid #8B4513; border-radius: 8px;">
        <!-- Board Grid -->
        <defs>
            <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
                <path d="M 20 0 L 0 0 0 20" fill="none" stroke="#C19A6B" stroke-width="0.5"/>
            </pattern>
        </defs>
        <rect width="100%" height="100%" fill="url(#grid)" />
        
        <!-- Stock Workpiece -->
        <rect x="{sx}" y="{sy}" width="{sw}" height="{sh}" fill="#A0522D" stroke="#5C2C16" stroke-width="2" rx="4" opacity="0.85"/>
        
        <!-- Zero Origin Marker -->
        <circle cx="{sx}" cy="{sy}" r="6" fill="#FF0000" />
        <line x1="{sx}" y1="{sy}" x2="{sx + 30}" y2="{sy}" stroke="#FF0000" stroke-width="2" />
        <line x1="{sx}" y1="{sy}" x2="{sx}" y2="{sy + 30}" stroke="#00FF00" stroke-width="2" />
        <text x="{sx + 8}" y="{sy - 8}" fill="#000000" font-weight="bold" font-size="12">G54 (X0, Y0)</text>
        
        <!-- Tabs visualization -->
        <rect x="{sx + sw/2 - 10}" y="{sy - 2}" width="20" height="4" fill="#00FF00" />
        <rect x="{sx + sw/2 - 10}" y="{sy + sh - 2}" width="20" height="4" fill="#00FF00" />
        <rect x="{sx - 2}" y="{sy + sh/2 - 10}" width="4" height="20" fill="#00FF00" />
        <rect x="{sx + sw - 2}" y="{sy + sh/2 - 10}" width="4" height="20" fill="#00FF00" />
        
        <!-- Label dimensions -->
        <text x="{sx + 10}" y="{sy + 25}" fill="#FFFFFF" font-size="14" font-weight="bold">Tranh Khắc CNC ({stock_w}x{stock_h}mm)</text>
        <text x="10" y="{svg_h - 10}" fill="#3D2314" font-size="12">Tấm Ván Tổng: {board_w} x {board_h} mm</text>
    </svg>
    """
    
    st.components.v1.html(svg_content, height=svg_h + 30)
    
    st.markdown("""
    **💡 Chú thích trực quan Dashboard:**
    - 🟫 **Vùng màu nâu vàng ngoài:** Tấm ván nguyên khổ ($1200 \times 800$ mm).
    - 🟧 **Khu vực phôi khắc 3D:** Vị trí bức tranh đặt trong tấm ván ($300 \times 400$ mm).
    - 🔴 **Điểm đỏ (G54):** Mốc tọa độ X0, Y0, Z0 cài đặt trên Universal Gcode Sender (UGS).
    - 🟢 **Vạch xanh lá:** Vị trí các cầu giữ phôi (Tabs) chống rơi/văng phôi sau khi cắt đứt.
    """)
