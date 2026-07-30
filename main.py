import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import zipfile

# ==============================================================================
# CẤU HÌNH TRANG WEB
# ==============================================================================
st.set_page_config(
    page_title="AI Wood CAM GRBL/UGS Professional",
    page_icon="🪵",
    layout="wide"
)

st.title("🪵 Hệ Thống AI CAM Tranh Gỗ 3D Phân Lớp (Chuẩn GRBL / UGS)")
st.caption("Cập nhật: AI Tư vấn An Toàn Cắt Biên (Z Stepdown) & Chia Nhiều Lớp Ăn Dao Tránh Gãy Dao / Văng Phôi")
st.markdown("---")

# ==============================================================================
# SIDEBAR: CẤU HÌNH PHÔI & WORK ZERO
# ==============================================================================
with st.sidebar:
    st.header("📂 1. Kích Thước Phôi & Work Zero")
    stock_x = st.number_input("Chiều dài phôi X (mm)", value=300.0, step=10.0)
    stock_y = st.number_input("Chiều rộng phôi Y (mm)", value=200.0, step=10.0)
    stock_z = st.number_input("Chiều dày phôi Z (mm)", value=30.0, step=5.0)
    relief_depth = st.number_input("Độ sâu tranh 3D Z (mm)", value=15.0, step=1.0)
    safe_z = st.number_input("Chiều cao an toàn Safe Z (mm)", value=10.0, step=1.0)
    
    st.subheader("🎯 Đặt Gốc Phôi (Work Zero X0 Y0 Z0)")
    work_zero = st.selectbox(
        "Vị trí lấy gốc dao:",
        [
            "Góc dưới bên trái (Bottom-Left - Chuẩn UGS)",
            "Tâm phôi (Center)",
            "Góc trên bên trái (Top-Left)",
            "Góc trên bên phải (Top-Right)",
            "Góc dưới bên phải (Bottom-Right)"
        ]
    )

    st.markdown("---")
    st.header("🪵 2. Chọn Loại Gỗ Gia Công")
    wood_type = st.selectbox(
        "Vật liệu gỗ phôi:",
        [
            "Gỗ Gụ / Hương / Mộc (Cứng vừa)",
            "Gỗ Trắc / Cẩm / Cừu (Rất cứng)",
            "Gỗ Thông / Cao Su (Mềm)"
        ]
    )

    st.markdown("---")
    st.info("🤖 **Bộ điều khiển:** Chuẩn **GRBL / UGS (Universal Gcode Sender)**")

# ==============================================================================
# BỘ TƯ VẤN THÔNG SỐ GIA CÔNG AN TOÀN (AI CAM ENGINE)
# ==============================================================================
def calculate_ai_parameters(wood, depth, width, height, thickness):
    """Tính toán thông số an toàn chống gãy dao và văng phôi"""
    if "Trắc" in wood:
        f_rough, f_finish, f_pencil, f_cutout = 1200, 1800, 1000, 800
        rpm_rough, rpm_finish, rpm_pencil, rpm_cutout = 18000, 22000, 20000, 16000
        stepdown_r = 2.0
        stepdown_cut = 1.5  # Gỗ rất cứng -> Ăn rất nông 1.5mm/lượt
    elif "Gụ" in wood:
        f_rough, f_finish, f_pencil, f_cutout = 1800, 2400, 1200, 1000
        rpm_rough, rpm_finish, rpm_pencil, rpm_cutout = 16000, 20000, 18000, 15000
        stepdown_r = 3.0
        stepdown_cut = 2.5  # Gỗ cứng vừa -> Ăn 2.5mm/lượt
    else:  # Gỗ mềm
        f_rough, f_finish, f_pencil, f_cutout = 2500, 3000, 1500, 1200
        rpm_rough, rpm_finish, rpm_pencil, rpm_cutout = 14000, 18000, 16000, 14000
        stepdown_r = 4.0
        stepdown_cut = 3.5  # Gỗ mềm -> Ăn 3.5mm/lượt

    d_rough = 6.0 if width >= 200 else 4.0
    d_finish = 3.0 if width >= 200 else 2.0
    d_pencil = 1.0
    d_cutout = 6.0  # Dao End Mill Ø6mm cắt biên

    # Tính số lượt cắt biên bắt buộc
    total_cut_passes = int(np.ceil(thickness / stepdown_cut))

    return {
        "l1": {"tool": f"End Mill Ø{d_rough}mm", "dia": d_rough, "f": f_rough, "s": rpm_rough, "stepdown": stepdown_r},
        "l2": {"tool": f"Ball Nose Ø{d_finish}mm", "dia": d_finish, "f": f_finish, "s": rpm_finish, "stepover": 0.12},
        "l3": {"tool": f"V-Bit / Mũi Tỉa Ø{d_pencil}mm", "dia": d_pencil, "f": f_pencil, "s": rpm_pencil},
        "l4": {
            "tool": f"End Mill Cắt Biên Ø{d_cutout}mm",
            "dia": d_cutout,
            "f": f_cutout,
            "s": rpm_cutout,
            "stepdown": stepdown_cut,
            "passes": total_cut_passes
        }
    }

# ==============================================================================
# NÚT BẤM VÀ BẢNG TƯ VẤN AN TOÀN
# ==============================================================================
st.header("🤖 Trợ Lý AI: Tính Toán & Tư Vấn Cắt An Toàn")

col_btn, col_space = st.columns([1, 3])
with col_btn:
    calc_pressed = st.button("🧮 TÍNH TOÁN THÔNG SỐ AI", use_container_width=True, type="primary")

if calc_pressed or "ai_rec" not in st.session_state:
    st.session_state["ai_rec"] = calculate_ai_parameters(wood_type, relief_depth, stock_x, stock_y, stock_z)
    if calc_pressed:
        st.toast("✅ Đã cập nhật tính toán Z-Stepdown An Toàn!", icon="🛡️")

ai_rec = st.session_state["ai_rec"]

# BẢNG TƯ VẤN THÔNG SỐ CÓ CẮT BIÊN AN TOÀN
col_ai1, col_ai2, col_ai3, col_ai4 = st.columns(4)

with col_ai1:
    st.success("🔴 **Layer 1: Phá Thô**")
    st.write(f"- Dao gợi ý: **{ai_rec['l1']['tool']}**")
    st.write(f"- Tốc độ ăn (F): **{ai_rec['l1']['f']} mm/min**")
    st.write(f"- Spindle (S): **{ai_rec['l1']['s']} RPM**")
    st.write(f"- Lát ăn Z: **{ai_rec['l1']['stepdown']} mm/lượt**")

with col_ai2:
    st.info("🟢 **Layer 2: Chạy Tinh**")
    st.write(f"- Dao gợi ý: **{ai_rec['l2']['tool']}**")
    st.write(f"- Tốc độ ăn (F): **{ai_rec['l2']['f']} mm/min**")
    st.write(f"- Spindle (S): **{ai_rec['l2']['s']} RPM**")
    st.write(f"- Dịch dao Stepover: **12% bán kính**")

with col_ai3:
    st.warning("🔵 **Layer 3: Điêu Khắc Nét**")
    st.write(f"- Dao gợi ý: **{ai_rec['l3']['tool']}**")
    st.write(f"- Tốc độ ăn (F): **{ai_rec['l3']['f']} mm/min**")
    st.write(f"- Spindle (S): **{ai_rec['l3']['s']} RPM**")
    st.write(f"- Lọc nét hoa văn sâu")

with col_ai4:
    st.error("🟠 **Layer 4: Cắt Biên Tranh (AN TOÀN)**")
    st.write(f"- Dao cắt: **{ai_rec['l4']['tool']}**")
    st.write(f"- Tốc độ ăn (F): **{ai_rec['l4']['f']} mm/min**")
    st.write(f"- Spindle (S): **{ai_rec['l4']['s']} RPM**")
    # TỰ VẤN RÕ RÀNG MỖI LƯỢT ĂN Z
    st.markdown(f"👉 **Ăn sâu Z mỗi lượt:** `<span style='color:red; font-weight:bold;'>{ai_rec['l4']['stepdown']} mm/lượt</span>`", unsafe_allow_html=True)
    st.markdown(f"👉 **Tổng số lượt cắt:** `<span style='color:red; font-weight:bold;'>{ai_rec['l4']['passes']} lượt (Passes)</span>`", unsafe_allow_html=True)

st.markdown("---")

# ==============================================================================
# XỬ LÝ HÌNH ẢNH & XUẤT G-CODE CHO UGS
# ==============================================================================
uploaded_file = st.file_uploader("Tải lên hình ảnh mẫu tranh gỗ", type=["png", "jpg", "jpeg", "webp"])

if uploaded_file:
    raw_img = Image.open(uploaded_file).convert("RGB")
    if max(raw_img.size) > 1000:
        raw_img.thumbnail((1000, 1000))
    
    gray_img = cv2.cvtColor(np.array(raw_img), cv2.COLOR_RGB2GRAY)
    
    sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=5)
    edge_gradient = np.sqrt(sobelx**2 + sobely**2)
    
    base_depth = (255 - gray_img).astype(np.float64)
    max_grad = edge_gradient.max() if edge_gradient.max() > 0 else 1e-5
    ai_depth_raw = base_depth * 0.7 + (edge_gradient / max_grad * 255) * 0.3
    ai_depth_raw = cv2.GaussianBlur(ai_depth_raw, (7, 7), 0)
    
    depth_map_mm = (ai_depth_raw / ai_depth_raw.max()) * relief_depth
    img_h, img_w = depth_map_mm.shape
    scale_x = stock_x / img_w
    scale_y = stock_y / img_h

    # LỆCH TỌA ĐỘ WORK ZERO
    if "Center" in work_zero:
        offset_x, offset_y = -stock_x / 2.0, -stock_y / 2.0
    elif "Top-Left" in work_zero:
        offset_x, offset_y = 0.0, -stock_y
    elif "Top-Right" in work_zero:
        offset_x, offset_y = -stock_x, -stock_y
    elif "Bottom-Right" in work_zero:
        offset_x, offset_y = -stock_x, 0.0
    else:  # Bottom-Left
        offset_x, offset_y = 0.0, 0.0

    def make_grbl_header(layer_name, tool_desc, rpm):
        return [
            f"(--- {layer_name.upper()} ---)",
            f"(TOOL: {tool_desc})",
            f"(WORK ZERO: {work_zero})",
            "G21 ; Unit mm",
            "G90 ; Absolute Coordinates",
            "G54 ; Work Coordinate System",
            f"G0 Z{safe_z:.3f}",
            f"M3 S{int(rpm)} ; Start Spindle",
            "G4 P2 ; Wait 2 sec for spindle"
        ]

    # 1. LAYER PHÁ THÔ
    def generate_l1_roughing():
        t = ai_rec['l1']
        gcode = make_grbl_header("Layer 1 - Pha Tho", t['tool'], t['s'])
        step_px = max(1, int(t['dia'] / scale_x))
        max_z = np.max(depth_map_mm)
        passes = int(np.ceil(max_z / t['stepdown']))
        
        for p in range(1, passes + 1):
            cur_z = min(p * t['stepdown'], max_z)
            for y in range(0, img_h, step_px):
                x_range = range(0, img_w, step_px) if (y // step_px) % 2 == 0 else range(img_w - 1, -1, -step_px)
                for x in x_range:
                    target_z = depth_map_mm[y, x]
                    if target_z > 0.5:
                        cut_z = -min(cur_z, target_z)
                        real_x = x * scale_x + offset_x
                        real_y = (img_h - y) * scale_y + offset_y
                        gcode.append(f"G1 X{real_x:.3f} Y{real_y:.3f} Z{cut_z:.3f} F{t['f']}")
            gcode.append(f"G0 Z{safe_z:.3f}")
        gcode.extend(["M5", "M30"])
        return "\n".join(gcode)

    # 2. LAYER CHẠY TINH
    def generate_l2_finishing():
        t = ai_rec['l2']
        gcode = make_grbl_header("Layer 2 - Chay Tinh Mnin", t['tool'], t['s'])
        step_px = max(1, int((t['dia'] * t['stepover']) / scale_x))
        
        r_px = int(np.ceil((t['dia'] / 2.0) / scale_x))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r_px * 2 + 1, r_px * 2 + 1))
        compensated_map = cv2.erode(depth_map_mm.astype(np.float32), k)
        
        for y in range(0, img_h, step_px):
            x_range = range(0, img_w, step_px) if (y // step_px) % 2 == 0 else range(img_w - 1, -1, -step_px)
            for x in x_range:
                cut_z = -float(compensated_map[y, x])
                real_x = x * scale_x + offset_x
                real_y = (img_h - y) * scale_y + offset_y
                gcode.append(f"G1 X{real_x:.3f} Y{real_y:.3f} Z{cut_z:.3f} F{t['f']}")
        gcode.extend([f"G0 Z{safe_z:.3f}", "M5", "M30"])
        return "\n".join(gcode)

    # 3. LAYER ĐIÊU KHẮC CHI TIẾT
    def generate_l3_pencil():
        t = ai_rec['l3']
        gcode = make_grbl_header("Layer 3 - Dieu Khac Chi Tiet", t['tool'], t['s'])
        
        grad_thresh = max_grad * 0.35
        mask_detail = (edge_gradient > grad_thresh).astype(np.uint8)
        
        in_cut = False
        for y in range(0, img_h, 2):
            for x in range(0, img_w, 2):
                if mask_detail[y, x] > 0:
                    rx = x * scale_x + offset_x
                    ry = (img_h - y) * scale_y + offset_y
                    rz = -float(depth_map_mm[y, x])
                    if not in_cut:
                        gcode.append(f"G0 X{rx:.3f} Y{ry:.3f}")
                        gcode.append(f"G1 Z{rz:.3f} F{t['f']/2}")
                        in_cut = True
                    else:
                        gcode.append(f"G1 X{rx:.3f} Y{ry:.3f} Z{rz:.3f} F{t['f']}")
                else:
                    if in_cut:
                        gcode.append(f"G0 Z{safe_z:.3f}")
                        in_cut = False
        gcode.extend([f"G0 Z{safe_z:.3f}", "M5", "M30"])
        return "\n".join(gcode)

    # 4. LAYER CẮT BIÊN AN TOÀN (MULTI-PASS STEPDOWN + RAMPING)
    def generate_l4_cutout():
        t = ai_rec['l4']
        gcode = make_grbl_header("Layer 4 - Cat Bien An Toan (Multi-Pass Stepdown)", t['tool'], t['s'])
        
        r_cut = t['dia'] / 2.0
        x_min, x_max = 0.0 - r_cut + offset_x, stock_x + r_cut + offset_x
        y_min, y_max = 0.0 - r_cut + offset_y, stock_y + r_cut + offset_y
        
        step_z = t['stepdown']
        passes = t['passes']
        
        gcode.append(f"(THONG SO CẮT: Dày phôi {stock_z}mm | Ăn Z mỗi lượt: {step_z}mm | Tong cong: {passes} luot cut)")
        gcode.append(f"G0 X{x_min:.3f} Y{y_min:.3f}")
        
        # VÒNG LẶP CẮT TỪNG LƯỢT NÔNG CHO ĐẾN KHỦNG ĐỨT PHÔI
        for p in range(1, passes + 1):
            target_cut_z = -min(p * step_z, stock_z)
            gcode.append(f"(--- LUOT CUT AN TOAN SO {p}/{passes}: Z = {target_cut_z:.2f}mm ---)")
            
            # Đâm dao xuống Z từ từ (Plunge Rate = F / 2)
            gcode.append(f"G1 Z{target_cut_z:.3f} F{t['f']/2}")
            
            # Chạy đường viền 4 cạnh
            gcode.append(f"G1 X{x_max:.3f} Y{y_min:.3f} F{t['f']}")
            gcode.append(f"G1 X{x_max:.3f} Y{y_max:.3f} F{t['f']}")
            gcode.append(f"G1 X{x_min:.3f} Y{y_max:.3f} F{t['f']}")
            gcode.append(f"G1 X{x_min:.3f} Y{y_min:.3f} F{t['f']}")
            
        gcode.extend([f"G0 Z{safe_z:.3f}", "M5", "M30"])
        return "\n".join(gcode)

    # ==============================================================================
    # XUẤT FILE VÀ TẢI VỀ
    # ==============================================================================
    st.markdown("---")
    st.header("💾 Tải Về File G-Code Tương Ứng Với Thông Số Đã Tính")

    gc1 = generate_l1_roughing()
    gc2 = generate_l2_finishing()
    gc3 = generate_l3_pencil()
    gc4 = generate_l4_cutout()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.download_button("💾 1. Phá Thô (.nc)", data=gc1, file_name="01_PhaTho_GRBL.nc")
    with c2:
        st.download_button("💾 2. Chạy Tinh (.nc)", data=gc2, file_name="02_ChayTinh_GRBL.nc")
    with c3:
        st.download_button("💾 3. Điêu Khắc (.nc)", data=gc3, file_name="03_DieuKhac_GRBL.nc")
    with c4:
        st.download_button("💾 4. Cắt Biên An Toàn (.nc)", data=gc4, file_name="04_CatBien_AnToan_GRBL.nc")

    # Đóng gói ZIP
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        zf.writestr("01_PhaTho_EndMill.nc", gc1)
        zf.writestr("02_ChayTinh_BallNose.nc", gc2)
        zf.writestr("03_DieuKhac_Pencil.nc", gc3)
        zf.writestr("04_CatBien_ProfileCut_AnToan.nc", gc4)

    st.markdown("---")
    st.download_button(
        label="📦 TẢI TRỌN BỘ ZIP 4 LAYER G-CODE AN TOÀN (MỞ TRỰC TIẾP TRÊN UGS)",
        data=zip_buf.getvalue(),
        file_name="Tron_Bo_GCode_GRBL_UGS_AnToan.zip",
        mime="application/zip",
        use_container_width=True
    )
