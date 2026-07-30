import Streamlit as st
Import numpy as np
Import cv2
From PIL import Image
Import io
Import zipfile

# ==============================================================================
# CẤU HÌNH TRANG WEB
# ==============================================================================
St.set_page_config(
    Page_title="AI Wood CAM GRBL/UGS Professional",
    Page_icon="🪵",
    Layout="wide"
)

St.title("🪵 Hệ Thống AI CAM Tranh Gỗ 3D Phân Lớp (Chuẩn GRBL / UGS)")
St.caption("Sửa lỗi KeyError, Tự động Khởi tạo Session State An toàn & Cắt Biên Đa Tầng")
St.markdown("---")

# ==============================================================================
# SIDEBAR: CẤU HÌNH PHÔI & WORK ZERO
# ==============================================================================
With st.sidebar:
    St.header("📂 1. Kích Thước Phôi & Work Zero")
    Stock_x = st.number_input("Chiều dài phôi X (mm)", value=300.0, step=10.0)
    Stock_y = st.number_input("Chiều rộng phôi Y (mm)", value=200.0, step=10.0)
    Stock_z = st.number_input("Chiều dày phôi Z (mm)", value=30.0, step=5.0)
    Relief_depth = st.number_input("Độ sâu tranh 3D Z (mm)", value=15.0, step=1.0)
    Safe_z = st.number_input("Chiều cao an toàn Safe Z (mm)", value=10.0, step=1.0)
    
    St.subheader("🎯 Đặt Gốc Phôi (Work Zero X0 Y0 Z0)")
    Work_zero = st.selectbox(
        "Vị trí lấy gốc dao:",
        [
            "Góc dưới bên trái (Bottom-Left - Chuẩn UGS)",
            "Tâm phôi (Center)",
            "Góc trên bên trái (Top-Left)",
            "Góc trên bên phải (Top-Right)",
            "Góc dưới bên phải (Bottom-Right)"
        ]
    )

    St.markdown("---")
    St.header("🪵 2. Chọn Loại Gỗ Gia Công")
    Wood_type = st.selectbox(
        "Vật liệu gỗ phôi:",
        [
            "Gỗ Gụ / Hương / Mộc (Cứng vừa)",
            "Gỗ Trắc / Cẩm / Cừu (Rất cứng)",
            "Gỗ Thông / Cao Su (Mềm)"
        ]
    )

    St.markdown("---")
    St.info("🤖 **Bộ điều khiển:** Chuẩn **GRBL / UGS (Universal Gcode Sender)**")

# ==============================================================================
# BỘ TƯ VẤN THÔNG SỐ GIA CÔNG AN TOÀN (AI CAM ENGINE)
# ==============================================================================
Def calculate_ai_parameters(wood, depth, width, height, thickness):
    """Tính toán thông số an toàn chống gãy dao và văng phôi"""
    If "Trắc" in wood:
        F_rough, f_finish, f_pencil, f_cutout = 1200, 1800, 1000, 800
        Rpm_rough, rpm_finish, rpm_pencil, rpm_cutout = 18000, 22000, 20000, 16000
        Stepdown_r = 2.0
        Stepdown_cut = 1.5  # Gỗ rất cứng -> Ăn 1.5mm/lượt
    Elif "Gụ" in wood:
        F_rough, f_finish, f_pencil, f_cutout = 1800, 2400, 1200, 1000
        Rpm_rough, rpm_finish, rpm_pencil, rpm_cutout = 16000, 20000, 18000, 15000
        Stepdown_r = 3.0
        Stepdown_cut = 2.5  # Gỗ cứng vừa -> Ăn 2.5mm/lượt
    Else:  # Gỗ mềm
        F_rough, f_finish, f_pencil, f_cutout = 2500, 3000, 1500, 1200
        Rpm_rough, rpm_finish, rpm_pencil, rpm_cutout = 14000, 18000, 16000, 14000
        Stepdown_r = 4.0
        Stepdown_cut = 3.5  # Gỗ mềm -> Ăn 3.5mm/lượt

    D_rough = 6.0 if width >= 200 else 4.0
    D_finish = 3.0 if width >= 200 else 2.0
    D_pencil = 1.0
    D_cutout = 6.0

    # Tính toán tổng số lượt cắt biên
    Total_cut_passes = int(np.ceil(thickness / stepdown_cut))

    Return {
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
# QUẢN LÝ SESSION STATE AN TOÀN - CHỐNG LỖI KEYERROR
# ==============================================================================
St.header("🤖 Trợ Lý AI: Tính Toán & Tư Vấn Cắt An Toàn")

Col_btn, col_space = st.columns([1, 3])
With col_btn:
    Calc_pressed = st.button("🧮 TÍNH TOÁN THÔNG SỐ AI", use_container_width=True, type="primary")

# KIỂM TRA CHẶT CHẼ: Nếu chưa có session hoặc thiếu key 'passes' thì tính lại ngay
Needs_recalc = (
    Calc_pressed 
    Or "ai_rec" not in st.session_state 
    Or "passes" not in st.session_state.get("ai_rec", {}).get("l4", {})
)

If needs_recalc:
    St.session_state["ai_rec"] = calculate_ai_parameters(wood_type, relief_depth, stock_x, stock_y, stock_z)
    If calc_pressed:
        St.toast("✅ Đã tính toán xong thông số cắt biên an toàn!", icon="🛡️")

Ai_rec = st.session_state["ai_rec"]

# Lấy giá trị an toàn bằng .get() để phòng ngừa hoàn toàn KeyError
Passes_val = ai_rec['l4'].get('passes', int(np.ceil(stock_z / ai_rec['l4']['stepdown'])))
Stepdown_val = ai_rec['l4'].get('stepdown', 2.0)

# ==============================================================================
# HIỂN THỊ BẢNG THÔNG SỐ
# ==============================================================================
Col_ai1, col_ai2, col_ai3, col_ai4 = st.columns(4)

With col_ai1:
    St.success("🔴 **Layer 1: Phá Thô**")
    St.write(f"- Dao gợi ý: **{ai_rec['l1']['tool']}**")
    St.write(f"- Tốc độ ăn (F): **{ai_rec['l1']['f']} mm/min**")
    St.write(f"- Spindle (S): **{ai_rec['l1']['s']} RPM**")
    St.write(f"- Lát ăn Z: **{ai_rec['l1']['stepdown']} mm/lượt**")

With col_ai2:
    St.info("🟢 **Layer 2: Chạy Tinh**")
    St.write(f"- Dao gợi ý: **{ai_rec['l2']['tool']}**")
    St.write(f"- Tốc độ ăn (F): **{ai_rec['l2']['f']} mm/min**")
    St.write(f"- Spindle (S): **{ai_rec['l2']['s']} RPM**")
    St.write(f"- Dịch dao Stepover: **12% bán kính**")

With col_ai3:
    St.warning("🔵 **Layer 3: Điêu Khắc Nét**")
    St.write(f"- Dao gợi ý: **{ai_rec['l3']['tool']}**")
    St.write(f"- Tốc độ ăn (F): **{ai_rec['l3']['f']} mm/min**")
    St.write(f"- Spindle (S): **{ai_rec['l3']['s']} RPM**")
    St.write(f"- Lọc nét hoa văn sâu")

With col_ai4:
    St.error("🟠 **Layer 4: Cắt Biên Tranh (AN TOÀN)**")
    St.write(f"- Dao cắt: **{ai_rec['l4']['tool']}**")
    St.write(f"- Tốc độ ăn (F): **{ai_rec['l4']['f']} mm/min**")
    St.write(f"- Spindle (S): **{ai_rec['l4']['s']} RPM**")
    St.markdown(f"👉 **Ăn sâu Z mỗi lượt:** `<span style='color:red; font-weight:bold;'>{stepdown_val} mm/lượt</span>`", unsafe_allow_html=True)
    St.markdown(f"👉 **Tổng số lượt cắt:** `<span style='color:red; font-weight:bold;'>{passes_val} lượt (Passes)</span>`", unsafe_allow_html=True)

St.markdown("---")

# ==============================================================================
# XỬ LÝ HÌNH ẢNH & XUẤT G-CODE CHO UGS
# ==============================================================================
Uploaded_file = st.file_uploader("Tải lên hình ảnh mẫu tranh gỗ", type=["png", "jpg", "jpeg", "webp"])

If uploaded_file:
    Raw_img = Image.open(uploaded_file).convert("RGB")
    If max(raw_img.size) > 1000:
        Raw_img.thumbnail((1000, 1000))
    
    Gray_img = cv2.cvtColor(np.array(raw_img), cv2.COLOR_RGB2GRAY)
    
    Sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=5)
    Sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=5)
    Edge_gradient = np.sqrt(sobelx**2 + sobely**2)
    
    Base_depth = (255 - gray_img).astype(np.float64)
    Max_grad = edge_gradient.max() if edge_gradient.max() > 0 else 1e-5
    Ai_depth_raw = base_depth * 0.7 + (edge_gradient / max_grad * 255) * 0.3
    Ai_depth_raw = cv2.GaussianBlur(ai_depth_raw, (7, 7), 0)
    
    Depth_map_mm = (ai_depth_raw / ai_depth_raw.max()) * relief_depth
    Img_h, img_w = depth_map_mm.shape
    Scale_x = stock_x / img_w
    Scale_y = stock_y / img_h

    If "Center" in work_zero:
        Offset_x, offset_y = -stock_x / 2.0, -stock_y / 2.0
    Elif "Top-Left" in work_zero:
        Offset_x, offset_y = 0.0, -stock_y
    Elif "Top-Right" in work_zero:
        Offset_x, offset_y = -stock_x, -stock_y
    Elif "Bottom-Right" in work_zero:
        Offset_x, offset_y = -stock_x, 0.0
    Else:  # Bottom-Left
        Offset_x, offset_y = 0.0, 0.0

    Def make_grbl_header(layer_name, tool_desc, rpm):
        Return [
            F"(--- {layer_name.upper()} ---)",
            F"(TOOL: {tool_desc})",
            F"(WORK ZERO: {work_zero})",
            "G21 ; Unit mm",
            "G90 ; Absolute Coordinates",
            "G54 ; Work Coordinate System",
            F"G0 Z{safe_z:.3f}",
            F"M3 S{int(rpm)} ; Start Spindle",
            "G4 P2 ; Wait 2 sec for spindle"
        ]

    Def generate_l1_roughing():
        T = ai_rec['l1']
        Gcode = make_grbl_header("Layer 1 - Pha Tho", t['tool'], t['s'])
        Step_px = max(1, int(t['dia'] / scale_x))
        Max_z = np.max(depth_map_mm)
        Passes = int(np.ceil(max_z / t['stepdown']))
        
        For p in range(1, passes + 1):
            Cur_z = min(p * t['stepdown'], max_z)
            For y in range(0, img_h, step_px):
                X_range = range(0, img_w, step_px) if (y // step_px) % 2 == 0 else range(img_w - 1, -1, -step_px)
                For x in x_range:
                    Target_z = depth_map_mm[y, x]
                    If target_z > 0.5:
                        Cut_z = -min(cur_z, target_z)
                        Real_x = x * scale_x + offset_x
                        Real_y = (img_h - y) * scale_y + offset_y
                        Gcode.append(f"G1 X{real_x:.3f} Y{real_y:.3f} Z{cut_z:.3f} F{t['f']}")
            Gcode.append(f"G0 Z{safe_z:.3f}")
        Gcode.extend(["M5", "M30"])
        Return "\n".join(gcode)

    Def generate_l2_finishing():
        T = ai_rec['l2']
        Gcode = make_grbl_header("Layer 2 - Chay Tinh Mnin", t['tool'], t['s'])
        Step_px = max(1, int((t['dia'] * t['stepover']) / scale_x))
        
        R_px = int(np.ceil((t['dia'] / 2.0) / scale_x))
        K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r_px * 2 + 1, r_px * 2 + 1))
        Compensated_map = cv2.erode(depth_map_mm.astype(np.float32), k)
        
        For y in range(0, img_h, step_px):
            X_range = range(0, img_w, step_px) if (y // step_px) % 2 == 0 else range(img_w - 1, -1, -step_px)
            For x in x_range:
                Cut_z = -float(compensated_map[y, x])
                Real_x = x * scale_x + offset_x
                Real_y = (img_h - y) * scale_y + offset_y
                Gcode.append(f"G1 X{real_x:.3f} Y{real_y:.3f} Z{cut_z:.3f} F{t['f']}")
        Gcode.extend([f"G0 Z{safe_z:.3f}", "M5", "M30"])
        Return "\n".join(gcode)

    Def generate_l3_pencil():
        T = ai_rec['l3']
        Gcode = make_grbl_header("Layer 3 - Dieu Khac Chi Tiet", t['tool'], t['s'])
        
        Grad_thresh = max_grad * 0.35
        Mask_detail = (edge_gradient > grad_thresh).astype(np.uint8)
        
        In_cut = False
        For y in range(0, img_h, 2):
            For x in range(0, img_w, 2):
                If mask_detail[y, x] > 0:
                    Rx = x * scale_x + offset_x
                    Ry = (img_h - y) * scale_y + offset_y
                    Rz = -float(depth_map_mm[y, x])
                    If not in_cut:
                        Gcode.append(f"G0 X{rx:.3f} Y{ry:.3f}")
                        Gcode.append(f"G1 Z{rz:.3f} F{t['f']/2}")
                        In_cut = True
                    Else:
                        Gcode.append(f"G1 X{rx:.3f} Y{ry:.3f} Z{rz:.3f} F{t['f']}")
                Else:
                    If in_cut:
                        Gcode.append(f"G0 Z{safe_z:.3f}")
                        In_cut = False
        Gcode.extend([f"G0 Z{safe_z:.3f}", "M5", "M30"])
        Return "\n".join(gcode)

    Def generate_l4_cutout():
        T = ai_rec['l4']
        Gcode = make_grbl_header("Layer 4 - Cat Bien An Toan (Multi-Pass Stepdown)", t['tool'], t['s'])
        
        R_cut = t['dia'] / 2.0
        X_min, x_max = 0.0 - r_cut + offset_x, stock_x + r_cut + offset_x
        Y_min, y_max = 0.0 - r_cut + offset_y, stock_y + r_cut + offset_y
        
        Step_z = stepdown_val
        Passes = passes_val
        
        Gcode.append(f"(THONG SO CẮT: Dày phôi {stock_z}mm | Ăn Z mỗi lượt: {step_z}mm | Tong cong: {passes} luot cut)")
        Gcode.append(f"G0 X{x_min:.3f} Y{y_min:.3f}")
        
        For p in range(1, passes + 1):
            Target_cut_z = -min(p * step_z, stock_z)
            Gcode.append(f"(--- LUOT CUT AN TOAN SO {p}/{passes}: Z = {target_cut_z:.2f}mm ---)")
            Gcode.append(f"G1 Z{target_cut_z:.3f} F{t['f']/2}")
            Gcode.append(f"G1 X{x_max:.3f} Y{y_min:.3f} F{t['f']}")
            Gcode.append(f"G1 X{x_max:.3f} Y{y_max:.3f} F{t['f']}")
            Gcode.append(f"G1 X{x_min:.3f} Y{y_max:.3f} F{t['f']}")
            Gcode.append(f"G1 X{x_min:.3f} Y{y_min:.3f} F{t['f']}")
            
        Gcode.extend([f"G0 Z{safe_z:.3f}", "M5", "M30"])
        Return "\n".join(gcode)

    # ==============================================================================
    # XUẤT FILE VÀ TẢI VỀ
    # ==============================================================================
    St.markdown("---")
    St.header("💾 Tải Về File G-Code Tương Ứng Với Thông Số Đã Tính")

    Gc1 = generate_l1_roughing()
    Gc2 = generate_l2_finishing()
    Gc3 = generate_l3_pencil()
    Gc4 = generate_l4_cutout()

    C1, c2, c3, c4 = st.columns(4)
    With c1:
        St.download_button("💾 1. Phá Thô (.nc)", data=gc1, file_name="01_PhaTho_GRBL.nc")
    With c2:
        St.download_button("💾 2. Chạy Tinh (.nc)", data=gc2, file_name="02_ChayTinh_GRBL.nc")
    With c3:
        St.download_button("💾 3. Điêu Khắc (.nc)", data=gc3, file_name="03_DieuKhac_GRBL.nc")
    With c4:
        St.download_button("💾 4. Cắt Biên An Toàn (.nc)", data=gc4, file_name="04_CatBien_AnToan_GRBL.nc")

    Zip_buf = io.BytesIO()
    With zipfile.ZipFile(zip_buf, "w") as zf:
        Zf.writestr("01_PhaTho_EndMill.nc", gc1)
        Zf.writestr("02_ChayTinh_BallNose.nc", gc2)
        Zf.writestr("03_DieuKhac_Pencil.nc", gc3)
        Zf.writestr("04_CatBien_ProfileCut_AnToan.nc", gc4)

    St.markdown("---")
    St.download_button(
        Label="📦 TẢI TRỌN BỘ ZIP 4 LAYER G-CODE AN TOÀN (MỞ TRỰC TIẾP TRÊN UGS)",
        Data=zip_buf.getvalue(),
        File_name="Tron_Bo_GCode_GRBL_UGS_AnToan.zip",
        Mime="application/zip",
        Use_container_width=True
    )
