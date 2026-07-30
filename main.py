import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import io
import zipfile

# ==============================================================================
# CẤU HÌNH TRANG WEB
# ==============================================================================
st.set_page_config(
    page_title="AI Wood Multi-Layer CAM System",
    page_icon="🪵",
    layout="wide"
)

st.title("🪵 Hệ Thống CAM Tranh Gỗ 3D Phân Lớp (Multi-Layer CAM & G-Code Export)")
st.caption("Giải pháp gia công thực tế: Tách Layer Phá thô - Chạy tinh - Điêu khắc chi tiết & Xuất G-Code độc lập theo từng dao")
st.markdown("---")

WOOD_COLORSCALE = [
    [0.0, 'rgb(50, 25, 10)'],
    [0.5, 'rgb(140, 80, 35)'],
    [1.0, 'rgb(215, 155, 90)']
]

# ==============================================================================
# SIDEBAR: THIẾT LẬP PHÔI & CẤU HÌNH DAO CHO TỪNG LAYER
# ==============================================================================
with st.sidebar:
    st.header("📂 1. Kích Thước Phôi & Tọa Độ")
    stock_x = st.number_input("Dài X (mm)", value=300.0, step=10.0)
    stock_y = st.number_input("Rộng Y (mm)", value=200.0, step=10.0)
    stock_z = st.number_input("Dày Z (mm)", value=30.0, step=5.0)
    relief_depth = st.number_input("Sâu tranh Max Z (mm)", value=15.0, step=1.0)
    safe_z = st.number_input("Safe Z (mm)", value=10.0, step=1.0)
    post_proc = st.selectbox("Post Processor", ["Mach3/Mach4", "GRBL", "LinuxCNC", "Syntec"])

    st.markdown("---")
    st.header("⚙️ 2. Thiết Lập Dao Cho 3 Layer")
    
    # --- LAYER 1: PHÁ THÔ ---
    st.subheader("🔴 Layer 1: Phá Thô (Roughing)")
    tool1_dia = st.number_input("Đường kính Dao Phá (End Mill mm)", value=6.0, step=0.5)
    tool1_stepdown = st.number_input("Lớp cắt Z Stepdown (mm)", value=4.0, step=0.5)
    tool1_feed = st.number_input("Tốc độ ăn dao F1 (mm/min)", value=2000, step=100)
    
    # --- LAYER 2: CHẠY TINH ---
    st.subheader("🟢 Layer 2: Bề Mặt Mịn (Finishing)")
    tool2_dia = st.number_input("Đường kính Dao Cầu (Ball Nose mm)", value=3.0, step=0.5)
    tool2_stepover = st.slider("Dịch dao % Stepover 2", 5, 30, 10) / 100.0
    tool2_feed = st.number_input("Tốc độ ăn dao F2 (mm/min)", value=2500, step=100)

    # --- LAYER 3: CHI TIẾT SẮC NÉT ---
    st.subheader("🔵 Layer 3: Điêu Khắc Chi Tiết (Pencil/Detail)")
    tool3_dia = st.number_input("Mũi Dao V-Bit / Cầu Nhỏ (mm)", value=1.0, step=0.1)
    detail_sensitivity = st.slider("Ngưỡng nhận diện nét đục (Detail Sensitivity)", 1, 10, 5)
    tool3_feed = st.number_input("Tốc độ ăn dao F3 (mm/min)", value=1200, step=100)

# ==============================================================================
# XỬ LÝ ẢNH & TẠO DEPTH MAP
# ==============================================================================
uploaded_file = st.file_uploader("Tải lên ảnh mẫu tranh gỗ", type=["png", "jpg", "jpeg", "webp"])

if uploaded_file:
    raw_img = Image.open(uploaded_file).convert("RGB")
    if max(raw_img.size) > 1000:
        raw_img.thumbnail((1000, 1000))
    
    gray_img = cv2.cvtColor(np.array(raw_img), cv2.COLOR_RGB2GRAY)
    
    # AI/Gradient Depth Map Generation
    sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=5)
    edge_gradient = np.sqrt(sobelx**2 + sobely**2)
    
    base_depth = (255 - gray_img).astype(np.float64)
    max_grad = edge_gradient.max() if edge_gradient.max() > 0 else 1e-5
    ai_depth_raw = base_depth * 0.7 + (edge_gradient / max_grad * 255) * 0.3
    ai_depth_raw = cv2.GaussianBlur(ai_depth_raw, (7, 7), 0)
    
    # Ma trận độ sâu Z thực tế (mm)
    depth_map_mm = (ai_depth_raw / ai_depth_raw.max()) * relief_depth
    img_h, img_w = depth_map_mm.shape
    scale_x = stock_x / img_w
    scale_y = stock_y / img_h

    # ==============================================================================
    # BƯỚC THUẬT TOÁN: TÁCH 3 LAYER PHÂN VÙNG GIA CÔNG
    # ==============================================================================
    st.header("🎯 Phân Tách Layer & Mặt Nạ Gia Công (Toolpath Masking)")
    
    col_l1, col_l2, col_l3 = st.columns(3)
    
    with col_l1:
        st.subheader("🔴 Layer 1: Phá Thô")
        st.write("Cắt phôi theo từng lát ngang $Z$ để bỏ gỗ thừa.")
        # Layer 1 phủ toàn bộ vùng có độ sâu > 0.5mm
        mask_l1 = (depth_map_mm > 0.5).astype(np.uint8) * 255
        st.image(mask_l1, caption="Vùng phá thô (Roughing Area)", use_column_width=True)

    with col_l2:
        st.subheader("🟢 Layer 2: Chạy Mịn")
        st.write("Chạy phủ toàn bộ lòng bản đồ 3D Relief.")
        mask_l2 = (depth_map_mm > 0.1).astype(np.uint8) * 255
        st.image(mask_l2, caption="Vùng quét mịn (Finishing Area)", use_column_width=True)

    with col_l3:
        st.subheader("🔵 Layer 3: Điêu Khắc Nét")
        st.write("Chỉ lọc lấy góc hẹp, nét đục, hoa văn sâu.")
        # Tách layer chi tiết dựa trên ngưỡng Gradient dốc
        grad_thresh = (11 - detail_sensitivity) * (max_grad / 15.0)
        mask_l3 = (edge_gradient > grad_thresh).astype(np.uint8) * 255
        st.image(mask_l3, caption="Vùng khắc chi tiết (Pencil Area)", use_column_width=True)

    # ==============================================================================
    # HÀM TẠO G-CODE ĐỘC LẬP CHO TỪNG LAYER
    # ==============================================================================
    def build_layer_1_gcode(depth_mat, tool_d, s_down, feed, s_z, px_x, px_y):
        """G-Code Phá thô nhiều lớp Z (Multi-Pass Stepdown)"""
        h, w = depth_mat.shape
        step_px = max(1, int(tool_d / px_x))
        max_z = np.max(depth_mat)
        passes = int(np.ceil(max_z / s_down))
        
        gcode = [f"(--- LAYER 1: PHÁ THÔ - DAO END MILL {tool_d}mm ---)", "G21", "G90", "G54", f"G0 Z{s_z:.3f}", "M3 S12000"]
        
        for p in range(1, passes + 1):
            current_target_z = min(p * s_down, max_z)
            gcode.append(f"(--- PASS Z = -{current_target_z:.2f}mm ---)")
            
            for y in range(0, h, step_px):
                x_range = range(0, w, step_px) if (y // step_px) % 2 == 0 else range(w - 1, -1, -step_px)
                for x in x_range:
                    target_z = depth_mat[y, x]
                    if target_z > 0.5:
                        cut_z = -min(current_target_z, target_z)
                        gcode.append(f"G1 X{x*px_x:.3f} Y{y*px_y:.3f} Z{cut_z:.3f} F{feed}")
            gcode.append(f"G0 Z{s_z:.3f}")
            
        gcode.extend(["M5", "M30"])
        return "\n".join(gcode)

    def build_layer_2_gcode(depth_mat, tool_d, s_over, feed, s_z, px_x, px_y):
        """G-Code Chạy Tinh 3D Surface (Raster Scanning)"""
        h, w = depth_mat.shape
        step_px = max(1, int((tool_d * s_over) / px_x))
        
        gcode = [f"(--- LAYER 2: CHẠY TINH MỊN - DAO BALL NOSE {tool_d}mm ---)", "G21", "G90", "G54", f"G0 Z{s_z:.3f}", "M3 S18000"]
        
        for y in range(0, h, step_px):
            x_range = range(0, w, step_px) if (y // step_px) % 2 == 0 else range(w - 1, -1, -step_px)
            for x in x_range:
                cut_z = -float(depth_mat[y, x])
                gcode.append(f"G1 X{x*px_x:.3f} Y{y*px_y:.3f} Z{cut_z:.3f} F{feed}")
                
        gcode.extend([f"G0 Z{s_z:.3f}", "M5", "M30"])
        return "\n".join(gcode)

    def build_layer_3_gcode(depth_mat, mask_detail, tool_d, feed, s_z, px_x, px_y):
        """G-Code Điêu khắc nét đục - Chỉ chạy tại vùng Mask Layer 3"""
        h, w = depth_mat.shape
        gcode = [f"(--- LAYER 3: ĐIÊU KHẮC CHI TIẾT - DAO V-BIT/SMALL {tool_d}mm ---)", "G21", "G90", "G54", f"G0 Z{s_z:.3f}", "M3 S20000"]
        
        in_cutting_zone = False
        for y in range(0, h, 2):
            for x in range(0, w, 2):
                if mask_detail[y, x] > 0: # Chỉ đi dao vào vùng chi tiết sắc nét
                    rx, ry, rz = x * px_x, y * px_y, -float(depth_mat[y, x])
                    if not in_cutting_zone:
                        gcode.append(f"G0 X{rx:.3f} Y{ry:.3f}")
                        gcode.append(f"G1 Z{rz:.3f} F{feed/2}")
                        in_cutting_zone = True
                    else:
                        gcode.append(f"G1 X{rx:.3f} Y{ry:.3f} Z{rz:.3f} F{feed}")
                else:
                    if in_cutting_zone:
                        gcode.append(f"G0 Z{s_z:.3f}")
                        in_cutting_zone = False
                        
        gcode.extend([f"G0 Z{s_z:.3f}", "M5", "M30"])
        return "\n".join(gcode)

    # ==============================================================================
    # XUẤT FILE G-CODE THEO LAYER
    # ==============================================================================
    st.markdown("---")
    st.header("💾 Xuất Bộ File Mã Lệnh G-Code Theo Layer Gia Công")

    col_out1, col_out2, col_out3 = st.columns(3)
    
    # Bù bán kính dao cho Layer 2
    r2_px = int(np.ceil((tool2_dia / 2.0) / scale_x))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r2_px * 2 + 1, r2_px * 2 + 1))
    depth_comp_l2 = cv2.erode(depth_map_mm.astype(np.float32), k2)

    gcode_l1 = build_layer_1_gcode(depth_map_mm, tool1_dia, tool1_stepdown, tool1_feed, safe_z, scale_x, scale_y)
    gcode_l2 = build_layer_2_gcode(depth_comp_l2, tool2_dia, tool2_stepover, tool2_feed, safe_z, scale_x, scale_y)
    gcode_l3 = build_layer_3_gcode(depth_map_mm, mask_l3, tool3_dia, tool3_feed, safe_z, scale_x, scale_y)

    with col_out1:
        st.write("**Layer 1 (Phá thô - End Mill):**")
        st.download_button("💾 Tải G-Code Layer 1 (.nc)", data=gcode_l1, file_name="NC_Layer1_PhaTho.nc")

    with col_out2:
        st.write("**Layer 2 (Chạy tinh - Ball Nose):**")
        st.download_button("💾 Tải G-Code Layer 2 (.nc)", data=gcode_l2, file_name="NC_Layer2_ChayTinh.nc")

    with col_out3:
        st.write("**Layer 3 (Nét đục - Pencil V-Bit):**")
        st.download_button("💾 Tải G-Code Layer 3 (.nc)", data=gcode_l3, file_name="NC_Layer3_DieuKhac.nc")

    # Đóng gói toàn bộ thành 1 File ZIP
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        zip_file.writestr("01_Layer1_PhaTho_EndMill.nc", gcode_l1)
        zip_file.writestr("02_Layer2_ChayTinh_BallNose.nc", gcode_l2)
        zip_file.writestr("03_Layer3_DieuKhac_Pencil.nc", gcode_l3)

    st.markdown("---")
    st.download_button(
        label="📦 TẢI TRỌN BỘ G-CODE (ZIP FILE DÀNH CHO XƯỞNG GIA CÔNG)",
        data=zip_buffer.getvalue(),
        file_name="Tron_Bo_GCode_Tranh_Go_3D.zip",
        mime="application/zip",
        use_container_width=True
    )
