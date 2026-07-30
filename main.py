import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="Xử lý ảnh 3D CNC - Aspire", layout="wide")

st.title("🛠️ Tool Tối Ưu Ảnh Mờ Để Khắc 3D CNC (Aspire)")
st.write("Xử lý nhiễu, tăng độ nét biên dạng và tạo Depth Map mượt cho dao CNC đi tinh xảo.")

# Sidebar - Các thông số điều chỉnh
st.sidebar.header("⚙️ Tùy chỉnh tham số CNC")
upscale_factor = st.sidebar.select_slider("Tăng kích thước (Upscale)", options=[1, 2, 3, 4], value=2)
blur_strength = st.sidebar.slider("Độ mịn bề mặt (Khử gờ nhiễu)", 1, 15, 5, step=2)
sharpen_strength = st.sidebar.slider("Độ sắc nét biên dạng (Unsharp)", 0.5, 3.0, 1.5, step=0.1)
contrast = st.sidebar.slider("Độ tương phản khối (Contrast)", 0.8, 2.5, 1.2, step=0.1)

uploaded_file = st.file_uploader("Tải ảnh từ khách hàng lên (JPG, PNG, WEBP)", type=["jpg", "jpeg", "png", "webp"])

def process_cnc_image(img, scale, blur, sharp, c_val):
    # 1. Chuyển sang OpenCV format
    img_np = np.array(img)
    if len(img_np.shape) == 3 and img_np.shape[2] == 4:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
        
    # 2. Resize / Upscale (Bicubic Interpolation giúp giảm vỡ hạt)
    h, w = img_np.shape[:2]
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    # 3. Chuyển sang ảnh xám (Grayscale Depth Map)
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    
    # 4. Bilateral Filter: Làm mịn bề mặt phẳng nhưng GIỮ NGUYÊN cạnh sắc nét
    filtered = cv2.bilateralFilter(gray, d=blur, sigmaColor=75, sigmaSpace=75)
    
    # 5. Tăng độ tương phản để làm rõ khối nông/sâu
    enhanced = cv2.convertScaleAbs(filtered, alpha=c_val, beta=0)
    
    # 6. Unsharp Masking: Tăng cường nét biên cho chi tiết đục
    gaussian_blur = cv2.GaussianBlur(enhanced, (0, 0), 3)
    sharpened = cv2.addWeighted(enhanced, 1 + sharp, gaussian_blur, -sharp, 0)
    
    return sharpened

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🖼️ Ảnh gốc của khách (Mờ/Vỡ hạt)")
        st.image(image, use_container_width=True)
        st.caption(f"Kích thước gốc: {image.width} x {image.height} px")

    # Xử lý ảnh
    processed_np = process_cnc_image(image, upscale_factor, blur_strength, sharpen_strength, contrast)
    processed_img = Image.fromarray(processed_np)

    with col2:
        st.subheader("✨ Ảnh đã tối ưu (Sẵn sàng nhập Aspire)")
        st.image(processed_img, use_container_width=True)
        st.caption(f"Kích thước sau xử lý: {processed_img.width} x {processed_img.height} px")

    # Nút tải ảnh về
    buf = io.BytesIO()
    processed_img.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button(
        label="💾 Tải ảnh chuẩn CNC (PNG HQ)",
        data=byte_im,
        file_name="cnc_ready_depthmap.png",
        mime="image/png"
    )
