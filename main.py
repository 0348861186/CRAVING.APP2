import io
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from rembg import remove
from transformers import pipeline
import plotly.graph_objects as go
import streamlit as st

# ==============================================================================
# 1. KIẾN TRÚC REAL-ESRGAN (RRDBNET) & THUẬT TOÁN CHIA TILE
# ==============================================================================
class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock_5C(nf, gc)
        self.rdb2 = ResidualDenseBlock_5C(nf, gc)
        self.rdb3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        return self.rdb1(x) * 0.2 + self.rdb2(x) * 0.2 + self.rdb3(x) * 0.2 + x

class RRDBNet(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4):
        super(RRDBNet, self).__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1, bias=True)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)

        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out

def enhance_tile_process(img_np, model, device, tile_size=400, tile_pad=10, scale=4):
    """Cắt tile xử lý nâng phân giải AI để tránh tràn VRAM/RAM"""
    img_tensor = torch.from_numpy(np.transpose(img_np, (2, 0, 1))).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    b, c, h, w = img_tensor.size()
    output_shape = (b, c, h * scale, w * scale)
    output = torch.zeros(output_shape, device=device)

    tiles_x = (w + tile_size - 1) // tile_size
    tiles_y = (h + tile_size - 1) // tile_size

    for y in range(tiles_y):
        for x in range(tiles_x):
            ofs_x = x * tile_size
            ofs_y = y * tile_size

            input_start_x = max(ofs_x - tile_pad, 0)
            input_end_x = min(ofs_x + tile_size + tile_pad, w)
            input_start_y = max(ofs_y - tile_pad, 0)
            input_end_y = min(ofs_y + tile_size + tile_pad, h)

            input_tile = img_tensor[:, :, input_start_y:input_end_y, input_start_x:input_end_x]

            with torch.no_grad():
                output_tile = model(input_tile)

            output_start_x = input_start_x * scale
            output_end_x = input_end_x * scale
            output_start_y = input_start_y * scale
            output_end_y = input_end_y * scale

            target_start_x = ofs_x * scale
            target_end_x = min((ofs_x + tile_size) * scale, w * scale)
            target_start_y = ofs_y * scale
            target_end_y = min((ofs_y + tile_size) * scale, h * scale)

            tile_slice_x_start = (target_start_x - output_start_x)
            tile_slice_x_end = tile_slice_x_start + (target_end_x - target_start_x)
            tile_slice_y_start = (target_start_y - output_start_y)
            tile_slice_y_end = tile_slice_y_start + (target_end_y - target_start_y)

            output[:, :, target_start_y:target_end_y, target_start_x:target_end_x] = \
                output_tile[:, :, tile_slice_y_start:tile_slice_y_end, tile_slice_x_start:tile_slice_x_end]

    output_np = output.squeeze().float().cpu().clamp_(0, 1).numpy()
    output_np = np.transpose(output_np, (1, 2, 0))
    return (output_np * 255.0).round().astype(np.uint8)

# ==============================================================================
# 2. KHỞI TẠO VÀ CACHE MÔ HÌNH AI
# ==============================================================================
st.set_page_config(page_title="AI Ultra 3D CNC - Aspire Optimization 16-Bit", layout="wide")

@st.cache_resource
def load_real_esrgan_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    model_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
    checkpoint = torch.hub.load_state_dict_from_url(model_url, map_location=device)
    state_dict = checkpoint.get('params_ema', checkpoint.get('params', checkpoint))
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(device)
    return model, device

@st.cache_resource
def load_depth_model():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf", device=device)

# ==============================================================================
# 3. HÀM XỬ LÝ HÌNH ẢNH & NORMAL MAP
# ==============================================================================
def generate_normal_map(depth_float_norm, strength=1.5):
    """Tạo Normal Map từ mảng depth chuẩn hóa float32 [0, 1]"""
    zy, zx = np.gradient(depth_float_norm)
    zx, zy = zx * strength, zy * strength
    normal = np.dstack((-zx, -zy, np.ones_like(depth_float_norm)))
    n = np.linalg.norm(normal, axis=2, keepdims=True)
    return ((normal / n + 1) / 2.0 * 255).astype(np.uint8)

# ==============================================================================
# 4. GIAO DIỆN STREAMLIT
# ==============================================================================
st.title("🛠️ Tool Tối Ưu Ảnh AI 3D CNC Pro (Xuất 16-Bit Cho Aspire)")

# Sidebar Configuration
st.sidebar.header("⚙️ Tùy chỉnh tham số AI & CNC")
use_esrgan = st.sidebar.checkbox("Bật AI Super-Resolution (Real-ESRGAN 4x)", value=True)
remove_bg = st.sidebar.checkbox("Tách loại bỏ nền tự động (Rembg)", value=False)
use_ai_depth = st.sidebar.checkbox("Bật AI Depth Estimation (Depth Anything 3D)", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("📐 Tối ưu Height Map & Lọc bề mặt")
use_denoise = st.sidebar.checkbox("Bật AI Denoise (Khử hạt nhiễu)", value=True)
denoise_strength = st.sidebar.slider("Cường độ Denoise", 3, 20, 7)
blur_strength = st.sidebar.slider("Độ mịn bề mặt (Bilateral Filter)", 1, 15, 5, step=2)
gamma_val = st.sidebar.slider("Độ dốc khối (Gamma Curved)", 0.3, 2.5, 1.0, step=0.1)
invert_z = st.sidebar.checkbox("🔄 Đảo ngược chiều Z (Invert Lồi/Lõm)", value=False)
normal_strength = st.sidebar.slider("Độ sâu Normal Map", 0.5, 5.0, 1.5, step=0.1)

uploaded_file = st.file_uploader("Tải ảnh từ khách hàng (JPG, PNG, WEBP)", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image_input = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🖼️ Ảnh gốc")
        st.image(image_input, use_container_width=True)
        st.caption(f"Kích thước gốc: {image_input.width} x {image_input.height} px")

    with st.spinner("🚀 AI đang xử lý (Upscale Tile, Depth Anything 16-Bit & Filtering)..."):
        # 1. Tách nền (nếu bật)
        alpha_mask = None
        if remove_bg:
            img_no_bg = remove(image_input)
            img_np_bg = np.array(img_no_bg)
            if img_np_bg.shape[2] == 4:
                alpha_mask = img_np_bg[:, :, 3]
            image_proc = img_no_bg.convert("RGB")
        else:
            image_proc = image_input

        # 2. Upscale Real-ESRGAN với Tiling Chống Tràn RAM
        img_np = np.array(image_proc)
        if use_esrgan:
            esr_model, device = load_real_esrgan_model()
            upscaled_np = enhance_tile_process(img_np, esr_model, device, tile_size=400, scale=4)
        else:
            upscaled_np = img_np

        # 3. Denoise trước khi tạo khối
        if use_denoise:
            upscaled_np = cv2.fastNlMeansDenoisingColored(
                upscaled_np, None, h=denoise_strength, hColor=denoise_strength, templateWindowSize=7, searchWindowSize=21
            )

        # 4. Tạo Depth Map (Mô hình AI chuyên sâu hoặc Chuyển Xám)
        if use_ai_depth:
            depth_pipe = load_depth_model()
            depth_pil = depth_pipe(Image.fromarray(upscaled_np))["depth"]
            depth_raw = np.array(depth_pil, dtype=np.float32)
        else:
            depth_raw = cv2.cvtColor(upscaled_np, cv2.COLOR_RGB2GRAY).astype(np.float32)

        # 5. Chuẩn hóa dải float32 [0.0, 1.0]
        d_min, d_max = depth_raw.min(), depth_raw.max()
        if d_max - d_min > 0:
            depth_norm = (depth_raw - d_min) / (d_max - d_min)
        else:
            depth_norm = np.zeros_like(depth_raw, dtype=np.float32)

        # Đảo chiều Z nếu chọn
        if invert_z:
            depth_norm = 1.0 - depth_norm

        # Gamma Correction
        if gamma_val != 1.0:
            depth_norm = np.power(depth_norm, gamma_val)

        # Áp dụng Mask loại bỏ nền tuyệt đối nếu có
        if alpha_mask is not None:
            mask_resized = cv2.resize(alpha_mask, (depth_norm.shape[1], depth_norm.shape[0]))
            depth_norm[mask_resized == 0] = 0.0

        # 6. Chuyển sang dải 16-bit (0 -> 65535) & Lọc mịn Bilateral Filter
        depth_16bit_f = depth_norm * 65535.0
        smoothed = cv2.bilateralFilter(depth_16bit_f.astype(np.float32), d=blur_strength, sigmaColor=75, sigmaSpace=75)

        if alpha_mask is not None:
            smoothed[mask_resized == 0] = 0.0

        final_depth_16bit = np.clip(smoothed, 0, 65535).astype(np.uint16)

        # 7. Tạo Normal Map từ dải depth_norm
        normal_np = generate_normal_map(depth_norm, strength=normal_strength)

        # Tạo đối tượng ảnh 16-bit PIL để lưu file
        depth_img_16bit = Image.fromarray(final_depth_16bit, mode="I;16")
        normal_img = Image.fromarray(normal_np)

        # Mẫu 8-bit phục vụ hiển thị preview mượt trên web
        preview_8bit = (final_depth_16bit / 256).astype(np.uint8)

    with col2:
        st.subheader("✨ AI Depth Map 16-Bit (Chuẩn Aspire Relief)")
        st.image(preview_8bit, use_container_width=True)
        st.caption(f"Kích thước sau xử lý: {depth_img_16bit.width} x {depth_img_16bit.height} px (Dải độ sâu 65,536 mức)")

    st.markdown("---")
    st.subheader("🗺️ Normal Map (Kiểm tra góc dốc đường đục)")
    st.image(normal_img, use_container_width=True)

    # 3D Interactive Preview
    st.markdown("---")
    st.subheader("🧊 Mô phỏng xem trước Relief 3D")
    preview_size = 200
    depth_small = cv2.resize(final_depth_16bit, (preview_size, preview_size)).astype(np.float32) / 65535.0
    x = np.linspace(0, 1, preview_size)
    y = np.linspace(0, 1, preview_size)
    X, Y = np.meshgrid(x, y)
    Z = np.flipud(depth_small)

    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Earth')])
    fig.update_layout(
        title='Mô hình 3D Relief (Dùng chuột xoay/phóng to để kiểm tra độ nông sâu)',
        autosize=False, width=800, height=600,
        margin=dict(l=10, r=10, b=10, t=40),
        scene=dict(zaxis=dict(range=[0, 1]))
    )
    st.plotly_chart(fig, use_container_width=True)

    # Nút Tải File
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        buf_depth = io.BytesIO()
        depth_img_16bit.save(buf_depth, format="PNG")
        st.download_button(
            "💾 Tải Depth Map 16-Bit (.PNG cho Aspire)",
            data=buf_depth.getvalue(),
            file_name="cnc_ai_depthmap_16bit.png",
            mime="image/png"
        )
    with col_dl2:
        buf_normal = io.BytesIO()
        normal_img.save(buf_normal, format="PNG")
        st.download_button(
            "💾 Tải Normal Map (PNG HQ)",
            data=buf_normal.getvalue(),
            file_name="cnc_ai_normalmap.png",
            mime="image/png"
        )
