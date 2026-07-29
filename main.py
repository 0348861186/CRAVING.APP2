import io
import cv2
import easyocr
import mediapipe as mp
import numpy as np
from PIL import Image
from rembg import remove
import streamlit as st


# =====================================================
# 1. LOAD AI MODEL
# =====================================================

@st.cache_resource
def load_ai_models():
    reader = easyocr.Reader(['en', 'vi'], gpu=False)

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=5,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )

    return reader, face_mesh


ocr_engine, face_engine = load_ai_models()


# =====================================================
# 2. TOOL FUNCTIONS
# =====================================================

def feather_mask(mask, blur=5):
    return cv2.GaussianBlur(mask, (blur, blur), 0)


def rgba_extract(image_rgba, mask):
    layer = np.zeros_like(image_rgba)
    layer[mask > 10] = image_rgba[mask > 10]
    return layer


def image_to_png_bytes(img):
    buffer = io.BytesIO()
    Image.fromarray(img).save(buffer, format="PNG")
    return buffer.getvalue()


# =====================================================
# 3. AI LAYER DECOMPOSER
# =====================================================

class AILayerDecomposer:

    def __init__(self, image):
        self.image = image.convert("RGB")
        self.rgb = np.array(self.image)
        self.h, self.w, _ = self.rgb.shape

        self.rgba = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2RGBA)

        self.face_mask = np.zeros((self.h, self.w), np.uint8)
        self.text_mask = np.zeros((self.h, self.w), np.uint8)
        self.hole_mask = np.zeros((self.h, self.w), np.uint8)
        self.pattern_mask = np.zeros((self.h, self.w), np.uint8)
        self.object_mask = np.zeros((self.h, self.w), np.uint8)

    # -------------------------------------------------
    # FACE DETECTION
    # -------------------------------------------------

    def detect_face(self):
        result = face_engine.process(self.rgb)

        if result.multi_face_landmarks:
            oval = [
                10, 338, 297, 332,
                284, 251, 389,
                356, 454, 323,
                361, 288, 397,
                365, 379, 378,
                400, 377, 152,
                148, 176, 149,
                150, 136, 172,
                58, 132, 93,
                234, 127, 162,
                21, 54, 103,
                67, 109,
            ]

            for face in result.multi_face_landmarks:
                pts = []
                for i in oval:
                    p = face.landmark[i]
                    pts.append([int(p.x * self.w), int(p.y * self.h)])

                pts = np.array(pts, np.int32)
                cv2.fillPoly(self.face_mask, [pts], 255)

        self.face_mask = feather_mask(self.face_mask, 7)

    # -------------------------------------------------
    # OCR TEXT
    # -------------------------------------------------

    def detect_text(self):
        results = ocr_engine.readtext(self.rgb)

        for box, text, score in results:
            if score > 0.25:
                pts = np.array(box, np.int32)
                cv2.fillPoly(self.text_mask, [pts], 255)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        self.text_mask = cv2.dilate(self.text_mask, kernel)

    # -------------------------------------------------
    # OBJECT + HOLES
    # -------------------------------------------------

    def detect_object(self):
        cut = remove(self.image).convert("RGBA")
        rgba = np.array(cut)
        self.object_mask = rgba[:, :, 3]

        contours, hierarchy = cv2.findContours(
            self.object_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        if hierarchy is not None:
            hierarchy = hierarchy[0]
            for i, c in enumerate(contours):
                parent = hierarchy[i][3]
                if parent != -1:
                    if cv2.contourArea(c) > 30:
                        cv2.drawContours(self.hole_mask, [c], -1, 255, -1)

    # -------------------------------------------------
    # CNC PATTERN EXTRACTION
    # -------------------------------------------------

    def detect_pattern(self):
        gray = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        texture = cv2.Laplacian(blur, cv2.CV_8U)

        _, mask = cv2.threshold(texture, 18, 255, cv2.THRESH_BINARY)

        assigned = cv2.bitwise_or(self.face_mask, self.text_mask)
        assigned = cv2.bitwise_or(assigned, self.hole_mask)

        mask = cv2.bitwise_and(mask, self.object_mask)
        self.pattern_mask = cv2.bitwise_and(mask, cv2.bitwise_not(assigned))

    # -------------------------------------------------
    # BUILD OUTPUT LAYERS
    # -------------------------------------------------

    def build(self):
        self.detect_face()
        self.detect_text()
        self.detect_object()
        self.detect_pattern()

        layers = {}

        layers["Face Layer"] = rgba_extract(self.rgba, self.face_mask)
        layers["Text Layer"] = rgba_extract(self.rgba, self.text_mask)
        layers["Hole Layer"] = rgba_extract(self.rgba, self.hole_mask)
        layers["Pattern Layer"] = rgba_extract(self.rgba, self.pattern_mask)

        # Background remove
        bg = cv2.inpaint(self.rgb, self.object_mask, 7, cv2.INPAINT_TELEA)
        layers["Background Layer"] = cv2.cvtColor(bg, cv2.COLOR_RGB2RGBA)

        return layers


# =====================================================
# 4. STREAMLIT DASHBOARD
# =====================================================

st.set_page_config(
    page_title="AI CNC Layer Separation Studio",
    page_icon="🪵",
    layout="wide",
)

st.title("🧩 AI CNC Layer Separation Studio")

st.markdown(
    """
    Hệ thống AI tự động tách ảnh thành các Layer phục vụ:
    
    - CNC Wood Carving
    - Aspire / VCarve
    - Photoshop
    - Vector Conversion
    - Relief Preparation
    
    Layer xuất:
    
    ✅ Face  
    ✅ Text  
    ✅ Hole  
    ✅ Pattern  
    ✅ Background
    """
)

# =====================================================
# UPLOAD IMAGE
# =====================================================

uploaded = st.file_uploader(
    "📂 Chọn ảnh cần xử lý",
    type=["png", "jpg", "jpeg", "webp"],
)

if uploaded:
    image = Image.open(uploaded)
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Ảnh gốc")
        st.image(image, use_container_width=True)

        process = st.button(
            "🚀 Tách Layer bằng AI",
            type="primary",
            use_container_width=True,
        )

    if process:
        with st.spinner("AI đang phân tích OCR + Face Mesh + Object + Texture..."):
            decomposer = AILayerDecomposer(image)
            layers = decomposer.build()

        with col2:
            st.subheader("🎯 Kết quả AI Layer")

            for name, img in layers.items():
                st.divider()
                st.write("### " + name)

                preview = Image.fromarray(img)
                st.image(preview, use_container_width=True)

                png = image_to_png_bytes(img)
                filename = name.replace(" ", "_") + ".png"

                st.download_button(
                    label="⬇ Download " + filename,
                    data=png,
                    file_name=filename,
                    mime="image/png",
                )

# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    """
    AI CNC Layer Separation Engine
    - EasyOCR
    - MediaPipe Face Mesh
    - Rembg Background AI
    - OpenCV Texture Analysis
    
    Ready for next stage:
    SVG Vector / DXF / Height Map / GRBL G-code
    """
)
