import io
import cv2
import easyocr
import mediapipe as mp
import numpy as np
from PIL import Image
import streamlit as st


# =====================================================
# STREAMLIT CONFIG PHẢI Ở ĐẦU
# =====================================================

st.set_page_config(
    page_title="AI CNC Layer Studio",
    page_icon="🪵",
    layout="wide"
)


# =====================================================
# LOAD AI
# =====================================================

@st.cache_resource
def load_models():

    reader = easyocr.Reader(
        ['en','vi'],
        gpu=False
    )


    face = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=5,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )


    return reader,face



ocr_engine, face_engine = load_models()



# =====================================================
# TOOL
# =====================================================

def png_bytes(img):

    buf = io.BytesIO()

    Image.fromarray(img).save(
        buf,
        format="PNG"
    )

    return buf.getvalue()



def extract_rgba(img,mask):

    layer=np.zeros_like(img)

    layer[mask>20]=img[mask>20]

    return layer



# =====================================================
# AI CLASS
# =====================================================

class AILayerDecomposer:


    def __init__(self,image):

        self.image=image.convert("RGB")


        # giảm tải RAM
        if max(self.image.size)>1600:

            self.image.thumbnail(
                (1600,1600)
            )


        self.rgb=np.array(
            self.image
        )


        self.h,self.w,_=self.rgb.shape


        self.rgba=cv2.cvtColor(
            self.rgb,
            cv2.COLOR_RGB2RGBA
        )


        self.face_mask=np.zeros(
            (self.h,self.w),
            np.uint8
        )

        self.text_mask=np.zeros(
            (self.h,self.w),
            np.uint8
        )


        self.object_mask=np.zeros(
            (self.h,self.w),
            np.uint8
        )


        self.pattern_mask=np.zeros(
            (self.h,self.w),
            np.uint8
        )



    # -------------------------------
    # FACE
    # -------------------------------

    def detect_face(self):

        result=face_engine.process(
            self.rgb
        )


        if result.multi_face_landmarks:


            oval=[
            10,338,297,332,
            284,251,389,356,
            454,323,361,288,
            397,365,379,378,
            400,377,152,148,
            176,149,150,136,
            172,58,132,93,
            234,127,162,21,
            54,103,67,109
            ]


            for face in result.multi_face_landmarks:


                pts=[]


                for i in oval:

                    p=face.landmark[i]

                    pts.append([
                        int(p.x*self.w),
                        int(p.y*self.h)
                    ])


                cv2.fillPoly(
                    self.face_mask,
                    [np.array(pts)],
                    255
                )


        self.face_mask=cv2.GaussianBlur(
            self.face_mask,
            (7,7),
            0
        )



    # -------------------------------
    # TEXT OCR
    # -------------------------------

    def detect_text(self):

        results=ocr_engine.readtext(
            self.rgb
        )


        for box,text,score in results:

            if score>0.25:

                cv2.fillPoly(
                    self.text_mask,
                    [np.array(box,np.int32)],
                    255
                )


        self.text_mask=cv2.dilate(
            self.text_mask,
            np.ones((5,5),np.uint8)
        )
        # =====================================================
# OBJECT SEGMENTATION (THAY CHO REMBG)
# =====================================================

    def detect_object(self):

        gray = cv2.cvtColor(
            self.rgb,
            cv2.COLOR_RGB2GRAY
        )


        # Tạo mask đối tượng bằng threshold thích nghi
        blur = cv2.GaussianBlur(
            gray,
            (5,5),
            0
        )


        _, mask = cv2.threshold(
            blur,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )


        # Làm sạch nhiễu
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (7,7)
        )


        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )


        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )


        self.object_mask = mask



    # =================================================
    # CNC PATTERN DETECTION
    # =================================================

    def detect_pattern(self):


        gray = cv2.cvtColor(
            self.rgb,
            cv2.COLOR_RGB2GRAY
        )


        blur = cv2.GaussianBlur(
            gray,
            (5,5),
            0
        )


        edge = cv2.Canny(
            blur,
            50,
            150
        )


        kernel=cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (9,9)
        )


        texture=cv2.dilate(
            edge,
            kernel,
            iterations=1
        )


        assigned=cv2.bitwise_or(
            self.face_mask,
            self.text_mask
        )


        texture=cv2.bitwise_and(
            texture,
            self.object_mask
        )


        self.pattern_mask=cv2.bitwise_and(
            texture,
            cv2.bitwise_not(assigned)
        )



    # =================================================
    # BUILD LAYERS
    # =================================================

    def build(self):


        self.detect_face()

        self.detect_text()

        self.detect_object()

        self.detect_pattern()



        layers={}



        layers["Face_Layer.png"] = extract_rgba(
            self.rgba,
            self.face_mask
        )


        layers["Text_Layer.png"] = extract_rgba(
            self.rgba,
            self.text_mask
        )


        layers["Pattern_Layer.png"] = extract_rgba(
            self.rgba,
            self.pattern_mask
        )



        # Object layer

        layers["Object_Mask.png"] = extract_rgba(
            self.rgba,
            self.object_mask
        )



        # Background

        bg=cv2.inpaint(
            self.rgb,
            self.object_mask,
            7,
            cv2.INPAINT_TELEA
        )


        layers["Background_Layer.png"]=cv2.cvtColor(
            bg,
            cv2.COLOR_RGB2RGBA
        )


        return layers




# =====================================================
# STREAMLIT DASHBOARD
# =====================================================


st.title(
    "🪵 AI CNC Wood Carving Layer Studio"
)


st.markdown(
"""
### AI Image Separation

Xuất Layer:

✅ Face  
✅ Text  
✅ Pattern  
✅ Object  
✅ Background  

Dùng cho:

- Aspire
- VCarve
- Photoshop
- CNC Relief
"""
)



uploaded = st.file_uploader(
    "📂 Upload ảnh",
    type=[
        "png",
        "jpg",
        "jpeg",
        "webp"
    ]
)



if uploaded:


    image=Image.open(
        uploaded
    )


    col1,col2=st.columns(
        [1,2]
    )


    with col1:

        st.subheader(
            "Ảnh gốc"
        )


        st.image(
            image,
            use_container_width=True
        )


        run=st.button(
            "🚀 Phân tích AI",
            type="primary",
            use_container_width=True
        )



    if run:


        with st.spinner(
            "AI đang phân tích..."
        ):


            ai=AILayerDecomposer(
                image
            )


            layers=ai.build()



        with col2:


            st.subheader(
                "Kết quả Layer"
            )


            for name,img in layers.items():


                st.divider()


                st.write(
                    "### "+name
                )


                st.image(
                    img,
                    use_container_width=True
                )


                st.download_button(

                    "⬇ Download "+name,

                    data=png_bytes(img),

                    file_name=name,

                    mime="image/png"

                )



# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
"""
AI CNC Layer Engine V2

Modules:
- EasyOCR
- MediaPipe Face Mesh
- OpenCV Segmentation
- Texture Analysis

Next:
SVG / DXF / Height Map / GRBL G-code
"""
)
