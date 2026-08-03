import json
import re
from typing import Any, Dict, List, Optional
import cv2
import ezdxf
from google import genai
from google.genai import types
import numpy as np
from pydantic import BaseModel, Field

# ==========================================
# 1. BỔ SUNG: OPENCV GEOMETRY VALIDATION ENGINE
# (Dựa trên nhận xét: Dùng OpenCV kiểm chứng hình học trước khi dựng CAD)
# ==========================================


class CVGeometryEngine:
  """Thêm thuật toán OpenCV để xác định góc, tâm tròn, đường viền chuẩn pixel

  trước khi phụ thuộc hoàn toàn vào Gemini.
  """

  @staticmethod
  def validate_and_refine_geometry(cv_img: np.ndarray) -> List[Dict[str, Any]]:
    """Trích xuất các tham số hình học chuẩn xác bằng OpenCV"""
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    refined_shapes = []

    h, w = cv_img.shape[:2]
    img_area = w * h

    for cnt in contours:
      area = cv2.contourArea(cnt)
      if area < 100 or area > img_area * 0.9:  # Bỏ qua nhiễu
        continue

      peri = cv2.arcLength(cnt, True)

      # 1. HoughCircles / Circularity test cho lỗ tròn / hình tròn
      circularity = 4 * np.pi * (area / (peri * peri)) if peri > 0 else 0
      if circularity > 0.8:
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        refined_shapes.append({
            "type": "CIRCLE",
            "center_px": (round(cx, 2), round(cy, 2)),
            "radius_px": round(radius, 2),
        })
        continue

      # 2. approxPolyDP & minAreaRect cho hình chữ nhật / góc vuông (kể cả bị xoay)
      approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
      if len(approx) == 4:
        (cx, cy), (width, height), angle = cv2.minAreaRect(cnt)
        refined_shapes.append({
            "type": "RECTANGLE",
            "center_px": (round(cx, 2), round(cy, 2)),
            "width_px": round(max(width, height), 2),
            "height_px": round(min(width, height), 2),
            "angle_deg": round(angle, 2),
        })
      else:
        # 3. Đa giác tự do
        pts = approx.reshape(-1, 2).tolist()
        refined_shapes.append({"type": "POLYGON", "points_px": pts})

    return refined_shapes


# ==========================================
# 2. CODE GỐC CỦA BẠN (GIỮ NGUYÊN)
# ==========================================


class BoundingDimension(BaseModel):
  overall_width_mm: Optional[float] = Field(
      None, description="Chiều rộng tổng thể (mm)"
  )
  overall_height_mm: Optional[float] = Field(
      None, description="Chiều cao tổng thể (mm)"
  )


class FeatureNote(BaseModel):
  feature_type: str = Field(
      ..., description="HOLE, SLOT, RECTANGLE, CUTOUT, CONTOUR"
  )
  dimension_label: str = Field(..., description="Ví dụ: 'Ø20', 'R5', '10x20'")
  value_mm: float = Field(..., description="Giá trị số thực tế mm")


class DrawingAnalysisResult(BaseModel):
  overall_bounds: BoundingDimension
  features: List[FeatureNote] = Field(default_factory=list)


class PipelineCADCAM:

  def __init__(self, api_key: str):
    self.client = genai.Client(api_key=api_key)

  def preprocess_image(self, image_path: str) -> np.ndarray:
    """Nắn phẳng ảnh gốc bằng OpenCV (Code gốc của bạn)"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 200)

    contours, _ = cv2.findContours(
        edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours:
      peri = cv2.arcLength(cnt, True)
      approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
      if len(approx) == 4:
        # Thực hiện Perspective Transform nắn phẳng
        pts = approx.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        (tl, tr, br, bl) = rect
        width = max(
            np.linalg.norm(br - bl),
            np.linalg.norm(tr - tl),
        )
        height = max(
            np.linalg.norm(tr - br),
            np.linalg.norm(tl - bl),
        )

        dst = np.array(
            [
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1],
            ],
            dtype="float32",
        )
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(img, M, (int(width), int(height)))

    return img

  def analyze_with_gemini(self, cv_img: np.ndarray) -> DrawingAnalysisResult:
    """Phân tích bằng Gemini để lấy OCR kích thước & intent thiết kế (Code gốc)"""
    _, img_encoded = cv2.imencode(".jpg", cv_img)
    image_bytes = img_encoded.tobytes()

    prompt = (
        "Phân tích bản vẽ kỹ thuật này, trích xuất kích thước tổng thể và các"
        " chi tiết."
    )

    response = self.client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            prompt,
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=DrawingAnalysisResult,
            temperature=0.0,
        ),
    )
    return DrawingAnalysisResult.model_validate_json(response.text)

  # ==========================================
  # BỔ SUNG: HÀM KẾT HỢP VÀ DỰNG CAD/G-CODE TỐI ƯU
  # ==========================================
  def process_and_export(self, image_path: str, output_dxf="output.dxf"):
    # Bước 1: Nắn phẳng ảnh
    warped_img = self.preprocess_image(image_path)

    # Bước 2 (BỔ SUNG BẰNG OPENCV): Kiểm chứng hình học trước khi làm CAD
    cv_geometries = CVGeometryEngine.validate_and_refine_geometry(warped_img)
    print(
        f"[OpenCV Validation] Tìm thấy {len(cv_geometries)} hình học chuẩn"
        " pixel."
    )

    # Bước 3: Dùng Gemini đọc kích thước OCR
    analysis = self.analyze_with_gemini(warped_img)
    print(
        "[Gemini OCR] Kích thước đọc được:",
        analysis.overall_bounds.model_dump_json(),
    )

    # Bước 4: Khởi tạo DXF
    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()

    # Tính tỷ lệ scale mm/pixel (Lấy kích thước Gemini OCR chia cho Pixel OpenCV)
    scale = 1.0
    if (
        cv_geometries
        and cv_geometries[0]["type"] == "RECTANGLE"
        and analysis.overall_bounds.overall_width_mm
    ):
      scale = (
          analysis.overall_bounds.overall_width_mm
          / cv_geometries[0]["width_px"]
      )

    # Dựng CAD từ hình học OpenCV đã qua kiểm chứng
    for geo in cv_geometries:
      if geo["type"] == "CIRCLE":
        cx, cy = geo["center_px"][0] * scale, -geo["center_px"][1] * scale
        r = geo["radius_px"] * scale
        msp.add_circle((cx, cy), r)
      elif geo["type"] == "RECTANGLE":
        cx, cy = geo["center_px"][0] * scale, -geo["center_px"][1] * scale
        w, h = geo["width_px"] * scale, geo["height_px"] * scale
        pts = [
            (cx - w / 2, cy - h / 2),
            (cx + w / 2, cy - h / 2),
            (cx + w / 2, cy + h / 2),
            (cx - w / 2, cy + h / 2),
        ]
        msp.add_lwpolyline(pts, close=True)

    doc.saveas(output_dxf)
    print(f"[Export SUCCESS] File DXF chính xác đã được xuất tại: {output_dxf}")


# ==========================================
# THỰC THI
# ==========================================
if __name__ == "__main__":
  pipeline = PipelineCADCAM(api_key="YOUR_GEMINI_API_KEY")
  # pipeline.process_and_export("drawing.jpg")
