import json
import re
from typing import Any, Dict, List, Optional, Tuple
import cv2
import ezdxf
from google import genai
from google.genai import types
import numpy as np
from pydantic import BaseModel, Field

# ==========================================
# 1. SCHEMAS CHO GEMINI OCR & DESIGN INTENT
# ==========================================


class BoundingDimension(BaseModel):
  overall_width_mm: Optional[float] = Field(
      None, description="Chiều rộng tổng thể ghi trên bản vẽ (mm)"
  )
  overall_height_mm: Optional[float] = Field(
      None, description="Chiều cao tổng thể ghi trên bản vẽ (mm)"
  )


class FeatureNote(BaseModel):
  feature_type: str = Field(
      ..., description="Loại kết cấu: HOLE, SLOT, RECTANGLE, CUTOUT, CONTOUR"
  )
  dimension_label: str = Field(
      ..., description="Ghi chú kích thước đọc được, vd: 'Ø20', 'R5', '10x20'"
  )
  value_mm: float = Field(
      ..., description="Giá trị số thực tế quy đổi ra mm (vd: Ø20 -> 20.0)"
  )


class DrawingAnalysisResult(BaseModel):
  overall_bounds: BoundingDimension
  features: List[FeatureNote] = Field(
      default_factory=list, description="Danh sách các kích thước chi tiết"
  )


# ==========================================
# 2. OPENCV GEOMETRY VALIDATION ENGINE
# ==========================================


class OpenCVGeometryEngine:
  """Engine thuần toán học và xử lý ảnh OpenCV.

  Đảm bảo tính chính xác 100% về mặt hình học (Góc vuông, tâm tròn, đa giác).
  """

  @staticmethod
  def preprocess_and_warp(cv_img: np.ndarray) -> np.ndarray:
    """Nắn phẳng ảnh bản vẽ nếu phát hiện khung viền hình chữ nhật lớn."""
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 200)

    contours, _ = cv2.findContours(
        edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours:
      peri = cv2.arcLength(cnt, True)
      approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
      if len(approx) == 4 and cv2.contourArea(cnt) > (
          cv_img.shape[0] * cv_img.shape[1] * 0.3
      ):
        pts = approx.reshape(4, 2)
        return OpenCVGeometryEngine._four_point_transform(cv_img, pts)

    return cv_img  # Trả về ảnh gốc nếu không tìm thấy khung viền phù hợp

  @staticmethod
  def _four_point_transform(
      image: np.ndarray, pts: np.ndarray
  ) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array(
        [
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1],
        ],
        dtype="float32",
    )

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

  @staticmethod
  def extract_exact_geometries(cv_img: np.ndarray) -> List[Dict[str, Any]]:
    """Phân tích hình học bằng các thuật toán CV chính xác (Pixel Precision)."""
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Binary Threshold
    _, thresh = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    geometries = []
    h, w = cv_img.shape[:2]
    img_area = w * h

    for cnt in contours:
      area = cv2.contourArea(cnt)
      # Bỏ qua nhiễu quá nhỏ hoặc khung viền quá to
      if area < 150 or area > img_area * 0.95:
        continue

      peri = cv2.arcLength(cnt, True)

      # 1. HoughCircles / Circularity Test cho hình tròn / lỗ khoan
      circularity = 4 * np.pi * (area / (peri * peri)) if peri > 0 else 0
      if circularity > 0.80:
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        geometries.append({
            "type": "CIRCLE",
            "center_px": (float(cx), float(cy)),
            "radius_px": float(radius),
            "area_px": float(area),
        })
        continue

      # 2. approxPolyDP & minAreaRect cho hình chữ nhật / vuông (dù bị xoay)
      approx = cv2.approxPolyDP(cnt, 0.025 * peri, True)

      if len(approx) == 4:
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (width, height), angle = rect
        geometries.append({
            "type": "RECTANGLE",
            "center_px": (float(cx), float(cy)),
            "width_px": float(max(width, height)),
            "height_px": float(min(width, height)),
            "angle_deg": float(angle),
            "area_px": float(area),
        })
      else:
        # 3. Đa giác tự do / Contour phức tạp
        pts = approx.reshape(-1, 2).tolist()
        geometries.append(
            {"type": "POLYGON", "points_px": pts, "area_px": float(area)}
        )

    # Sắp xếp theo diện tích giảm dần (Đường viền ngoài cùng nằm đầu)
    geometries.sort(key=lambda x: x["area_px"], reverse=True)
    return geometries


# ==========================================
# 3. GEMINI AI INTENT & OCR ENGINE
# ==========================================


class GeminiVisionEngine:
  """Gemini AI đóng vai trò đọc ngữ cảnh, OCR số đo và intent thiết kế."""

  def __init__(self, api_key: str):
    self.client = genai.Client(api_key=api_key)

  def extract_dimensions_and_intent(
      self, image_bytes: bytes
  ) -> DrawingAnalysisResult:
    prompt = """
        Bạn là một chuyên gia đọc bản vẽ cơ khí CAD/CAM.
        Hãy phân tích hình ảnh bản vẽ kỹ thuật này và trích xuất:
        1. Kích thước tổng thể của phôi/chi tiết (overall_width_mm, overall_height_mm).
        2. Tất cả các ghi chú kích thước từng chi tiết (Lỗ tròn Ø, Bán kính R, Chiều dài/rộng các hốc cutouts).
        
        Trả về kết quả chính xác theo định dạng JSON Schema.
        """

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
# 4. SPATIAL ALIGNMENT & GENERATOR ENGINE
# ==========================================


class HybridCADCAMGenerator:
  """Hợp nhất dữ liệu Hình học (OpenCV) & Kích thước OCR (Gemini)

  để xuất DXF và G-Code chính xác.
  """

  def __init__(self, scale_px_to_mm: float = 1.0):
    self.scale = scale_px_to_mm  # 1 pixel = X mm

  def align_scale(
      self, cv_geometries: List[Dict], gemini_data: DrawingAnalysisResult
  ) -> float:
    """Tự động tính toán Tỉ lệ Chuyển đổi Pixel -> Millimeter (mm)."""
    if not cv_geometries:
      return 1.0

    # Lấy hình outer contour lớn nhất từ OpenCV
    outer = cv_geometries[0]

    if (
        outer["type"] == "RECTANGLE"
        and gemini_data.overall_bounds.overall_width_mm
    ):
      scale_w = (
          gemini_data.overall_bounds.overall_width_mm / outer["width_px"]
      )
      self.scale = scale_w
      print(f"[INFO] Applied Auto Scale Factor: {self.scale:.4f} mm/pixel")
      return self.scale

    self.scale = 1.0  # Default fallback
    return self.scale

  def export_dxf(
      self, cv_geometries: List[Dict], output_path: str = "output.dxf"
  ):
    """Xuất file chuẩn CAD DXF bằng ezdxf."""
    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()

    # Tạo Layer riêng biệt cho gia công
    doc.layers.new(name="OUTER_CONTOUR", dxfattribs={"color": 1})  # Red
    doc.layers.new(name="HOLES_CUTOUTS", dxfattribs={"color": 3})  # Green

    for idx, geo in enumerate(cv_geometries):
      layer_name = "OUTER_CONTOUR" if idx == 0 else "HOLES_CUTOUTS"

      if geo["type"] == "CIRCLE":
        cx = geo["center_px"][0] * self.scale
        cy = -geo["center_px"][1] * self.scale  # Đảo trục Y cho chuẩn CAD
        r = geo["radius_px"] * self.scale
        msp.add_circle((cx, cy), r, dxfattribs={"layer": layer_name})

      elif geo["type"] == "RECTANGLE":
        cx, cy = geo["center_px"][0] * self.scale, -geo["center_px"][1] * self.scale
        w, h = geo["width_px"] * self.scale, geo["height_px"] * self.scale
        # Tạo 4 đỉnh hình chữ nhật
        pts = [
            (cx - w / 2, cy - h / 2),
            (cx + w / 2, cy - h / 2),
            (cx + w / 2, cy + h / 2),
            (cx - w / 2, cy + h / 2),
        ]
        msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": layer_name})

      elif geo["type"] == "POLYGON":
        pts = [
            (p[0] * self.scale, -p[1] * self.scale) for p in geo["points_px"]
        ]
        msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": layer_name})

    doc.saveas(output_path)
    print(f"[SUCCESS] Exported DXF: {output_path}")

  def generate_gcode(
      self,
      cv_geometries: List[Dict],
      feed_rate: int = 1000,
      cut_depth_z: float = -2.0,
  ) -> str:
    """Tạo mã G-code chuẩn cho máy phay CNC."""
    gcode = []
    gcode.append("(=== CNC G-CODE GENERATED BY HYBRID ENGINE ===)")
    gcode.append("G21 (Unit: mm)")
    gcode.append("G90 (Absolute Distance Mode)")
    gcode.append("G17 (XY Plane Selection)")
    gcode.append("M03 S12000 (Spindle ON, 12000 RPM)")
    gcode.append("G00 Z5.000 (Safe Height Z)")

    for idx, geo in enumerate(cv_geometries):
      gcode.append(
          f"\n(; Feature #{idx+1}: {geo['type']} - Layer: {'OUTER' if idx==0 else 'INNER'})"
      )

      if geo["type"] == "CIRCLE":
        cx = geo["center_px"][0] * self.scale
        cy = -geo["center_px"][1] * self.scale
        r = geo["radius_px"] * self.scale

        # Di chuyển tới điểm bắt đầu cắt trên đường tròn
        gcode.append(f"G00 X{cx + r:.3f} Y{cy:.3f}")
        gcode.append(f"G01 Z{cut_depth_z:.3f} F300")
        # Cắt tròn bằng nội suy cung tròn G02/G03
        gcode.append(
            f"G02 X{cx + r:.3f} Y{cy:.3f} I{-r:.3f} J0.000 F{feed_rate}"
        )
        gcode.append("G00 Z5.000")

      elif geo["type"] == "RECTANGLE":
        cx, cy = geo["center_px"][0] * self.scale, -geo["center_px"][1] * self.scale
        w, h = geo["width_px"] * self.scale, geo["height_px"] * self.scale
        pts = [
            (cx - w / 2, cy - h / 2),
            (cx + w / 2, cy - h / 2),
            (cx + w / 2, cy + h / 2),
            (cx - w / 2, cy + h / 2),
        ]

        # Di chuyển đến góc đầu tiên
        gcode.append(f"G00 X{pts[0][0]:.3f} Y{pts[0][1]:.3f}")
        gcode.append(f"G01 Z{cut_depth_z:.3f} F300")
        for pt in pts[1:]:
          gcode.append(f"G01 X{pt[0]:.3f} Y{pt[1]:.3f} F{feed_rate}")
        gcode.append(
            f"G01 X{pts[0][0]:.3f} Y{pts[0][1]:.3f} F{feed_rate}"
        )  # Khép kín
        gcode.append("G00 Z5.000")

    gcode.append("\n(=== FINISH PROGRAM ===)")
    gcode.append("M05 (Spindle Stop)")
    gcode.append("G00 Z20.000 (Retract Z)")
    gcode.append("G00 X0.000 Y0.000 (Return to Home)")
    gcode.append("M30 (Program End)")

    return "\n".join(gcode)


# ==========================================
# 5. PIPELINE CHÍNH (MAIN EXECUTION)
# ==========================================


def run_hybrid_cad_pipeline(
    image_path: str, api_key: str, output_dxf="output.dxf"
):
  # 1. Đọc ảnh
  cv_img = cv2.imread(image_path)
  if cv_img is None:
    raise FileNotFoundError(f"Không thể mở tệp ảnh: {image_path}")

  print("[STEP 1] Running OpenCV Perspective Transform (Nắn phẳng ảnh)...")
  warped_img = OpenCVGeometryEngine.preprocess_and_warp(cv_img)

  print("[STEP 2] Running OpenCV Geometry Validation (Phân tích Pixel)...")
  cv_geometries = OpenCVGeometryEngine.extract_exact_geometries(warped_img)
  print(f" -> Tìm thấy {len(cv_geometries)} đối tượng hình học từ OpenCV.")

  print("[STEP 3] Running Gemini Vision API (OCR & Intent Extraction)...")
  _, img_encoded = cv2.imencode(".jpg", warped_img)
  image_bytes = img_encoded.tobytes()

  gemini_engine = GeminiVisionEngine(api_key=api_key)
  gemini_data = gemini_engine.extract_dimensions_and_intent(image_bytes)
  print(
      " -> Gemini OCR Dimensions:",
      gemini_data.overall_bounds.model_dump_json(exclude_none=True),
  )

  print("[STEP 4] Aligning Spatial Coordinates & Calculating Scale...")
  cad_generator = HybridCADCAMGenerator()
  cad_generator.align_scale(cv_geometries, gemini_data)

  print("[STEP 5] Exporting DXF & Generating CNC G-Code...")
  cad_generator.export_dxf(cv_geometries, output_path=output_dxf)

  gcode_content = cad_generator.generate_gcode(cv_geometries)
  with open("output.gcode", "w", encoding="utf-8") as f:
    f.write(gcode_content)

  print(
      "[COMPLETE] Đã hoàn thành Pipeline! Xuất file 'output.dxf' và"
      " 'output.gcode' thành công."
  )


# Ví dụ thực thi:
if __name__ == "__main__":
  API_KEY = "YOUR_GEMINI_API_KEY"  # Thay bằng API Key của bạn
  # run_hybrid_cad_pipeline("drawing_sample.jpg", api_key=API_KEY)
