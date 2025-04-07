import cv2
import numpy as np
import pytesseract
from paddleocr import PaddleOCR

def show_image(title, image):
    """디버깅용 이미지 출력 함수"""
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def preprocess_image(image_path):
  """이미지 전처리: 흑백 변환, 대비 조정, 노이즈 제거 등"""
  
  # 원본 이미지 불러오기
  image = cv2.imread(image_path)
  show_image("Original Image", image)

  # 1. **흑백 변환**
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  show_image("Grayscale Image", gray)

  # 2. **밝기 보정 및 대비 강화**
  equalized = cv2.equalizeHist(gray)
  show_image("Histogram Equalized", equalized)

  # 3. **Adaptive Thresholding 적용** (로컬 대비 최적화)
  adaptive_thresh = cv2.adaptiveThreshold(equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 11, 2)
  show_image("Adaptive Thresholding", adaptive_thresh)

  # 4. **Sharpening (선명화) 적용**
  kernel = np.array([[0, -1, 0],
                      [-1,  5, -1],
                      [0, -1, 0]])
  sharpened = cv2.filter2D(adaptive_thresh, -1, kernel)
  show_image("Sharpened Image", sharpened)

  # 5. **엣지 검출 (Canny)**
  edged = cv2.Canny(sharpened, 50, 200)
  show_image("Edge Detection (Canny)", edged)

  return image, edged

def detect_text_box(image, edged):
  """사각형 박스 검출하여 OCR 영역 크롭"""
  
  contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  best_box = None
  max_area = 0

  for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
    if len(approx) == 4:
      x, y, w, h = cv2.boundingRect(approx)
      aspect_ratio = w / float(h)

      # 특정 비율과 크기를 만족하는 경우 선택
      if 1.5 < aspect_ratio < 3 and w * h > 5000:  
        if w * h > max_area:
          best_box = (x, y, w, h)
          max_area = w * h

  if best_box:
    x, y, w, h = best_box
    cropped = image[y:y+h, x:x+w]
    show_image("Cropped ROI", cropped)
    cv2.imwrite("cropped_output.jpg", cropped)
    print("✅ [INFO] 크롭된 이미지 저장 완료: cropped_output.jpg")
    return cropped
  else:
    print("⚠️ [ERROR] 텍스트 박스를 찾지 못했습니다.")
    return None

def extract_text_from_image(image):
  """OCR 실행"""
  
  config = "--oem 3 --psm 6"
  text_tesseract = pytesseract.image_to_string(image, config=config, lang='kor+eng')

  # PaddleOCR 한글 & 영어 OCR 따로 수행
  ocr_korean = PaddleOCR(lang="korean")
  result_korean = ocr_korean.ocr(image, cls=True)

  ocr_english = PaddleOCR(lang="en")
  result_english = ocr_english.ocr(image, cls=True)

  # OCR 결과 예외 처리
  def extract_text(result):
    return " ".join([word[1][0] for line in result if line for word in line]) if result and result != [[]] else ""

  text_paddle_korean = extract_text(result_korean)
  text_paddle_english = extract_text(result_english)

  # 두 OCR 결과 합치기
  return text_tesseract.strip() + " " + text_paddle_korean.strip() + " " + text_paddle_english.strip()


def main(image_path):
  print("🔍 [INFO] 특정 위치 기반 텍스트 추출 중...")

  image, edged = preprocess_image(image_path)

  cropped_image = detect_text_box(image, edged)
  if cropped_image is None:
    return None

  # 크롭된 이미지 해상도 증가 후 OCR 실행
  resized = cv2.resize(cropped_image, (cropped_image.shape[1] * 2, cropped_image.shape[0] * 2))
  show_image("Resized Image for OCR", resized)

  print("✅ [INFO] 텍스트 박스 크롭 성공. OCR 적용 중...")
  text_result = extract_text_from_image(resized)

  print("\n✅ [FINAL RESULT]")
  print(text_result)

if __name__ == "__main__":
  image_path = "C:\\DB\\images\\KakaoTalk_20220829_090207582_01.jpg"
  main(image_path)
