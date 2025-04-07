import cv2
import pytesseract
import numpy as np
import re

# Tesseract OCR 경로 설정 (Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def preprocess_image(image_path):
    """ 이미지 전처리 (대비 조정, 이진화, 샤프닝) """
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 대비 증가 (이미지가 흐릴 경우)
    contrast = cv2.convertScaleAbs(gray, alpha=1.8, beta=10)

    # 노이즈 제거 (샤프닝)
    blur = cv2.GaussianBlur(contrast, (3, 3), 0)
    sharpen = cv2.addWeighted(contrast, 1.5, blur, -0.5, 0)

    # Adaptive Thresholding 적용 (더 정밀한 이진화)
    binary = cv2.adaptiveThreshold(sharpen, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    return binary

def extract_text_from_roi(image_path):
    """ 이미지에서 특정 위치(ROI)만 잘라서 OCR 실행 """
    
    # 이미지 불러오기
    image = cv2.imread(image_path)
    h, w, _ = image.shape  # 높이, 너비

    # 관심 영역(ROI) 좌표 설정 (배달 내역 정보)
    roi_coords = {
        "운행일": (w * 0.70, h * 0.06, w * 0.95, h * 0.12),
        "배달 건수": (w * 0.70, h * 0.14, w * 0.95, h * 0.18),
        "이동 거리": (w * 0.70, h * 0.20, w * 0.95, h * 0.24),
        "배달료 합계": (w * 0.70, h * 0.26, w * 0.95, h * 0.30)
    }

    extracted_data = {}

    for key, (x1, y1, x2, y2) in roi_coords.items():
        # ROI 영역 자르기
        roi = image[int(y1):int(y2), int(x1):int(x2)]

        # 전처리 (흑백 변환 및 이진화)
        processed_roi = preprocess_image(image_path)

        # OCR 실행
        text = pytesseract.image_to_string(processed_roi, lang='kor+eng', config='--psm 6')

        # 후처리: 숫자와 한글, 특수 문자만 남기기
        cleaned_text = re.sub(r"[^가-힣0-9\s월일건.km원]", "", text).strip()
        
        extracted_data[key] = cleaned_text

    return extracted_data

if __name__ == "__main__":
    image_path = "C:\\DB\\images\\KakaoTalk_20220829_090207582_01.jpg"
    
    print("\n🔍 [INFO] 특정 위치 기반 텍스트 추출 중...")
    result = extract_text_from_roi(image_path)

    print("\n✅ [FINAL RESULT]")
    for key, value in result.items():
        print(f"{key}: {value}")
