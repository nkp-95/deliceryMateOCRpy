import cv2
import numpy as np
import pytesseract
import re
from paddleocr import PaddleOCR, draw_ocr

# PaddleOCR 모델을 한 번만 로드하여 속도 개선
ocr_model = PaddleOCR(lang='korean', det_db_box_thresh=0.3, rec_algorithm='CRNN')

def preprocess_image(image_path):
    """이미지 전처리 (명암 조절, 노이즈 제거, 대비 향상)"""
    image = cv2.imread(image_path)

    # 밝기 & 대비 조절
    alpha, beta = 1.5, 30  # 밝기와 대비 조절 값
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # 흑백 변환 후 선명도 향상
    gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
    sharp = cv2.filter2D(gray, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))

    # 블러 제거 및 이진화
    blur = cv2.GaussianBlur(sharp, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary

def extract_text_tesseract(image):
    """Tesseract OCR을 사용하여 텍스트 추출"""
    config = "--oem 3 --psm 6"
    text = pytesseract.image_to_string(image, config=config, lang='kor+eng')
    return clean_text(text)

def extract_text_paddleocr(image_path):
    """PaddleOCR을 사용하여 텍스트 추출 (신뢰도 필터링 추가)"""
    result = ocr_model.ocr(image_path, cls=True)
    
    detected_text = []
    for line in result:
        for word in line:
            text, confidence = word[1][0], word[1][1]
            if confidence > 0.5:  # 신뢰도 50% 이상인 텍스트만 저장
                detected_text.append(text)

    return clean_text(" ".join(detected_text))

def clean_text(text):
    """OCR 결과에서 불필요한 문자 제거"""
    text = re.sub(r"[^가-힣0-9\s./:]", "", text)  # 한글, 숫자, 일부 특수문자만 남김
    text = re.sub(r"\s+", " ", text).strip()  # 연속된 공백 제거
    return text

def extract_relevant_info(text):
    """OCR 결과에서 원하는 정보만 추출"""
    운행일 = re.search(r"운행일\s*(\d{1,2}월\s*\d{1,2}일)", text)
    배달건수 = re.search(r"배달건\s*/\s*이동거리\s*(\d+건)\s*/\s*([\d.]+km)", text)
    배달료합계 = re.search(r"배달료\s*합계\s*([\d,]+원)", text)

    extracted_data = {
        "운행일": 운행일.group(1) if 운행일 else "정보 없음",
        "배달 건수": 배달건수.group(1) if 배달건수 else "건수 없음",
        "이동 거리": 배달건수.group(2) if 배달건수 else "거리 없음",
        "배달료 합계": 배달료합계.group(1) if 배달료합계 else "배달료 없음"
    }

    return extracted_data

def main(image_path):
    print("\n🔍 [INFO] 이미지 전처리 중...")
    processed_image = preprocess_image(image_path)

    print("\n🔍 [INFO] Tesseract OCR로 텍스트 추출 중...")
    tesseract_text = extract_text_tesseract(processed_image)

    print("\n🔍 [INFO] PaddleOCR로 텍스트 추출 중...")
    paddle_text = extract_text_paddleocr(image_path)

    full_text = tesseract_text + " " + paddle_text
    print("\n📜 [OCR OUTPUT]")
    print(full_text)

    extracted_info = extract_relevant_info(full_text)

    print("\n✅ [FINAL RESULT]")
    for key, value in extracted_info.items():
        print(f"{key}: {value}")

    return extracted_info

if __name__ == "__main__":
    image_path = "C:\\DB\\images\\KakaoTalk_20220829_090207582_01.jpg"
    result = main(image_path)
