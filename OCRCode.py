import cv2
import numpy as np
import pytesseract
from paddleocr import PaddleOCR
import os

# 📌 Tesseract-OCR 경로 직접 설정
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def preprocess_image(image_path):
    """
    이미지 전처리 (흑백 변환, 노이즈 제거, 이진화)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {image_path}")

    image = cv2.imread(image_path)
    
    # 📌 예외 처리: 이미지가 정상적으로 불러와졌는지 확인
    if image is None:
        raise ValueError(f"cv2.imread()가 None을 반환함. 이미지 파일 경로를 확인하세요: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def extract_text_tesseract(image):
    """
    Tesseract OCR을 사용하여 텍스트 추출
    """
    config = "--oem 3 --psm 6"
    try:
        text = pytesseract.image_to_string(image, config=config, lang='kor+eng')
        return text.strip()
    except Exception as e:
        print(f"[ERROR] Tesseract OCR 오류: {e}")
        return ""

def extract_text_paddleocr(image_path):
    """
    PaddleOCR을 사용하여 텍스트 추출
    """
    ocr = PaddleOCR(lang='korean', use_angle_cls=True)
    
    try:
        result = ocr.ocr(image_path, cls=True)
        text_output = ""
        for line in result:
            for word in line:
                text_output += word[1][0] + " "
        return text_output.strip()
    except Exception as e:
        print(f"[ERROR] PaddleOCR 오류: {e}")
        return ""

def identify_platform(text):
    """
    추출된 텍스트를 기반으로 배달의 민족 vs 쿠팡이츠 판별
    """
    if "배달료 합계" in text or "배달 건수" in text:
        return "배달의 민족"
    elif "미션 보너스" in text or "내 수입" in text:
        return "쿠팡이츠"
    else:
        return "알 수 없음"

def main(image_path):
    """
    이미지에서 데이터 추출 및 플랫폼 판별 실행
    """
    try:
        print("[INFO] 이미지 전처리 중...")
        processed_image = preprocess_image(image_path)
        
        print("[INFO] Tesseract OCR로 텍스트 추출...")
        tesseract_text = extract_text_tesseract(processed_image)
        print("[Tesseract OCR Output]")
        print(tesseract_text)
        
        print("[INFO] PaddleOCR로 텍스트 추출...")
        paddle_text = extract_text_paddleocr(image_path)
        print("[PaddleOCR Output]")
        print(paddle_text)
        
        platform = identify_platform(paddle_text + " " + tesseract_text)
        print(f"[INFO] 인식된 플랫폼: {platform}")
        
        return {
            "platform": platform,
            "text": paddle_text + " " + tesseract_text
        }
    except Exception as e:
        print(f"[ERROR] 프로그램 실행 중 오류 발생: {e}")
        return {"platform": "오류", "text": ""}

if __name__ == "__main__":
    image_path = r"C:\DB\images\KakaoTalk_20220829_090207582_01.jpg"  # 테스트용 이미지 경로
    result = main(image_path)
    print("[FINAL RESULT]", result)
