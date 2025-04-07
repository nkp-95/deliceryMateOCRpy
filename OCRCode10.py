import cv2
import numpy as np
import pytesseract
from paddleocr import PaddleOCR
import re

def show_image(title, image):
    """디버깅용 이미지 출력 함수"""
    if image is None:
        print(f"⚠️ [ERROR] '{title}' 이미지가 None입니다. 출력 불가.")
        return
    resized_image = cv2.resize(image, (600, 400))
    cv2.imshow(title, resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def preprocess_image(image_path):
    """이미지 전처리: 흑백 변환, 대비 조정, 노이즈 제거 등"""
    image = cv2.imread(image_path)

    if image is None:
        print("⚠️ [ERROR] 이미지 로드 실패")
        return None, None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    adaptive_thresh = cv2.adaptiveThreshold(equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    edged = cv2.Canny(adaptive_thresh, 50, 200)

    return image, edged

def detect_text_box(image, edged):
    """사각형 박스 검출하여 OCR 영역 크롭"""
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_box = None
    max_area = 0

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        x, y, w, h = cv2.boundingRect(approx)

        if w * h > max_area:
            best_box = (x, y, w, h)
            max_area = w * h

    if best_box:
        x, y, w, h = best_box
        x = max(0, x - 110)  # 🔹 왼쪽 여백 110px 추가
        y = max(0, y - 130)  # 🔹 위쪽 여백 130px 추가
        cropped = image[y:y+h, x:x+w]
        show_image("Cropped ROI", cropped)
        return cropped
    else:
        print("⚠️ [ERROR] 텍스트 박스를 찾지 못했습니다.")
        return None

def extract_text_from_image(image):
    """OCR 실행"""
    if image is None:
        print("⚠️ [ERROR] OCR 수행할 이미지가 None입니다.")
        return None

    config = "--oem 3 --psm 3"
    text_tesseract = pytesseract.image_to_string(image, config=config, lang='kor+eng')

    ocr = PaddleOCR(lang="korean")
    result_paddle = ocr.ocr(image, cls=True)

    extracted_texts = []

    if result_paddle and result_paddle != [[]]:
        for line in result_paddle:
            for word in line:
                extracted_texts.append(word[1][0])

    extracted_text = text_tesseract.strip() + " " + " ".join(extracted_texts).strip()
    
    # 🔹 OCR 결과 출력 🔹
    print("📌 [DEBUG] OCR 추출 텍스트:", extracted_text.split())

    return extracted_text.split()

def parse_ocr_results(ocr_results):
    """OCR 결과에서 날짜, 배달 건수, 이동 거리, 수익 정보 추출"""
    date, delivery_count, distance, earnings = None, None, None, None
    month, day = None, None

    for text in ocr_results:
        text = text.strip()

        if re.match(r"^\d+건/$", text):
            delivery_count = text[:-2]  # '/' 제거 후 저장
            continue

        if re.match(r"^/\d+[\.,]?\d*km$", text):
            distance = text[1:]  # '/' 제거 후 저장
            continue
        
        if "월" in text and len(text) <= 4:
            month = text.strip()
        elif "일" in text and len(text) <= 4:
            day = text.strip()
        elif "건" in text:
            delivery_count = text.replace("건", "").strip()
        elif "km" in text and distance is None:
            distance = text.strip()
        elif "원" in text:
            earnings = text.replace(",", "").strip()

    if month and day:
        date = f"{month} {day}"

    if earnings and earnings.isdigit():
        earnings = f"{int(earnings):,}원"

    return date, delivery_count, distance, earnings

def main(image_path):
    print("🔍 [INFO] 특정 위치 기반 텍스트 추출 중...")

    image, edged = preprocess_image(image_path)
    if image is None:
        return

    cropped_image = detect_text_box(image, edged)
    if cropped_image is None:
        return

    resized = cv2.resize(cropped_image, (cropped_image.shape[1] * 2, cropped_image.shape[0] * 2))
    show_image("Resized Image for OCR", resized)

    ocr_results = extract_text_from_image(resized)
    if ocr_results is None:
        return

    date, delivery_count, distance, earnings = parse_ocr_results(ocr_results)

    # ✅ 최종 결과 출력 ✅
    print("\n✅ [FINAL RESULT]")
    print("📌 배달의 민족")
    print("📅 날짜:", date)
    print("📦 배달건수:", delivery_count)
    print("🚴 이동거리:", distance)
    print("💰 수익:", earnings)

if __name__ == "__main__":
    image_path = "C:\\DB\\images\\KakaoTalk_20220829_090207582_01.jpg"
    main(image_path)
