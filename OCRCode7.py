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

    # 1. **흑백 변환**
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. **밝기 보정 및 대비 강화**
    equalized = cv2.equalizeHist(gray)

    # 3. **Adaptive Thresholding 적용** (로컬 대비 최적화)
    adaptive_thresh = cv2.adaptiveThreshold(equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY, 11, 2)

    # 4. **Sharpening (선명화) 적용**
    kernel = np.array([[0, -1, 0],
                        [-1,  5, -1],
                        [0, -1, 0]])
    sharpened = cv2.filter2D(adaptive_thresh, -1, kernel)

    # 5. **엣지 검출 (Canny)**
    edged = cv2.Canny(sharpened, 50, 200)

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

        # 🔹 크롭 영역 조정 (상단 부분 포함) 🔹
        y = max(0, y - int(h * 0.3))  # 상단을 30% 더 포함하도록 조정
        
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

    ocr = PaddleOCR(lang="korean")
    result_paddle = ocr.ocr(image, cls=True)

    extracted_texts = []
    
    if result_paddle and result_paddle != [[]]:
        for line in result_paddle:
            for word in line:
                extracted_texts.append(word[1][0])

    # 🔹 OCR 결과 출력 🔹
    print("📌 [DEBUG] OCR 추출 텍스트:", extracted_texts)

    return extracted_texts

def parse_ocr_results(ocr_results):
    """OCR 결과에서 날짜, 배달 건수, 이동 거리, 수익 정보 추출"""

    date, delivery_count, distance, earnings = None, None, None, None

    for text in ocr_results:
        if "월" in text and "일" in text:
            date = text.strip()
        elif "/" in text:  # 🔹 배달 건수와 이동 거리 구분 🔹
            split_values = text.split("/")
            if len(split_values) == 2:
                delivery_count = split_values[0].strip()
                distance = split_values[1].strip()
        elif "원" in text:
            earnings = text.strip()

    return date, delivery_count, distance, earnings

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
    ocr_results = extract_text_from_image(resized)

    # 🔹 OCR 결과 분류 🔹
    date, delivery_count, distance, earnings = parse_ocr_results(ocr_results)

    # ✅ 최종 결과 출력 ✅
    print("\n✅ [FINAL RESULT]")
    print("📌 배달의 민족")
    print("📅 날짜:", date)
    print("📦 배달건수:", delivery_count)
    print("🛵 이동거리:", distance)
    print("💰 수익:", earnings)

if __name__ == "__main__":
    image_path = "C:\\DB\\images\\KakaoTalk_20220829_090207582_01.jpg"
    main(image_path)
