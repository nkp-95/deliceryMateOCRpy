import cv2
import numpy as np
import pytesseract
from paddleocr import PaddleOCR

def detect_text_box(image_path):
    
    # 이미지에서 윤곽선을 찾아서 가장 큰 사각형 박스를 검출 후 해당 영역을 크롭.
    
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 흑백 변환
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # 노이즈 제거
    edged = cv2.Canny(blurred, 50, 150)  # 엣지 감지

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 가장 큰 사각형 찾기
    largest_box = None
    max_area = 0
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)  # 다각형 근사화
        if len(approx) == 4:  # 사각형일 경우만 처리
            x, y, w, h = cv2.boundingRect(approx)
            area = w * h
            if area > max_area:
                largest_box = (x, y, w, h)
                max_area = area
    
    if largest_box:
        x, y, w, h = largest_box
        cropped = image[y:y+h, x:x+w]  # ROI 크롭
        
        #  크롭된 이미지 저장 후 확인
        cv2.imwrite("cropped_output.jpg", cropped)
        print(" [INFO] 크롭된 이미지 저장 완료: cropped_output.jpg")
        
        return cropped
    else:
        print("⚠️ [ERROR] 텍스트 박스를 찾지 못했습니다.")
        return None

def extract_text_from_image(image):
    
    # 크롭된 이미지에서 텍스트를 OCR로 추출
    
    config = "--oem 3 --psm 6"
    text_tesseract = pytesseract.image_to_string(image, config=config, lang='kor+eng')
    
    ocr = PaddleOCR(lang="korean")
    result_paddle = ocr.ocr(image, cls=True)
    
    # OCR 결과 예외 처리
    if not result_paddle or result_paddle == [[]]:  # 빈 리스트 반환 방지
        text_paddle = ""
    else:
        text_paddle = " ".join([word[1][0] for line in result_paddle if line for word in line])
    
    # 두 OCR 결과 합치기
    return text_tesseract.strip() + " " + text_paddle.strip()

def main(image_path):
    print(" [INFO] 특정 위치 기반 텍스트 추출 중...")

    cropped_image = detect_text_box(image_path)
    if cropped_image is None:
        return None
    
    # 크롭된 이미지 해상도 증가 후 OCR 실행
    resized = cv2.resize(cropped_image, (cropped_image.shape[1] * 2, cropped_image.shape[0] * 2))
    
    print(" [INFO] 텍스트 박스 크롭 성공. OCR 적용 중...")
    text_result = extract_text_from_image(resized)

    print("\n [FINAL RESULT]")
    print(text_result)

if __name__ == "__main__":
    image_path = "C:\\DB\\images\\KakaoTalk_20220829_090207582_01.jpg"
    main(image_path)
