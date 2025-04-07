import cv2
import numpy as np
import pytesseract
from paddleocr import PaddleOCR

def show_image(title, image):
    """이미지를 윈도우 창에 표시하고 ESC 키를 누르면 다음 단계로 진행"""
    cv2.imshow(title, image)
    key = cv2.waitKey(0)  # 키 입력 대기
    if key == 27:  # ESC 키 입력 시 창 닫기
        cv2.destroyAllWindows()

def detect_text_box(image_path):
    """
    이미지에서 윤곽선을 찾아서 가장 큰 사각형 박스를 검출 후 해당 영역을 크롭.
    """
    image = cv2.imread(image_path)
    show_image("Original Image", image)  # 1️⃣ 원본 이미지 표시
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 그레이스케일 변환
    show_image("Grayscale Image", gray)  # 2️⃣ 흑백 변환 결과 표시

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # 노이즈 제거 (가우시안 블러)
    show_image("Blurred Image", blurred)  # 3️⃣ 블러 적용 결과 표시

    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)  # 적응형 이진화
    show_image("Binary Image (Thresholding)", binary)  # 4️⃣ 이진화 적용 결과 표시

    edged = cv2.Canny(binary, 50, 150)  # 엣지 감지
    show_image("Canny Edge Detection", edged)  # 5️⃣ 엣지 검출 결과 표시

    # 윤곽선 찾기
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

        # 원본 이미지에 검출된 박스 표시
        box_img = image.copy()
        cv2.rectangle(box_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        show_image("Detected Text Box", box_img)  # 6️⃣ 검출된 박스 표시

        cropped = image[y:y+h, x:x+w]  # ROI 크롭
        show_image("Cropped Image", cropped)  # 7️⃣ 크롭된 영역 표시

        # 크롭된 이미지 저장
        cv2.imwrite("cropped_output.jpg", cropped)
        print(" [INFO] 크롭된 이미지 저장 완료: cropped_output.jpg")

        return cropped
    else:
        print("⚠️ [ERROR] 텍스트 박스를 찾지 못했습니다.")
        return None

def extract_text_from_image(image):
    """
    크롭된 이미지에서 텍스트를 OCR로 추출
    """
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
    show_image("Resized Image for OCR", resized)  # OCR 전처리 후 확대된 이미지 표시

    print(" [INFO] 텍스트 박스 크롭 성공. OCR 적용 중...")
    text_result = extract_text_from_image(resized)

    print("\n [FINAL RESULT]")
    print(text_result)

if __name__ == "__main__":
    image_path = "C:\\DB\\images\\KakaoTalk_20220829_090207582_01.jpg"
    main(image_path)
