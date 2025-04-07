import os
import cv2
import pytesseract
from paddleocr import PaddleOCR

def detect_ui_type(image):
    """
    OCR을 사용해 이미지에서 '배달건 / 이동거리' 텍스트가 있는지 확인하여 UI 유형 분류
    """
    ocr = PaddleOCR(lang='korean')
    result = ocr.ocr(image, cls=False)

    for line in result:
        for box in line:
            text = box[1][0]
            if '배달건 / 이동거리' in text:
                return "with_distance"
    return "without_distance"

def crop_region(image, ui_type):
    h, w, _ = image.shape

    if ui_type == "with_distance":
        # 안드로이드 스타일: 상단 노란 박스
        top = int(h * 0.08)
        bottom = int(h * 0.22)
        left = int(w * 0.05)
        right = int(w * 0.95)
    else:
        # iOS 스타일: 상단 회색 박스
        top = int(h * 0.12)
        bottom = int(h * 0.26)
        left = int(w * 0.05)
        right = int(w * 0.95)

    return image[top:bottom, left:right]

def show_image(title, image):
    if image is None:
        print(f"[ERROR] '{title}' 이미지가 None입니다.")
        return
    resized = cv2.resize(image, (600, 200))
    cv2.imshow(title, resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    folder_path = r"C:\\DB\\images"
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png'))]

    for filename in image_files:
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"[ERROR] 이미지 로드 실패: {filename}")
            continue

        ui_type = detect_ui_type(image)
        cropped = crop_region(image, ui_type)
        print(f"[INFO] {filename} - UI 유형: {ui_type}")
        show_image(f"Cropped - {filename}", cropped)

if __name__ == "__main__":
    main()
