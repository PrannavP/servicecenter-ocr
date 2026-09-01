from paddleocr import PaddleOCR
import cv2
import numpy as np

ocr = PaddleOCR(
    use_angle_cls=True,
    lang="en",
    use_gpu=False
)

image_path = "./uploads/image11.jpg"

img = cv2.imread(image_path)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)

gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=10)

kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])

gray = cv2.filter2D(gray, -1, kernel)

gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

processed_path = "./uploads/processed_image.jpg"
cv2.imwrite(processed_path, gray)

result = ocr.ocr(processed_path, cls=True)

if result and result[0]:
    for line in result[0]:
        print(line[1][0])
else:
    print("No text detected")