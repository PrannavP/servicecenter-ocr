from flask import Flask, request, jsonify
import pytesseract
from PIL import Image
import os

import cv2
import numpy as np
from paddleocr import PaddleOCR
from helper.extracted_key_value import parse_ocr_result

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Tesseract (old endpoint)
pytesseract.pytesseract.tesseract_cmd = r"F:\Softwares\PyTesseract\tesseract.exe"

# PaddleOCR (NEW endpoint) - initialize ONCE
ocr = PaddleOCR(
    use_angle_cls=True,
    lang="en",
    use_gpu=False
)


@app.route("/")
def home():
    return "OK"


# -------------------------
# OLD ENDPOINT (Tesseract)
# -------------------------
@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    try:
        img = Image.open(file_path)
        text = pytesseract.image_to_string(img)

        return jsonify({
            "filename": file.filename,
            "extracted_text": text
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------
# NEW ENDPOINT (PaddleOCR + preprocessing)
# -------------------------
@app.route("/upload_v2", methods=["POST"])
def upload_file_v2():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    try:
        # -------------------------
        # PREPROCESSING
        # -------------------------
        img = cv2.imread(file_path)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # denoise
        gray = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)

        # contrast
        gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=10)

        # sharpen
        kernel = np.array([[0, -1, 0],
                            [-1, 5, -1],
                            [0, -1, 0]])
        gray = cv2.filter2D(gray, -1, kernel)

        # upscale
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        processed_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            "processed_" + file.filename
        )

        cv2.imwrite(processed_path, gray)

        # -------------------------
        # PADDLEOCR
        # -------------------------
        result = ocr.ocr(processed_path, cls=True)

        extracted_text = []

        if result and result[0]:
            for line in result[0]:
                extracted_text.append(line[1][0])

        parsed_data = parse_ocr_result(extracted_text)

        return jsonify({
            "extracted_text": "\n".join(extracted_text),
            "filename": file.filename,
            "data": parsed_data
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)