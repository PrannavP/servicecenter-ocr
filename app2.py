from flask import Flask, request, jsonify
import pytesseract
from PIL import Image
import os

import cv2
import numpy as np
# from paddleocr import PaddleOCR
from helper.extracted_key_value import parse_ocr_result

from helper.extracted_key_value_v2 import extract_key_value_pairs

from helper.llm_mapper import map_job_card

import boto3
from dotenv import load_dotenv

load_dotenv()

# use the aws textract service.
textract = boto3.client(
    "textract",
    region_name=os.getenv("AWS_REGION")
)

app = Flask(__name__)

# Exact Textract label -> schema key (cheap first pass before the AI mapper).
_EXACT_LABELS = {
    "customer name": "customer_name",
    "customer address": "customer_address",
    "customer number": "customer_number",
    "vehicle type": "vehicle_type",
    "vehicle chasis number": "vehicle_chasis_number",
    "vehicle registration number": "vehicle_registration_number",
    "date time": "date_time",
    "current odometer reading": "current_odometer_reading",
    "fuel level": "fuel_level",
    "service type": "service_type",
    "service advisor": "service_advisor",
    "helmet kept": "helmet_kept",
    "expected delivery date time": "expected_delivery_date_time",
}

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Tesseract (old endpoint)
# pytesseract.pytesseract.tesseract_cmd = r"F:\Softwares\PyTesseract\tesseract.exe"

# PaddleOCR (NEW endpoint) - initialize ONCE
# ocr = PaddleOCR(
#     use_angle_cls=True,
#     lang="en",
#     use_gpu=False
# )


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
# @app.route("/upload_v2", methods=["POST"])
# def upload_file_v2():
#     if "file" not in request.files:
#         return jsonify({"error": "No file provided"}), 400

#     file = request.files["file"]

#     if file.filename == "":
#         return jsonify({"error": "Empty filename"}), 400

#     file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
#     file.save(file_path)

#     try:
#         # -------------------------
#         # PREPROCESSING
#         # -------------------------
#         img = cv2.imread(file_path)

#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#         # denoise
#         gray = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)

#         # contrast
#         gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=10)

#         # sharpen
#         kernel = np.array([[0, -1, 0],
#                             [-1, 5, -1],
#                             [0, -1, 0]])
#         gray = cv2.filter2D(gray, -1, kernel)

#         # upscale
#         gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

#         processed_path = os.path.join(
#             app.config["UPLOAD_FOLDER"],
#             "processed_" + file.filename
#         )

#         cv2.imwrite(processed_path, gray)

#         # -------------------------
#         # PADDLEOCR
#         # -------------------------
#         result = ocr.ocr(processed_path, cls=True)

#         extracted_text = []

#         if result and result[0]:
#             for line in result[0]:
#                 extracted_text.append(line[1][0])

#         parsed_data = parse_ocr_result(extracted_text)

#         return jsonify({
#             "extracted_text": "\n".join(extracted_text),
#             "filename": file.filename,
#             "data": parsed_data
#         })

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


# new endpoint for uploading service center job card, this endpoint calls the aws textract service and then extracts the text from image
# then returns the data in key value pairs.
@app.route("/upload_v3", methods=["POST"])
def upload_file_v3():

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    file_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )

    file.save(file_path)

    try:

        with open(file_path, "rb") as document:
            image = document.read()

        response = textract.analyze_document(
            Document={
                "Bytes": image
            },
            FeatureTypes=[
                "FORMS"
            ]
        )

        extracted_text = []

        for block in response["Blocks"]:

            if block["BlockType"] == "LINE":
                extracted_text.append(block["Text"])

        kv_pairs = extract_key_value_pairs(response)

        # Rule-based parse first — used as a safety net for the AI mapper.
        parsed_data = parse_ocr_result(extracted_text)

        # Cheap rule-based merge of Textract's exact-label matches.
        for key, value in kv_pairs.items():
            json_key = _EXACT_LABELS.get(key.lower().strip())
            if json_key:
                parsed_data[json_key] = value

        # AI semantic mapping: understands label variations & normalises values,
        # with per-field confidence. Falls back to parsed_data if no model access.
        mapped = map_job_card(kv_pairs, extracted_text, parsed_data)

        return jsonify({
            "filename": file.filename,
            "textract_key_value": kv_pairs,
            "extracted_text": extracted_text,
            "data": mapped["data"],
            "confidence": mapped["confidence"],
            "mapping_source": mapped["mapping_source"],
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)