from flask import Flask, request, jsonify
import pytesseract
from PIL import Image
import os

import cv2
import numpy as np
from helper.extracted_key_value import parse_ocr_result

from helper.extracted_key_value_v2 import extract_key_value_pairs

from helper.llm_mapper import map_job_card

import boto3
from dotenv import load_dotenv

load_dotenv()

textract = boto3.client(
    "textract",
    region_name=os.getenv("AWS_REGION")
)

app = Flask(__name__)

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

@app.route("/")
def home():
    return "OK"

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

        parsed_data = parse_ocr_result(extracted_text)

        for key, value in kv_pairs.items():
            json_key = _EXACT_LABELS.get(key.lower().strip())
            if json_key:
                parsed_data[json_key] = value

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