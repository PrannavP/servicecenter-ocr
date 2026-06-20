from flask import Flask, request, jsonify
import pytesseract
from PIL import Image
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ONLY FOR WINDOWS
pytesseract.pytesseract.tesseract_cmd = r"F:\Softwares\PyTesseract\tesseract.exe"

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
        # Open image
        img = Image.open(file_path)

        # OCR extraction
        text = pytesseract.image_to_string(img)

        return jsonify({
            "filename": file.filename,
            "extracted_text": text
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)