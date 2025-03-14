from flask import Flask, request, jsonify
import pytesseract
import numpy as np
import cv2
from PIL import Image
import io

app = Flask(__name__)

pytesseract.pytesseract.tesseract_cmd = r"C:\cntt\HK4\TesseractOCR\tesseract.exe"

def preoprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


@app.route('/ocr', methods=['POST'])
def recoognize_text():
    if "file" not in request.files:
        return jsonify({"error": "no file uploaded"}), 400
    
    file = request.files["file"]
    image = Image.open(io.BytesIO(file.read()))
    image = np.array(image)

    processed_image = preoprocess_image(image)

    text = pytesseract.image_to_string(processed_image)

    return jsonify({"recognized_text": text})

if __name__ == "__main__":
    app.run(debug=True)