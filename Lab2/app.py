from flask import Flask, render_template, request, jsonify, url_for
import os
from PIL import Image, ImageOps
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads/"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY")

#Tải mô hình OCR từ Hugging Face
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-printed")

#Xử lý hình ảnh trước khi nhận dạng
def preprocess_image(filepath):
    image = Image.open(filepath).convert("RGB")
    image = ImageOps.invert(image.convert("L")).convert("RGB")
    image = image.resize((1024, 1024))
    return image

def split_image_into_sections(image, rows=5):
    width, height = image.size
    section_height = height // rows
    images = [image.crop((0, i* section_height, width, (i + 1) * section_height)) for i in range(rows)]
    return images

@app.route('/')
def index():
    return render_template('index.html', extracted_text=None, image_url=None)

@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'file' not in request.files:
            return render_template('index.html', extracted_text="No file uploaded", image_url=None)
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', extracted_text="No file uploaded", image_url=None)
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        image = preprocess_image(filepath)

        pixel_values = processor(images=image, return_tensors="pt").pixel_values

        generated_ids = model.generate(pixel_values, num_beams=10, early_stopping=True, max_length=512)
        extracted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return render_template('index.html', extracted_text=extracted_text, image_url=url_for('static', filename=f'uploads/{filename}'))
    except Exception as e:
        return render_template('index.html', extracted_text=f"Error: {str(e)}", image_url=None)
    
if __name__ == '__main__':
    app.run(debug=True)