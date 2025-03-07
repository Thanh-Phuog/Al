from flask import Flask, request, jsonify, render_template, url_for
import os
from werkzeug.utils import secure_filename
from PIL import Image
from models.models import load_classification_model, classify_image
from dotenv import load_dotenv

load_dotenv()

feature_extractor, model = load_classification_model()

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/classify', methods=['POST'])
def classify():
    if 'file' not in request.files:
        return jsonify({"error":"No file uploaded"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error":"No file uploaded"}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    predicted_label = classify_image(filepath, feature_extractor, model)
    # return jsonify({"predicted_label": predicted_label})
    return render_template("index.html", prediction=predicted_label, image_url=url_for('static', filename=f'uploads/{filename}'))
    

if __name__ == '__main__':
    app.run(debug=True)