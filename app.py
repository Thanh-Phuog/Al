from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
from models.image_classifier import classify_image
import requests
from dotenv import load_dotenv

app = Flask(__name__)
load_dotenv()

API_TOKEN = os.getenv("HUGGING_FACE_API_KEY")
API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
HEADERS = {"Authorization": f"Bearer {API_TOKEN.strip()}"}

UPLOAD_FOLDER = "static/uploads/"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

products = [
    {
        "name": "Giày búp bê lolita",
        "description": "Giày búp bê lolita nữ cao cấp quai ngọc cài độn đế phong cách tiểu thư",
        "image_url": "static/uploads/lolita.jpg",
        "predictions": "shoe_shop"
        
    },
    {
        "name": "Áo Dài",
        "description": "Áo  dài nữ ",
        "image_url": "static/uploads/ao_dai.jpg",
        "predictions": "áo dài"
    },
    {
        "name": "Điện thoại",
        "description": "Điện thoại cảm ứng",
        "image_url": "static/uploads/smartphone.jpg",
        "predictions": "Điện thoại"
    }
]

@app.route("/")
def home():
    return render_template("index.html", products=products)

# Phân loại ảnh sản phẩm
@app.route("/classify", methods=["POST"])
def upload_image():
    if "file" not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"})

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    
    predictions = classify_image(file_path)
    predictions_text = ", ".join(predictions)  # Chuyển list thành chuỗi

       # Lưu sản phẩm vào danh sách
    product = {
        "name": request.form.get("product_name", "Không có tên"),
        "description": request.form.get("description", "Không có mô tả"),
        "image_url": f"static/uploads/{file.filename}",
        "predictions": predictions_text
    }
    products.insert(0, product)

    return render_template("index.html", predictions=predictions_text, image_url=url_for("static", filename=f"uploads/{file.filename}"), products=products, success=True)

# Phân tích cảm xúc
@app.route("/analyze", methods=["POST"])
def analyze_sentiment():
    data = request.json
    text = data["text"]
    product_name = data.get("product_name", "Không có tên")


    response = requests.post(API_URL, headers=HEADERS, json={"inputs": text})

    if response.status_code != 200:
        return jsonify({"error": f"Error {response.status_code}: {response.text}"}), 400

    sentiment_results = response.json()
    
    return jsonify(sentiment_results)

@app.route("/product/<string:predictions>")
def products_by_prediction(predictions):
    related_products = [p for p in products if p.get("predictions") == predictions]

    if not related_products:
        return "Không có sản phẩm nào thuộc danh mục này", 404

    # Lấy sản phẩm đầu tiên để hiển thị chi tiết
    product = related_products[0]

    return render_template("product.html", product=product, related_products=related_products)


if __name__ == "__main__":
    app.run(debug=True)
