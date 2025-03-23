from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
from models.image_classifier import classify_image
import requests
from dotenv import load_dotenv
import uuid

app = Flask(__name__)
load_dotenv()

API_TOKEN = os.getenv("HUGGING_FACE_API_KEY")
API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
HEADERS = {"Authorization": f"Bearer {API_TOKEN.strip()}"}

UPLOAD_FOLDER = "static/uploads/"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

products = [
    {
        "id": str(uuid.uuid4()),
        "name": "Giày búp bê lolita",
        "description": "Giày búp bê lolita nữ cao cấp quai ngọc cài độn đế phong cách tiểu thư",
        "image_url": "static/uploads/lolita.jpg",
        "predictions": "shoe_shop"
        
    },
     {
        "id": str(uuid.uuid4()),
        "name": "Giày cao gót",
        "description": "Giày cao gót nữ",
        "image_url": "static/uploads/giaycaogot.jpg",
        "predictions": "shoe_shop"
        
    },
    {
        "id": str(uuid.uuid4()),
        "name": "Áo Dài",
        "description": "Áo  dài nữ ",
        "image_url": "static/uploads/ao_dai.jpg",
        "predictions": "áo dài"
    },
    {
        "id": str(uuid.uuid4()),
        "name": "Điện thoại",
        "description": "Điện thoại cảm ứng",
        "image_url": "static/uploads/smartphone.jpg",
        "predictions": "iPod"
    },
     {
        "id": str(uuid.uuid4()),
        "name": "Điện thoại Samsung Galaxy S21 Plus",
        "description": "Điện thoại Samsung Galaxy S21 sở hữu thiết kế tuyệt đẹp.",
        "image_url": "static/uploads/samsung.jpg",
        "predictions": "iPod"
    },
     {
        "id": str(uuid.uuid4()),
        "name": "Điện thoại OPPO A17 ",
        "description": "Thiết kế mỏng nhẹ, mặt lưng phủ giả da thời thượng",
        "image_url": "static/uploads/oppo.jpg",
        "predictions": "iPod"
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
    predictions_text = ", ".join(predictions) 


    product = {
        "id": str(uuid.uuid4()),
        "name": request.form.get("product_name", "Không có tên"),
        "description": request.form.get("description", "Không có mô tả"),
        "image_url": f"static/uploads/{file.filename}",
        "predictions": predictions_text
    }
    products.insert(0, product)

    return render_template("index.html", predictions=predictions_text, image_url=url_for("static", filename=f"uploads/{file.filename}"), products=products)

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

@app.route("/product/<string:id>")
def product_detail(id):
    product = next((p for p in products if p["id"] == id), None)
    
    if not product:
        return "Không tìm thấy sản phẩm", 404
    
    related_products = [p for p in products if p["predictions"] == product["predictions"] and p["id"] != id]


    if not related_products:
        return "Không có sản phẩm nào thuộc danh mục này", 404

    return render_template("product.html", product=product, related_products=related_products)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Lấy PORT từ .env hoặc biến môi trường hệ thống
    app.run(host="0.0.0.0", port=port)
    