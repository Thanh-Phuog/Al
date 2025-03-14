from flask import Flask, request, jsonify, render_template
import requests
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)

API_TOKEN = os.getenv("HUGGING_FACE_API_KEY")
print("API Key:", API_TOKEN)  # Kiểm tra API key có bị None không

API_URL = "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-en-vi"
HEADERS = {"Authorization": f"Bearer {API_TOKEN.strip()}"}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/translate", methods=["POST"])
def translate_text():
    data = request.json
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    response = requests.post(API_URL, headers=HEADERS, json={"inputs": text})
    return jsonify(response.json())

if __name__ == "__main__":
    app.run(debug=True)