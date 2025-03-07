import requests
from dotenv import load_dotenv
import os
from flask import Flask, request, jsonify, render_template
 
app = Flask(__name__)
load_dotenv()

API_TOKEN = os.getenv("HUGGING_FACE_API_KEY")
print("API Key:", API_TOKEN)  # Kiểm tra API key có bị None không

API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
HEADERS = {"Authorization": f"Bearer {API_TOKEN.strip()}"}

# def analyze_sentiment(text):
#     response = requests.post(API_URL, headers=HEADERS, json={"inputs": text})
    
#     if response.status_code != 200:
#         return f"Error {response.status_code}: {response.text}"

#     return response.json()

# print(analyze_sentiment("I love this product!"))
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze_sentiment():
    data = request.json
    text = data["text"]

    response = requests.post(API_URL, headers=HEADERS, json={"inputs": text})

    if response.status_code != 200:
        return jsonify({"error": f"Error {response.status_code}: {response.text}"}), 400

    return jsonify(response.json())

if __name__ == "__main__":
    app.run(debug=True)