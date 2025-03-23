import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

#Tải mô hình pre-trained từ TensorFlow Hub 
model_name = "https://tfhub.dev/google/translate_en_vi/1"
translator = hub.load(model_name)

def translate_text(text):
    input_text = tf.constant([text])
    translate_text = translator(input_text)
    return translate_text.numpy()[0].decode("utf-8")

print(translate_text("Hello, how are you?"))  # Xin chào, bạn khỏe không?

# Xây dựng API Flask để nhận yêu cầu dịch văn bản 
import tensorflow_text

model_name = "https://tfhub.dev/google/translate_en_vi/1"
translator = tf.saved_model.load(model_name)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/translate", methods=["POST"])
def translate():
    data = request.json
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    translate_text = translator(tf.constant([text]))[0].numpy()[0].decode("utf-8")
    return jsonify({"text": translate_text})

if __name__ == "__main__":
    app.run(debug=True)