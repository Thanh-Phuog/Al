# MODEL MOBILENETV2

# import tensorflow as tf
# from tensorflow.keras.applications import MobileNetV2

# model = MobileNetV2(weights='imagenet')
# model.save('model.h5')

import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
from flask import Flask, request, jsonify, render_template, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)
model = tf.keras.models.load_model("model.h5")
UPLOAD_FOLDER = "static/uploads/"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file uploaded'})
    filename = secure_filename(file.filename)
    filepath =  os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    image = Image.open(filepath).resize((224, 224))
    image_array = np.expand_dims(np.array(image) / 255.0, axis=0)
    prediction = model.predict(image_array)
    prediction_class = np.argmax(prediction, axis=1)
    
    labels_path = "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"
    labels = tf.keras.utils.get_file('ImageNetLabels.txt', labels_path) 
    with open(labels, "r") as f:
        labels = f.read().splitlines()
    print("Prediction: ", labels[prediction_class[0]])


    # return jsonify({'prediction': prediction_class.tolist()}, {'label': labels[prediction_class[0]]})

    return render_template("index.html", prediction=labels[prediction_class[0]], image_url=url_for('static', filename=f'uploads/{file.filename}'))
if __name__ == '__main__':
    app.run(debug=True)