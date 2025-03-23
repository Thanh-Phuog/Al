from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

model = MobileNetV2(weights='imagenet')

def classify_image(image_path):
    img = Image.open(image_path).resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=1)[0]

    result = [label for (_, label, conf) in decoded_predictions]  # Chỉ lấy class name
    return result