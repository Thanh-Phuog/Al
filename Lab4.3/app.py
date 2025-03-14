import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

model_name = "https://tfhub.dev/google/translate_en_vi/1"
translator = hub.load(model_name)

def translate_text(text):
    input_text = tf.constant([text])
    translate_text = translator(input_text)
    return translate_text.numpy()[0].decode("utf-8")

print(translate_text("Hello, how are you?"))  # Xin chào, bạn khỏe không?