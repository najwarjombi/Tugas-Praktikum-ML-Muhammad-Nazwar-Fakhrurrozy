import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

class_labels = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "model.keras")
    return tf.keras.models.load_model(model_path)

def preprocess_image(image):
    img_gray = ImageOps.grayscale(image)
    img_resized = img_gray.resize((28, 28))
    img_array = np.array(img_resized, dtype=np.float32)
    if np.mean(img_array) > 127:
        img_array = 255.0 - img_array
    img_norm = img_array / 255.0
    return img_norm.reshape(1, 28, 28, 1)

model = load_model()

st.title("Klasifikasi Gambar Pakaian Fashion MNIST")
uploaded_file = st.file_uploader("Unggah gambar pakaian", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diunggah", width=250)
    processed_img = preprocess_image(image)
    predictions = model.predict(processed_img, verbose=0)
    pred_idx = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0])) * 100
    st.subheader(f"Prediksi: {class_labels[pred_idx]}")
    st.write(f"Tingkat keyakinan: {confidence:.2f}%")