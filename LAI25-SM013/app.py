import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import requests
from io import BytesIO

# Load the saved model
model = load_model('LAI25-SM013/best_model_revised_98.h5') # Replace 'best_model.keras' with the actual filename

st.title("Muhun manga - LAI25-SM013")

option = st.radio("Pilih metode input gambar:", ("Upload File", "Kamera", "Link Gambar"), horizontal=True)

image = None

if option == "Upload File":
    uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
elif option == "Kamera":
    camera_file = st.camera_input("Ambil gambar dengan kamera")
    if camera_file is not None:
        image = Image.open(camera_file)
elif option == "Link Gambar":
    url = st.text_input("Masukkan URL gambar")
    if url:
        try:
            response = requests.get(url)
            image = Image.open(BytesIO(response.content))
        except Exception as e:
            st.error("Gagal mengambil gambar dari URL.")

if image is not None:
    image_resized = image.resize((150, 150))
    image_array = tf.keras.preprocessing.image.img_to_array(image_resized)
    image_array = tf.expand_dims(image_array, 0)
    image_array = image_array / 255.0

    predictions = model.predict(image_array)
    predicted_class = tf.argmax(predictions, axis=1).numpy()[0]
    class_names = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back', 'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mildew']
    st.write(f"Predicted class: {class_names[predicted_class]}")
    st.image(image, caption="Gambar yang dipilih", use_container_width=True)

    
