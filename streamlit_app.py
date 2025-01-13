import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

st.title("AI Mengklasifikasi Berbagai Macam Buah")
st.write(
    "Projek ini bertujuan untuk mengembangkan ML Model untuk tugas klasifikasi buah buahan."
)

st.header("Project Overview")
st.write("Industri perkebunan buah buahan merupakan industri yang cukup besar di wilayah tropis. Proses penyortiran berbagai macam buah-buahan diperlukan sebelum memasuki pemrosesan lebih lanjut.")
st.write("Sistem klasifikasi buah-buahan ini dapat bermanfaat untuk proses penyortiran dari berbagai macam buah-buahan.")

st.header("Model Overview")
st.image("assets/img/model.png")
st.write('Hyper parameter yang digunakan adalah: batch size = 32, input size = (128,128), rescale = 1/255, dan epoch = 30')
st.image("assets/img/trainvalgraph.png")
st.write("Hasil ujicoba menghasilkan: Test Loss: 0.5613 & Test Accuracy: 0.7246")
st.image("assets/img/confussionmatrix.png")

st.header("Showcase")
img_file_buffer = st.camera_input("Take a picture")
uploaded_file = st.file_uploader("Or, Choose a file")

loaded_model = tf.keras.models.load_model('assets/model/model.h5')
input_shape = (100,100)
label = ['Apple Braeburn', 'Apple Granny Smith', 'Apricot', 'Avocado', 'Banana', 'Blueberry', 'Cactus fruit', 'Cantaloupe', 'Cherry', 'Clementine', 'Corn', 'Cucumber Ripe', 'Grape Blue', 'Kiwi', 'Lemon', 'Limes', 'Mango', 'Onion White', 'Orange', 'Papaya', 'Passion Fruit', 'Peach', 'Pear', 'Pepper Green', 'Pepper Red', 'Pineapple', 'Plum', 'Pomegranate', 'Potato Red', 'Raspberry', 'Strawberry', 'Tomato', 'Watermelon']

if img_file_buffer is not None or uploaded_file is not None:
    if uploaded_file is None:
        input_file = img_file_buffer
    else: input_file = uploaded_file
    # To read image file buffer as a PIL Image:
    img = Image.open(input_file)
    img_array = np.array(img)

    image = Image.open(input_file)
    # Resize img to fit in model input shape
    image = image.resize(input_shape)
    # Convert the image to a NumPy array
    image_array = np.array(image)
    image_array = image_array[:, :, :3]  # Keep only the first 3 channels (RGB)
    # Normalize the pixel values
    image_array = image_array / 255.0
    # Expand dimensions to match the input shape dimension of the model
    input_image = np.expand_dims(image_array, axis=0)

    predictions = loaded_model.predict(input_image).tolist()[0]
    predicted_label = np.argmax(predictions, axis=-1)
    
    st.write("Result:")
    st.write(f"{100*predictions[predicted_label]:.2f}%",label[predicted_label])

    st.write(pd.DataFrame({
        'Label': label,
        'Result': predictions,
    }))