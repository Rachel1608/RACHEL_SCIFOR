import streamlit as st
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

import pickle

with open('C:/Users/HP/Desktop/Streamlit/Rachel1608/Streamlit/model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("Mask Detection App")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Processing...")

    # Preprocess the image to match the model's input requirements
    image = image.resize((227, 227))  # Resize image to 227x227
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Normalize pixel values

    prediction = model.predict(image)
    mask_prob = prediction[0][0]

    if mask_prob < 0.5:
       st.write("Prediction: Mask")
    else:
       st.write("Prediction: No Mask")
