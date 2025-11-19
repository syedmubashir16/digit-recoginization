import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load model
model = tf.keras.models.load_model("digit_model.h5")

st.title("🖊️ Handwritten Digit Recognition")
st.write("Upload an image of a handwritten digit (0–9) and I'll predict it!")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((28, 28))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    st.subheader(f"✅ Predicted Digit: {predicted_digit}")
    st.write(f"Confidence: {confidence:.2f}%")