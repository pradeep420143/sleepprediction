import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import mse

# Constants
IMAGE_SIZE = (64, 64)
MODEL_PATH = "sleep_predictor_model.h5.keras"

# Page setup
st.set_page_config(page_title="Sleep Hours Predictor", layout="centered")
st.title("😴 Sleep Hours Predictor from Face Image")
st.markdown("Upload a face image to predict estimated sleep hours.")

# File uploader
uploaded_file = st.file_uploader("📤 Upload Face Image", type=["jpg", "jpeg", "png"])

# Prediction function
def predict_sleep_hours(image):
    model = load_model(MODEL_PATH, custom_objects={'mse': mse})
    image = image.resize(IMAGE_SIZE).convert('L')
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 64, 64, 1)

    predicted = model.predict(img_array)[0][0]
    return predicted

# Prediction logic
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Face Image", use_column_width=True)

    with st.spinner("Predicting..."):
        prediction = predict_sleep_hours(img)
        st.subheader(f"🧠 Predicted Sleep Hours: `{prediction:.2f}` hours")

        if prediction >= 7.0:
            st.success("✅ Enough Sleep")
        else:
            needed = round(7.0 - prediction, 1)
            st.warning(f"⚠️ Not Enough Sleep\n\n💤 Recommended Additional Sleep: **{needed} hour(s)**")
