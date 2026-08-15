import os
import glob
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

# 1. Page Configuration
st.set_page_config(
    page_title="CNN Vehicle Classifier", page_icon="🚗", layout="centered"
)

st.title("🚗 Vehicle Classification App")
st.write(
    "Upload an image of a vehicle to classify it into one of the categories."
)

# 2. Class Names Definition
CLASS_NAMES = ["ambulance", "boat", "rickshaw", "scooter", "tractor"]


# 3. Automatic Model Finder & Loader
@st.cache_resource
def load_trained_model():
  # Pehle root folder aur phir saare subfolders mein .h5 / .keras files dhoondein
  model_files = glob.glob("**/*.h5", recursive=True) + glob.glob(
      "**/*.keras", recursive=True
  )

  if model_files:
    # Sab se pehli milli hui model file load karein
    st.write(f"🔍 Loading model file: `{model_files[0]}`")
    return tf.keras.models.load_model(model_files[0])

  return None


model = load_trained_model()

if model is None:
  st.error(
      "❌ Model file (.h5 / .keras) nahi mili! Baraye karam apni model file ko"
      " GitHub repository mein upload karein."
  )
else:
  # 4. File Uploader
  uploaded_file = st.file_uploader(
      "Choose an image...", type=["jpg", "jpeg", "png", "jfif"]
  )

  if uploaded_file is not None:
    # Display Uploaded Image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # 5. Image Preprocessing (128x128 & Normalization)
    img_resized = img.resize((128, 128))
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # 6. Prediction Logic
    with st.spinner("Classifying image..."):
      predictions = model.predict(img_array)

      if predictions.shape[-1] == 1:
        # Binary Classification
        prob = float(predictions[0][0])
        predicted_idx = 1 if prob >= 0.5 else 0
        confidence = (prob if prob >= 0.5 else 1 - prob) * 100
      else:
        # Multi-class Classification
        predicted_idx = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0])) * 100

      # SAFE INDEXING (IndexError Prevention)
      predicted_idx = min(predicted_idx, len(CLASS_NAMES) - 1)
      label = CLASS_NAMES[predicted_idx]

    # 7. Display Results
    st.markdown("---")
    st.subheader("🎯 Prediction Result")
    st.success(f"**Predicted Class:** {label.upper()}")
    st.info(f"**Confidence Score:** {confidence:.2f}%")
