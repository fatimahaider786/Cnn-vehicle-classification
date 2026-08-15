import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

st.set_page_config(page_title="Vehicle Classifier", page_icon="🚗", layout="centered")

st.title("🚗 Vehicle Classification (CNN)")
st.write("Upload an image to classify the vehicle type using trained TFLite model:")

MODEL_PATH = "vehicle_model.tflite"

# Training Generator wale exact class names
CLASS_NAMES = ['ambulance', 'boat', 'rickshaw', 'scooter', 'tractor']

@st.cache_resource
def load_tflite_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found in repo!")
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

model_loaded = False
try:
    interpreter = load_tflite_model()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model: {e}")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    if not model_loaded:
        st.error("Model is not loaded. Cannot perform prediction.")
    else:
        col1, col2 = st.columns(2)
        image = Image.open(uploaded_file)

        with col1:
            st.subheader("Uploaded Image")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("Prediction")
            with st.spinner("Classifying..."):
                input_shape = input_details[0]['shape']
                height, width = input_shape[1], input_shape[2]

                img = image.convert("RGB").resize((width, height))
                img_array = np.array(img, dtype=np.float32) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                interpreter.set_tensor(input_details[0]['index'], img_array)
                interpreter.invoke()
                predictions = interpreter.get_tensor(output_details[0]['index'])[0]

                predicted_idx = np.argmax(predictions)
                confidence = float(predictions[predicted_idx]) * 100
                label = CLASS_NAMES[predicted_idx]

                st.success(f"Predicted Vehicle: **{label.upper()}**")
                st.metric(label="Confidence", value=f"{confidence:.2f}%")
