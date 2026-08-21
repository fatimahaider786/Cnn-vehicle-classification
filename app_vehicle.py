import os
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

# 1. Page Configuration
st.set_page_config(
    page_title="CNN Vehicle Classifier", page_icon="🚗", layout="centered"
)

st.title("🚗 Vehicle Classification App")
st.write("Upload an image of a vehicle to classify it.")

# 2. Complete 20 Classes List (Aapke Dataset ke Mutabiq)
CLASS_NAMES = [
    "airplane",
    "ambulance",
    "bicycle",
    "boat",
    "bus",
    "car",
    "fire_truck",
    "helicopter",
    "hovercraft",
    "jet_ski",
    "kayak",
    "motorcycle",
    "rickshaw",
    "scooter",
    "skateboard",
    "tractor",
    "train",
    "unicycle",
    "van",
    "segway",
]


# 3. Load TFLite Model
@st.cache_resource
def load_tflite_interpreter():
  model_path = "vehicle_model.tflite"
  if not os.path.exists(model_path):
    # Root ya subfolder mein tflite file search karein
    for root, dirs, files in os.walk("."):
      for file in files:
        if file.endswith(".tflite"):
          model_path = os.path.join(root, file)
          break

  if os.path.exists(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter
  return None


interpreter = load_tflite_interpreter()

if interpreter is None:
  st.error(
      "❌ `vehicle_model.tflite` file nahi mili! Apni TFLite model file ko"
      " repository mein upload karein."
  )
else:
  # Get TFLite input/output details
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  # 4. File Uploader
  uploaded_file = st.file_uploader(
      "Choose an image...", type=["jpg", "jpeg", "png", "jfif"]
  )

  if uploaded_file is not None:
    col1, col2 = st.columns(2)

    img = Image.open(uploaded_file).convert("RGB")
    with col1:
      st.image(img, caption="Uploaded Image", use_container_width=True)

    with col2:
      with st.spinner("Classifying..."):
        # Image Preprocessing (128x128)
        img_resized = img.resize((128, 128))
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # TFLite Inference
        interpreter.set_tensor(input_details[0]["index"], img_array)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]["index"])[0]

        # Get Prediction Index & Confidence
        predicted_idx = int(np.argmax(predictions))
        confidence = float(np.max(predictions)) * 100

        # Safe Indexing
        predicted_idx = min(predicted_idx, len(CLASS_NAMES) - 1)
        label = CLASS_NAMES[predicted_idx]

        st.success(f"**Predicted Vehicle:** {label.upper()}")
        st.metric(label="Confidence", value=f"{confidence:.2f}%")
