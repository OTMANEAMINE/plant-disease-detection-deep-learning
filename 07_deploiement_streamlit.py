import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# ============================================================
# LOAD MODEL
# ============================================================
model = tf.keras.models.load_model("modele_plantes.keras")

# ============================================================
# CLASSES
# ============================================================
data_dir = "plantdisease/PlantVillage"
class_names = sorted(os.listdir(data_dir))

# ============================================================
# TITLE
# ============================================================
st.title("🌿 Détection des maladies des plantes")
st.write("Upload une image de feuille pour détecter la maladie")

# ============================================================
# UPLOAD IMAGE
# ============================================================
uploaded_file = st.file_uploader("Choisir une image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # عرض الصورة
    image = Image.open(uploaded_file)
    st.image(image, caption="Image uploadée", use_column_width=True)

    # ========================================================
    # PREPROCESSING
    # ========================================================
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ========================================================
    # PREDICTION
    # ========================================================
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    predicted_class = class_names[np.argmax(score)]
    confidence = np.max(score)

    # ========================================================
    # RESULT
    # ========================================================
    st.success(f"🌱 Maladie détectée : {predicted_class}")
    st.info(f"🔍 Confiance : {confidence*100:.2f}%")