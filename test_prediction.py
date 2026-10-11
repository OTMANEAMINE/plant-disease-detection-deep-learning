import tensorflow as tf
import numpy as np
import os

# ============================================================
# LOAD MODEL
# ============================================================
model = tf.keras.models.load_model("modele_plantes.keras")

# ============================================================
# CLASSES (نفس الترتيب)
# ============================================================
data_dir = "plantdisease/PlantVillage"
class_names = sorted(os.listdir(data_dir))

# ============================================================
# IMAGE
# ============================================================
img_path = "test_images/leaf_potato.jpg"   # 👈 بدلها باسم الصورة ديالك
img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
img_array = tf.keras.utils.img_to_array(img)
img_array = img_array / 255.0   # 🔥 مهم بزاف
img_array = tf.expand_dims(img_array, 0)

# ============================================================
# PREDICTION
# ============================================================
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

predicted_class = class_names[np.argmax(score)]
confidence = np.max(score)

# ============================================================
# RESULT
# ============================================================
print("\n🌿 RESULTAT :")
print(f"👉 Classe prédite : {predicted_class}")
print(f"🔥 Confiance : {confidence*100:.2f}%")