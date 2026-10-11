import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
import random

# ============================================================
# PARAMÈTRES
# ============================================================
img_size = (224, 224)

# ============================================================
# NORMALISATION
# ============================================================
normalization_layer = layers.Rescaling(1./255)

# ============================================================
# DATA AUGMENTATION
# ============================================================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
])

print("✅ Preprocessing prêt !")

# ============================================================
# TEST AUTOMATIQUE
# ============================================================
if __name__ == "__main__":

    print("🔄 Test augmentation...")

    # اختيار فولدر فيه الصور
    folder = "plantdisease/PlantVillage/Pepper__bell___Bacterial_spot"

    # تأكد أن الفولدر موجود
    if not os.path.exists(folder):
        raise ValueError(f"❌ Folder introuvable: {folder}")

    # اختيار صورة عشوائية
    image_name = random.choice(os.listdir(folder))
    image_path = os.path.join(folder, image_name)

    print(f"📸 Image choisie : {image_name}")

    # تحميل الصورة
    img = tf.keras.utils.load_img(image_path, target_size=img_size)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    # ============================================================
    # AFFICHAGE AUGMENTATION
    # ============================================================
    plt.figure(figsize=(8, 8))

    for i in range(6):
        augmented = data_augmentation(img_array)

        plt.subplot(2, 3, i + 1)
        plt.imshow(augmented[0].numpy().astype("uint8"))
        plt.axis("off")

    plt.suptitle("Data Augmentation Example", fontsize=14)
    plt.tight_layout()

    # sauvegarde image
    plt.savefig("augmentation_exemple.png")

    print("🖼️ Image sauvegardée : augmentation_exemple.png")

    plt.show()