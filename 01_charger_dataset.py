import tensorflow as tf
import matplotlib.pyplot as plt
import os

# ============================================================
# CHEMIN DU DATASET
# ============================================================
data_dir = "plantdisease/PlantVillage"

# تحقق من وجود المسار
if not os.path.exists(data_dir):
    raise ValueError(f"❌ Dataset introuvable : {data_dir}")

# ============================================================
# PARAMÈTRES
# ============================================================
img_size = (224, 224)
batch_size = 32

print("📦 Chargement du dataset...")

# ============================================================
# CHARGEMENT DU DATASET (SANS SHUFFLE ❗)
# ============================================================
train_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size,
    shuffle=False   # 🔥 مهم
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size,
    shuffle=False   # 🔥 مهم
)

# ============================================================
# AFFICHER LES CLASSES
# ============================================================
class_names = train_dataset.class_names
print(f"\n✅ Nombre de classes : {len(class_names)}")
print(f"📌 Classes : {class_names}")

# ============================================================
# OPTIMISATION (SANS SHUFFLE)
# ============================================================
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
val_dataset   = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# ============================================================
# تقليل البيانات (باش ما يثقلش)
# ============================================================
train_dataset = train_dataset.take(100)
val_dataset   = val_dataset.take(20)

print("\n🚀 Dataset prêt (version légère pour test)")

# ============================================================
# VISUALISATION
# ============================================================
plt.figure(figsize=(10, 8))

for images, labels in train_dataset.take(1):
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]], fontsize=8)
        plt.axis("off")

plt.suptitle("Exemples du Dataset PlantVillage", fontsize=14)
plt.tight_layout()

# sauvegarde image
output_path = "exemples_dataset.png"
plt.savefig(output_path)

print(f"\n🖼️ Image sauvegardée : {output_path}")

plt.show()