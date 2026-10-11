import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# ============================================================
# PARAMÈTRES
# ============================================================
data_dir = "plantdisease/PlantVillage"
img_size = (224, 224)
batch_size = 32

# ============================================================
# DATASET
# ============================================================
train_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

# ============================================================
# NORMALISATION + AUGMENTATION
# ============================================================
data_augmentation = tf.keras.Sequential([
    layers.Rescaling(1./255),
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
])

# ============================================================
# MODEL (MobileNet)
# ============================================================
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

model = tf.keras.Sequential([
    data_augmentation,
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(16, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ============================================================
# TRAINING (مرة وحدة فقط)
# ============================================================
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=3
)

# ============================================================
# SAVE MODEL
# ============================================================
model.save("modele_plantes.keras")
print("✅ Modèle sauvegardé !")

# ============================================================
# COURBE ACCURACY
# ============================================================
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.legend()
plt.title("Accuracy")
plt.savefig("courbe_accuracy.png")
plt.show()

# ============================================================
# COURBE LOSS
# ============================================================
plt.figure()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.legend()
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.savefig("courbe_loss.png")
plt.show()