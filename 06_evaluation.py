import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# LOAD MODEL
# ============================================================
model = tf.keras.models.load_model("modele_plantes.keras")

# ============================================================
# DATASET
# ============================================================
data_dir = "plantdisease/PlantVillage"
img_size = (224, 224)
batch_size = 32

val_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

class_names = val_dataset.class_names

# ============================================================
# PREDICTIONS
# ============================================================
y_true = []
y_pred = []

for images, labels in val_dataset:
    preds = model.predict(images)
    preds = np.argmax(preds, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(preds)

# ============================================================
# METRICS
# ============================================================
print("Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))

# ============================================================
# CONFUSION MATRIX
# ============================================================
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=False, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")

plt.savefig("matrice_confusion.png")
plt.show()




# ============================================================
history = model.fit(...)
# COURBE LOSS
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Courbe Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.savefig("courbe_loss.png")
plt.show()