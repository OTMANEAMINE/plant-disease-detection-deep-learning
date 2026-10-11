import tensorflow as tf
from tensorflow.keras import layers, models

# ============================================================
# PARAMÈTRES
# ============================================================
img_size = (224, 224)
num_classes = 16

# ============================================================
# BASE MODEL (MobileNetV2)
# ============================================================
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# freeze layers
base_model.trainable = False

# ============================================================
# MODEL
# ============================================================
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

# ============================================================
# COMPILATION
# ============================================================
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ============================================================
# AFFICHAGE
# ============================================================
model.summary()