"""
============================================================
MODULE PARTAGÉ : CONFIGURATIONS DES MODÈLES
============================================================
Ce fichier regroupe les fonctions de construction des modèles
pour éviter la duplication dans les autres scripts.
"""

import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model

IMAGE_SIZE  = (224, 224)
NUM_CLASSES = 38

data_augmentation = Sequential([
    layers.Rescaling(1./255),
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomBrightness(0.1),
], name="data_augmentation")


def construire_cnn_simple(num_classes=NUM_CLASSES, input_shape=(224, 224, 3)):
    model = Sequential([
        layers.Input(shape=input_shape),
        data_augmentation,
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.3),
        layers.Conv2D(256, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax', name='output')
    ], name="CNN_Simple")
    return model


def construire_modele_transfer(num_classes=NUM_CLASSES, input_shape=(224, 224, 3)):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights='imagenet'
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
    model = Model(inputs, outputs, name="MobileNetV2_TL")
    return model, base_model


def activer_fine_tuning(model, base_model, learning_rate=1e-5, nb_couches_a_debloquer=30):
    base_model.trainable = True
    for layer in base_model.layers[:-nb_couches_a_debloquer]:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    return model
