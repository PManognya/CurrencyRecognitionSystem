"""Denomination classifier: ResNet50 transfer learning (kept from the original project).

No augmentation layers live in the model any more - augmentation happens in the tf.data pipeline
(preprocessing.augment), so the exact same graph is used for training and inference.
"""
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50

import config


def caffe_preprocess(x):
    """ResNet50 (ImageNet) preprocessing: RGB->BGR + mean subtraction. Input is raw 0..255 RGB."""
    x = tf.reverse(x, axis=[-1])
    return x - tf.constant([103.939, 116.779, 123.68], dtype=x.dtype)


def build_classifier(num_classes: int, pretrained=None):
    pretrained = config.PRETRAINED if pretrained is None else pretrained
    inputs = layers.Input(shape=(*config.IMG_SIZE, 3), name="image")
    x = layers.Lambda(caffe_preprocess, name="resnet_preprocess")(inputs)
    base = ResNet50(weights="imagenet" if pretrained else None, include_top=False,
                    input_shape=(*config.IMG_SIZE, 3), name="resnet50")
    base.trainable = False
    x = base(x, training=False)                 # BatchNorm stays in inference mode
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="note")(x)
    return models.Model(inputs, outputs, name="currency_resnet50"), base


build_model = build_classifier   # backward-compatible name
