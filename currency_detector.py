"""Stage B - binary 'is this a banknote?' detector (plain CNN trained from scratch by default;
ResNet50 transfer learning available via config.DETECTOR_BACKBONE).

This is a *classifier* on a cropped region (the ROI produced by note_localizer), not a bounding-box detector.
It answers one question only: banknote vs. not-a-banknote. It is trained with an explicit negative dataset
(negative/ folder) so faces, walls, hands, objects ... are learned as "not currency".
"""
import json

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50

import config
import preprocessing as pre
from model import caffe_preprocess


def _conv_block(x, filters, name):
    for i in (1, 2):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv{i}")(x)
        x = layers.BatchNormalization(name=f"{name}_bn{i}")(x)
        x = layers.ReLU(name=f"{name}_relu{i}")(x)
    return layers.MaxPooling2D(2, name=f"{name}_pool")(x)


def build_detector(pretrained=None, backbone=None):
    """Returns (model, base). base is the ResNet50 sub-model for the 'resnet50' backbone, or None for the plain CNN."""
    pretrained = config.PRETRAINED if pretrained is None else pretrained
    backbone = (backbone or config.DETECTOR_BACKBONE).lower()
    inputs = layers.Input(shape=(*config.IMG_SIZE, 3), name="image")
    if backbone == "cnn":
        x = layers.Rescaling(1.0 / 255, name="cnn_preprocess")(inputs)
        for i, f in enumerate((32, 64, 128, 256, 256), 1):
            x = _conv_block(x, f, f"block{i}")
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.4)(x)
        base = None
    elif backbone == "resnet50":
        x = layers.Lambda(caffe_preprocess, name="resnet_preprocess")(inputs)
        base = ResNet50(weights="imagenet" if pretrained else None, include_top=False,
                        input_shape=(*config.IMG_SIZE, 3), name="resnet50")
        base.trainable = False
        x = base(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
    else:
        raise ValueError(f"DETECTOR_BACKBONE must be 'cnn' or 'resnet50', got {backbone!r}")
    outputs = layers.Dense(1, activation="sigmoid", name="p_currency")(x)
    return models.Model(inputs, outputs, name=f"currency_detector_{backbone}"), base


class CurrencyDetector:
    def __init__(self):
        if not config.DETECTOR_WEIGHTS.exists():
            raise FileNotFoundError(f"Detector weights not found: {config.DETECTOR_WEIGHTS}. Run `python train_detector.py`.")
        self.model, _ = build_detector(pretrained=False)   # backbone comes from config.DETECTOR_BACKBONE
        self.model.load_weights(config.DETECTOR_WEIGHTS)       # weights only - no optimizer state to restore
        self.meta = json.loads(config.DETECTOR_META.read_text()) if config.DETECTOR_META.exists() else {}

    def score_rgb(self, rgb: np.ndarray) -> float:
        """Probability (0..1) that the RGB uint8 image shows a banknote."""
        p = float(self.model(pre.rgb_to_batch(rgb), training=False).numpy()[0, 0])
        return p if np.isfinite(p) else 0.0        # fail closed: a broken score never counts as currency
