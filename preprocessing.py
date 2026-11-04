"""The ONE preprocessing pipeline shared by training and inference.

    uint8 RGB image  ->  float32 0..255  ->  bilinear resize to IMG_SIZE (no aspect-ratio crop)
    (the network itself then applies its backbone-specific scaling inside the model graph)

Training decodes files with `load_for_training`; inference (webcam / Streamlit / CLI) calls `rgb_to_batch`.
Both go through `resize_to_input`, so there is no train/inference mismatch.
Augmentation (`augment`) is used only on training data and lives here so it never ends up inside the saved model.
"""
import math

import numpy as np
import tensorflow as tf

import config


def decode_image_file(path):
    data = tf.io.read_file(path)
    img = tf.io.decode_image(data, channels=3, expand_animations=False)   # RGB, uint8
    img.set_shape([None, None, 3])
    return img


def resize_to_input(img):
    """Any RGB image tensor -> float32 [H, W, 3] in 0..255 at config.IMG_SIZE."""
    img = tf.cast(img, tf.float32)
    return tf.image.resize(img, config.IMG_SIZE, method="bilinear")


def load_for_training(path):
    return resize_to_input(decode_image_file(path))


def rgb_to_batch(rgb: np.ndarray) -> np.ndarray:
    """uint8 RGB array [H, W, 3] -> float32 batch [1, 224, 224, 3] (same maths as training)."""
    return tf.expand_dims(resize_to_input(tf.convert_to_tensor(rgb)), 0).numpy()


# ----------------------------------------------------------------------------- augmentation (training only)
def _uniform(lo, hi, shape=()):
    return tf.random.uniform(shape, lo, hi)


def _projective(img, theta, zoom, tx, ty, c0, c1):
    h, w = config.IMG_SIZE
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    cos, sin = tf.cos(theta), tf.sin(theta)
    a0, a1, b0, b1 = cos / zoom, -sin / zoom, sin / zoom, cos / zoom
    a2 = cx - a0 * cx - a1 * cy + tx
    b2 = cy - b0 * cx - b1 * cy + ty
    transform = tf.reshape(tf.stack([a0, a1, a2, b0, b1, b2, c0, c1]), [1, 8])
    out = tf.raw_ops.ImageProjectiveTransformV3(
        images=img[None], transforms=tf.cast(transform, tf.float32), output_shape=[h, w],
        fill_value=0.0, interpolation="BILINEAR", fill_mode="NEAREST")
    return out[0]


def augment(img, p):
    """Rotation (+ random 180 deg), zoom, translation, small perspective, then brightness / contrast /
    mild saturation / colour cast / noise / low-resolution blur. `img`: float32 [H, W, 3] in 0..255."""
    h, w = config.IMG_SIZE
    theta = _uniform(-p["rotation_deg"], p["rotation_deg"]) * math.pi / 180.0
    theta += tf.where(_uniform(0.0, 1.0) < p["flip180_prob"], math.pi, 0.0)
    zoom = _uniform(*p["zoom"])
    tx, ty = _uniform(-p["shift"], p["shift"]) * w, _uniform(-p["shift"], p["shift"]) * h
    c0, c1 = _uniform(-p["perspective"], p["perspective"]) / w, _uniform(-p["perspective"], p["perspective"]) / h
    x = _projective(img, theta, zoom, tx, ty, c0, c1)

    gray = tf.reduce_mean(x, axis=-1, keepdims=True)
    x = gray + (x - gray) * _uniform(*p["saturation"])
    mean = tf.reduce_mean(x)
    x = (x - mean) * _uniform(*p["contrast"]) + mean
    x = x * _uniform(*p["brightness"]) * (1.0 + _uniform(-p["color_cast"], p["color_cast"], [3]))
    x = x + tf.random.normal(tf.shape(x)) * _uniform(0.0, p["noise"])
    k = tf.where(_uniform(0.0, 1.0) < p["lowres_prob"], _uniform(0.4, 0.9), 1.0)      # webcam-like softness
    small = tf.image.resize(x, [tf.cast(h * k, tf.int32), tf.cast(w * k, tf.int32)])
    x = tf.image.resize(small, [h, w])
    x = tf.clip_by_value(x, 0.0, 255.0)
    x.set_shape([h, w, 3])
    return x
