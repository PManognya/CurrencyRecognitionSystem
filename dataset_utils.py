"""Dataset discovery, reproducible stratified splitting and tf.data construction (shared by all training scripts)."""
import random
from pathlib import Path

import tensorflow as tf

import config
import preprocessing as pre

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(folder: Path):
    return sorted(f for f in Path(folder).rglob("*") if f.is_file() and f.suffix.lower() in IMAGE_EXTS)


def currency_classes(root: Path = config.DATASET_DIR, max_per_class=None):
    """{'INR_10': [paths], ...} sorted by class name; hidden / legacy / empty folders are skipped."""
    out = {}
    if not Path(root).exists():
        return out
    for d in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        if d.name.startswith(".") or d.name in config.IGNORED_DATASET_FOLDERS:
            continue
        files = list_images(d)
        if max_per_class and len(files) > max_per_class:
            files = random.Random(config.SEED).sample(files, max_per_class)
        if files:
            out[d.name] = sorted(files)
    return out


def negative_groups(root: Path = config.NEGATIVE_DIR):
    """{'face': [paths], ...}; images placed directly in negative/ go to the 'misc' group."""
    out = {}
    root = Path(root)
    if not root.exists():
        return out
    for d in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        files = list_images(d)
        if files:
            out[d.name] = files
    direct = sorted(f for f in root.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS)
    if direct:
        out["misc"] = direct
    return out


def stratified_split(groups: dict, val_split=config.VAL_SPLIT, seed=config.SEED):
    """Reproducible per-group split -> (train, val) lists of (path, group_name)."""
    rng = random.Random(seed)
    train, val = [], []
    for name in sorted(groups):
        files = list(groups[name])
        rng.shuffle(files)
        n_val = max(1, round(len(files) * val_split)) if len(files) >= 2 else 0
        val += [(f, name) for f in files[:n_val]]
        train += [(f, name) for f in files[n_val:]]
    return train, val


def make_dataset(paths, labels, training, aug=None, extra=None, batch_size=config.BATCH_SIZE):
    """paths/labels (/extra): python lists. Yields (x, y) or (x, y, extra) batches.
    Corrupt / undecodable files are silently dropped by ignore_errors."""
    tensors = (list(map(str, paths)), list(labels)) if extra is None else (list(map(str, paths)), list(labels), list(extra))
    ds = tf.data.Dataset.from_tensor_slices(tensors)
    if training:
        ds = ds.shuffle(len(paths), seed=config.SEED, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, *rest: (pre.load_for_training(p), *rest), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.ignore_errors()
    if training and aug is not None:
        ds = ds.map(lambda x, *rest: (pre.augment(x, aug), *rest), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
