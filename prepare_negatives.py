"""Fill negative/ with general NOT-currency images from Caltech101 (faces, objects, scenes).

  Faces / Faces_easy      -> negative/face
  BACKGROUND_Google       -> negative/background
  everything else         -> negative/random_objects      (dollar_bill is always excluded)

This is only a starting point. The detector works far better when you ALSO add photos from YOUR webcam:
    python capture_images.py negative/face ; negative/background ; negative/random_objects ...

Usage: python prepare_negatives.py [--max 1500]      (needs: pip install tensorflow-datasets)
"""
import argparse
import random

import numpy as np
from PIL import Image

import config

ap = argparse.ArgumentParser()
ap.add_argument("--max", type=int, default=1500, help="max number of non-face object images")
args = ap.parse_args()

import tensorflow_datasets as tfds

ds, info = tfds.load("caltech101", split="train+test", with_info=True)
names = info.features["label"].names
skip = {i for i, n in enumerate(names) if "dollar" in n.lower()}
items = [(e["image"].numpy(), int(e["label"])) for e in ds if int(e["label"]) not in skip]
random.seed(config.SEED)
random.shuffle(items)

def group_of(label):
    n = names[label].lower()
    return "face" if "face" in n else "background" if "background" in n else "random_objects"

counts, kept_objects = {}, 0
for img, label in items:
    g = group_of(label)
    if g == "random_objects":
        if kept_objects >= args.max:
            continue
        kept_objects += 1
    out = config.NEGATIVE_DIR / g
    out.mkdir(parents=True, exist_ok=True)
    counts[g] = counts.get(g, 0) + 1
    Image.fromarray(np.asarray(img)).convert("RGB").save(out / f"c101_{counts[g]:05d}.jpg", quality=90)
print("Saved:", counts, "->", config.NEGATIVE_DIR)
