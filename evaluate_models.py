"""Print the accuracy of the models you already trained - no retraining.

    python evaluate_models.py                  # both models
    python evaluate_models.py --model detector # only the currency / not-currency CNN
    python evaluate_models.py --model classifier   # only the ResNet50 denomination classifier

It loads model/*.weights.h5, rebuilds the SAME validation split the training scripts used (same seed, same folders) and
evaluates on it. Result is also written to outputs/accuracy_summary.txt.

IMPORTANT: the validation split is derived from the images in dataset/ and negative/. Run this on the same dataset you
trained with (e.g. the Colab one). If your local folders contain different images, the split differs, some "validation"
images may have been training images, and the numbers will be too optimistic.
"""
import argparse
import json
import os
from datetime import datetime

import numpy as np
import tensorflow as tf

import config
import dataset_utils as du
from currency_detector import build_detector
from model import build_classifier


def _predict(model, ds):
    y, p = [], []
    for x, yy in ds:                              # one pass so labels and predictions stay aligned
        y.append(yy.numpy()); p.append(model(x, training=False).numpy())
    return np.concatenate(y), np.concatenate(p)


def eval_classifier():
    names = json.loads(config.CLASSES_PATH.read_text())
    groups = du.currency_classes(max_per_class=config.MAX_IMAGES_PER_CLASS)
    missing = [n for n in names if n not in groups]
    if missing:
        raise SystemExit(f"Class folders missing in {config.DATASET_DIR}: {missing}")
    groups = {n: groups[n] for n in names}        # same class order as the trained model
    _, val_items = du.stratified_split(groups)
    index = {n: i for i, n in enumerate(names)}
    ds = du.make_dataset([p for p, _ in val_items], [index[g] for _, g in val_items], False)
    model, _ = build_classifier(len(names), pretrained=False)
    model.load_weights(config.CLASSIFIER_WEIGHTS)
    y, probs = _predict(model, ds)
    pred = probs.argmax(1)
    acc = float((pred == y).mean())
    top3 = float(np.mean([t in np.argsort(r)[-3:] for t, r in zip(y, probs)]))
    return {"title": "ResNet50 classifier (which denomination?)", "images": len(y), "accuracy": acc,
            "extra": f"top-3 accuracy = {top3:.2%}, classes = {len(names)}"}


def eval_detector(threshold):
    meta = json.loads(config.DETECTOR_META.read_text()) if config.DETECTOR_META.exists() else {}
    backbone = meta.get("backbone", config.DETECTOR_BACKBONE)
    pos, neg = du.currency_classes(max_per_class=config.DETECTOR_MAX_POSITIVES_PER_CLASS), du.negative_groups()
    groups = {f"pos:{k}": v for k, v in pos.items()} | {f"neg:{k}": v for k, v in neg.items()}
    _, val_items = du.stratified_split(groups)
    label = lambda g: 1.0 if g.startswith("pos:") else 0.0
    ds = du.make_dataset([p for p, _ in val_items], [label(g) for _, g in val_items], False)
    model, _ = build_detector(pretrained=False, backbone=backbone)
    model.load_weights(config.DETECTOR_WEIGHTS)
    y, s = _predict(model, ds)
    s = np.nan_to_num(s.ravel(), nan=0.5)
    pred = s >= threshold
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    acc = (tp + tn) / max(len(y), 1)
    extra = (f"backbone = {backbone}, threshold = {threshold}, precision = {tp / max(tp + fp, 1):.3f}, "
             f"recall = {tp / max(tp + fn, 1):.3f}, false-positive rate = {fp / max(fp + tn, 1):.2%}")
    return {"title": f"{'CNN' if backbone == 'cnn' else 'ResNet50'} detector (currency or not?)", "images": len(y), "accuracy": acc, "extra": extra}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["both", "detector", "classifier"], default="both")
    ap.add_argument("--threshold", type=float, default=config.DETECTOR_THRESHOLD, help="detector decision threshold")
    args = ap.parse_args()

    jobs = []
    if args.model in ("both", "detector"):
        jobs.append((config.DETECTOR_WEIGHTS, lambda: eval_detector(args.threshold)))
    if args.model in ("both", "classifier"):
        jobs.append((config.CLASSIFIER_WEIGHTS, eval_classifier))

    results = []
    for weights, fn in jobs:
        if not weights.exists():
            print(f"Skipping: {weights} not found (copy the trained files into {config.MODEL_DIR}).")
            continue
        results.append(fn())

    lines = [f"Accuracy summary - {datetime.now():%Y-%m-%d %H:%M}", "(measured on the validation split, not on separate test data)", ""]
    for r in results:
        lines += [f"{r['title']}", f"  accuracy : {r['accuracy']:.2%}   ({r['images']} validation images)", f"  {r['extra']}", ""]
    text = "\n".join(lines)
    print("\n" + text)
    if results:
        config.OUTPUT_DIR.mkdir(exist_ok=True)
        (config.OUTPUT_DIR / "accuracy_summary.txt").write_text(text, encoding="utf-8")
        print(f"Saved -> {config.OUTPUT_DIR / 'accuracy_summary.txt'}")


if __name__ == "__main__":
    main()
