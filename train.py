"""Train the ResNet50 DENOMINATION classifier (currency classes only - no 'not currency' class here).

The currency / non-currency decision is made by a separate model: see train_detector.py.

Usage:  python train.py
        python train.py --epochs-head 6 --epochs-fine 10
"""
import argparse
import json
import warnings
from collections import Counter
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import callbacks, layers, optimizers

import checkpointing as ckpt
import config
import dataset_utils as du
from inference import assess_probabilities
from model import build_classifier


# tf.data cardinality is unknown because unreadable images are dropped on the fly; Keras then prints a harmless
# 'input ran out of data' notice at the end of every epoch. Silence only that message.
warnings.filterwarnings("ignore", message="Your input ran out of data")


def plot_history(run, phases):
    """Training curves from the per-epoch CSV logs (so they also cover epochs trained before a restart)."""
    dfs = [run.history(p) for p in phases]
    cat = lambda k: sum([d[k].tolist() for d in dfs], [])
    switch = len(dfs[0]) - 0.5
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, tr, va, title in zip(ax, ("accuracy", "loss"), ("val_accuracy", "val_loss"), ("Accuracy", "Loss")):
        a.plot(cat(tr), label="train"); a.plot(cat(va), label="val")
        a.axvline(switch, ls="--", c="gray", label="fine-tune starts"); a.set_title(title); a.set_xlabel("epoch"); a.legend()
    fig.tight_layout(); fig.savefig(config.OUTPUT_DIR / "training_curves.png", dpi=150); plt.close(fig)


def evaluate(model, val_ds, names):
    y_true, y_pred = [], []                       # single pass so labels and predictions stay aligned
    for x, y in val_ds:
        y_true.append(y.numpy()); y_pred.append(np.argmax(model(x, training=False).numpy(), axis=1))
    y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)
    labels = list(range(len(names)))
    report = classification_report(y_true, y_pred, labels=labels, target_names=names, digits=3, zero_division=0)
    print("\n" + report)
    (config.OUTPUT_DIR / "classification_report.txt").write_text(report)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    size = max(6, 0.5 * len(names))
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(labels); ax.set_xticklabels(names, rotation=90); ax.set_yticks(labels); ax.set_yticklabels(names)
    for i in labels:
        for j in labels:
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion matrix (validation)")
    fig.tight_layout(); fig.savefig(config.OUTPUT_DIR / "confusion_matrix.png", dpi=150); plt.close(fig)
    return float((y_true == y_pred).mean())


def open_set_report(model, names, limit=600):
    """Why the detector is needed: what the classifier ALONE does with images that are not currency."""
    groups = du.negative_groups()
    files = [f for fs in groups.values() for f in fs]
    if not files:
        msg = "No negative/ images found - cannot measure how the classifier alone treats non-currency inputs."
        print("\n" + msg); (config.OUTPUT_DIR / "open_set_report.txt").write_text(msg); return
    files = files[:limit] if len(files) <= limit else list(np.random.RandomState(config.SEED).choice(files, limit, replace=False))
    ds = du.make_dataset(files, [0] * len(files), training=False, batch_size=8)   # small batches: low memory
    accepted, confs = 0, []
    total = 0
    for x, _ in ds:
        for p in model(x, training=False).numpy():
            a = assess_probabilities(p, names)
            accepted += int(a.accepted); confs.append(a.confidence); total += 1
    confs = np.array(confs)
    msg = (f"Classifier-only behaviour on {total} NON-currency images (negative/):\n"
           f"  mean max-softmax          : {confs.mean():.3f}\n"
           f"  images with confidence >= {config.CONFIDENCE_THRESHOLD}: {(confs >= config.CONFIDENCE_THRESHOLD).mean():.1%}\n"
           f"  accepted after conf+margin+entropy checks: {accepted / total:.1%}  <- would be reported as currency\n"
           f"This is why the separate currency detector (train_detector.py) is mandatory.")
    print("\n" + msg); (config.OUTPUT_DIR / "open_set_report.txt").write_text(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs-head", type=int, default=config.EPOCHS_HEAD)
    ap.add_argument("--epochs-fine", type=int, default=config.EPOCHS_FINE)
    ap.add_argument("--fresh", action="store_true", help="ignore saved checkpoints and train from scratch")
    args = ap.parse_args()
    tf.keras.utils.set_random_seed(config.SEED)
    config.MODEL_DIR.mkdir(exist_ok=True); config.OUTPUT_DIR.mkdir(exist_ok=True)
    print("GPU available:", bool(tf.config.list_physical_devices("GPU")))

    groups = du.currency_classes(max_per_class=config.MAX_IMAGES_PER_CLASS)
    if len(groups) < 2:
        raise SystemExit(f"Need at least 2 class folders with images in {config.DATASET_DIR} (found {len(groups)}).")
    names = sorted(groups)
    index = {n: i for i, n in enumerate(names)}
    train_items, val_items = du.stratified_split(groups)
    print(f"Classes ({len(names)}): {names}\ntrain={len(train_items)} val={len(val_items)}")
    config.CLASSES_PATH.write_text(json.dumps(names))
    (config.OUTPUT_DIR / "classifier_split.json").write_text(json.dumps(
        {"seed": config.SEED, "val": [str(p.relative_to(config.BASE_DIR)) for p, _ in val_items]}, indent=1))

    train_ds = du.make_dataset([p for p, _ in train_items], [index[g] for _, g in train_items], True, config.CLASSIFIER_AUG)
    val_ds = du.make_dataset([p for p, _ in val_items], [index[g] for _, g in val_items], False)

    counts = Counter(g for _, g in train_items)
    total = sum(counts.values())
    class_weight = {index[n]: total / (len(names) * c) for n, c in counts.items()}

    run = ckpt.Run("classifier", monitor="val_accuracy", mode="max", fresh=args.fresh,
                   signature={"classes": names, "train": len(train_items), "val": len(val_items), "img": config.IMG_SIZE,
                              "batch": config.BATCH_SIZE, "seed": config.SEED, "fine_layers": config.FINE_TUNE_LAYERS,
                              "pretrained": config.PRETRAINED})
    model, base = build_classifier(len(names))
    lr_cb = lambda: callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, verbose=1)
    stop = lambda: callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)

    # Phase 1: train the new head on the frozen ResNet50.   (an interrupted run continues from its last finished epoch)
    model.compile(optimizers.Adam(1e-3), "sparse_categorical_crossentropy", metrics=["accuracy"])
    run.fit(model, "head", train_ds, val_ds, epochs=args.epochs_head, class_weight=class_weight, callbacks=[lr_cb(), stop()])
    model.load_weights(run.best_path)

    # Phase 2: fine-tune the top ResNet50 layers at a small learning rate.
    base.trainable = True
    for layer in base.layers[:-config.FINE_TUNE_LAYERS]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    model.compile(optimizers.Adam(1e-5), "sparse_categorical_crossentropy", metrics=["accuracy"])    # fresh optimizer state
    run.fit(model, "fine", train_ds, val_ds, epochs=args.epochs_fine, class_weight=class_weight, callbacks=[lr_cb(), stop()])

    run.export_best(model, config.CLASSIFIER_WEIGHTS)     # best weights -> model/currency_resnet50.weights.h5
    plot_history(run, ["head", "fine"])
    acc = evaluate(model, val_ds, names)
    open_set_report(model, names)
    config.CLASSIFIER_META.write_text(json.dumps({
        "trained_at": datetime.now().isoformat(timespec="seconds"), "classes": names, "img_size": list(config.IMG_SIZE),
        "preprocessing": "rgb uint8 -> float32 -> bilinear resize -> in-graph resnet50 caffe preprocess",
        "val_accuracy": acc, "images_train": len(train_items), "images_val": len(val_items)}, indent=1))
    print(f"\nSaved weights -> {config.CLASSIFIER_WEIGHTS}\nSaved class names -> {config.CLASSES_PATH}\nPlots/reports -> {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
