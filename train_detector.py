"""Train the currency / NON-currency detector (stage B of the pipeline).

positives : every image in dataset/<CODE>_<VALUE>/ (+ synthetic 'note on a background' composites, train split only)
negatives : every image in negative/<group>/   (faces, hands, phones, walls, objects, backgrounds, hard negatives ...)

Usage:  python train_detector.py
        python train_detector.py --composites 1500
"""
import argparse
import json
import warnings
import random
import shutil
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image, ImageFilter, ImageOps
from sklearn.metrics import roc_auc_score, roc_curve
from tensorflow.keras import callbacks, layers, optimizers

import checkpointing as ckpt
import config
import dataset_utils as du
from currency_detector import build_detector


warnings.filterwarnings("ignore", message="Your input ran out of data")   # harmless: unknown dataset cardinality


def make_composites(pos_files, neg_files, n, out_dir, seed=config.SEED):
    """Paste a (rotated, scaled) banknote picture onto a random negative background -> extra positives that look
    like 'a note lying in a scene'. Built from TRAIN files only, so validation stays untouched."""
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for i in range(n):
        try:
            bg = ImageOps.fit(Image.open(rng.choice(neg_files)).convert("RGB"), (640, 480))
            note = Image.open(rng.choice(pos_files)).convert("RGBA")
            width = int(640 * rng.uniform(0.35, 0.85))
            note = note.resize((width, max(20, int(width * note.height / note.width))))
            note = note.rotate(rng.uniform(-30, 30), expand=True, resample=Image.BICUBIC)   # transparent corners
            if note.height > 470:
                note = note.resize((int(note.width * 470 / note.height), 470))
            x, y = rng.randint(0, max(0, 640 - note.width)), rng.randint(0, max(0, 480 - note.height))
            bg.paste(note, (x, y), note)
            if rng.random() < 0.4:
                bg = bg.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.5)))
            path = out_dir / f"comp_{i:05d}.jpg"
            bg.save(path, quality=88)
            made.append(path)
        except Exception:
            continue
    return made


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--composites", type=int, default=config.DETECTOR_COMPOSITES)
    ap.add_argument("--epochs-head", type=int, default=config.DETECTOR_EPOCHS_HEAD)
    ap.add_argument("--epochs-fine", type=int, default=config.DETECTOR_EPOCHS_FINE)
    ap.add_argument("--fresh", action="store_true", help="ignore saved checkpoints and train from scratch")
    args = ap.parse_args()
    tf.keras.utils.set_random_seed(config.SEED)
    config.MODEL_DIR.mkdir(exist_ok=True); config.OUTPUT_DIR.mkdir(exist_ok=True)
    print("GPU available:", bool(tf.config.list_physical_devices("GPU")))

    pos_groups = du.currency_classes(max_per_class=config.DETECTOR_MAX_POSITIVES_PER_CLASS)
    neg_groups = du.negative_groups()
    n_pos, n_neg = sum(map(len, pos_groups.values())), sum(map(len, neg_groups.values()))
    if not pos_groups or not neg_groups:
        raise SystemExit("Need currency images in dataset/ AND non-currency images in negative/. "
                         "Run `python prepare_negatives.py` and/or `python capture_images.py negative/face`.")
    print(f"positives={n_pos} negatives={n_neg} ({len(neg_groups)} groups: {sorted(neg_groups)})")
    if n_neg < config.DETECTOR_MIN_NEGATIVES:
        print(f"WARNING: only {n_neg} negatives (< {config.DETECTOR_MIN_NEGATIVES}). The detector will be weak - add your own "
              "webcam negatives (python capture_images.py negative/face ...).")

    groups = {f"pos:{k}": v for k, v in pos_groups.items()} | {f"neg:{k}": v for k, v in neg_groups.items()}
    train_items, val_items = du.stratified_split(groups)
    composites = []
    if args.composites > 0:
        cdir = config.CACHE_DIR / "composites"
        shutil.rmtree(cdir, ignore_errors=True)
        composites = make_composites([p for p, g in train_items if g.startswith("pos:")],
                                     [p for p, g in train_items if g.startswith("neg:")], args.composites, cdir)
        print(f"synthetic composites: {len(composites)}")
        train_items += [(p, "pos:composite") for p in composites]

    label = lambda g: 1.0 if g.startswith("pos:") else 0.0
    ytr = [label(g) for _, g in train_items]
    train_ds = du.make_dataset([p for p, _ in train_items], ytr, True, config.DETECTOR_AUG)
    val_ds = du.make_dataset([p for p, _ in val_items], [label(g) for _, g in val_items], False)
    n1 = sum(ytr); n0 = len(ytr) - n1
    class_weight = {0: len(ytr) / (2 * max(n0, 1)), 1: len(ytr) / (2 * max(n1, 1))}

    run = ckpt.Run("detector", monitor="val_auc", mode="max", fresh=args.fresh,
                   signature={"backbone": config.DETECTOR_BACKBONE, "positives": n_pos, "negatives": n_neg,
                              "composites": len(composites), "train": len(train_items), "val": len(val_items),
                              "img": config.IMG_SIZE, "batch": config.BATCH_SIZE, "seed": config.SEED,
                              "fine_layers": config.DETECTOR_FINE_TUNE_LAYERS, "pretrained": config.PRETRAINED})
    model, base = build_detector()
    metrics = ["accuracy", tf.keras.metrics.AUC(name="auc")]
    lr_cb = lambda: callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, verbose=1)
    model.compile(optimizers.Adam(1e-3), "binary_crossentropy", metrics=metrics)
    if base is None:    # plain CNN: everything is trained from scratch in one phase
        run.fit(model, "train", train_ds, val_ds, epochs=args.epochs_head + args.epochs_fine,
                class_weight=class_weight, callbacks=[lr_cb()])
        model.load_weights(run.best_path)
    else:               # ResNet50: train the head, then fine-tune the top layers
        run.fit(model, "head", train_ds, val_ds, epochs=args.epochs_head, class_weight=class_weight, callbacks=[lr_cb()])
        model.load_weights(run.best_path)

        base.trainable = True
        for layer in base.layers[:-config.DETECTOR_FINE_TUNE_LAYERS]:
            layer.trainable = False
        for layer in base.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
        model.compile(optimizers.Adam(1e-5), "binary_crossentropy", metrics=metrics)
        run.fit(model, "fine", train_ds, val_ds, epochs=args.epochs_fine, class_weight=class_weight, callbacks=[lr_cb()])
    run.export_best(model, config.DETECTOR_WEIGHTS)       # best weights -> model/currency_detector.weights.h5

    # ---- evaluation (labels, scores and group names are carried together so nothing can misalign)
    group_names = sorted(groups)
    eval_ds = du.make_dataset([p for p, _ in val_items], [label(g) for _, g in val_items], False,
                              extra=[group_names.index(g) for _, g in val_items])
    y, s, gid = [], [], []
    for x, yy, gg in eval_ds:
        y.append(yy.numpy()); s.append(model(x, training=False).numpy().ravel()); gid.append(gg.numpy())
    y, s, gid = np.concatenate(y), np.nan_to_num(np.concatenate(s), nan=0.5), np.concatenate(gid)
    thr = config.DETECTOR_THRESHOLD
    pred = s >= thr
    tp, fp, fn, tn = int(((pred) & (y == 1)).sum()), int(((pred) & (y == 0)).sum()), int(((~pred) & (y == 1)).sum()), int(((~pred) & (y == 0)).sum())
    precision, recall, fpr = tp / max(tp + fp, 1), tp / max(tp + fn, 1), fp / max(fp + tn, 1)
    auc = float(roc_auc_score(y, s)) if len(set(y)) > 1 else float("nan")
    fprs, tprs, thrs = roc_curve(y, s)
    ok = np.where((fprs <= 0.01) & np.isfinite(thrs))[0]
    suggested = float(min(max(thrs[ok[-1]], 0.0), 1.0)) if len(ok) else 0.95
    lines = [f"Validation images: {len(y)} (currency {int(y.sum())}, non-currency {int((y == 0).sum())})",
             f"AUC={auc:.4f}  at DETECTOR_THRESHOLD={thr}: precision={precision:.3f} recall={recall:.3f} false-positive-rate={fpr:.3%}",
             f"Suggested threshold for <=1% false positives on this validation set: {suggested:.2f}", "",
             "False-accept rate per negative group (lower is better):"]
    for gi, name in enumerate(group_names):
        if name.startswith("neg:"):
            m = gid == gi
            if m.any():
                lines.append(f"  {name[4:]:<16} {(s[m] >= thr).mean():6.1%}  (n={int(m.sum())})")
    report = "\n".join(lines)
    print("\n" + report)
    (config.OUTPUT_DIR / "detector_report.txt").write_text(report)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    bins = np.linspace(0, 1, 31)
    ax[0].hist(s[y == 1], bins=bins, alpha=.7, label="currency"); ax[0].hist(s[y == 0], bins=bins, alpha=.7, label="not currency")
    ax[0].axvline(thr, c="k", ls="--"); ax[0].set_title("Detector score distribution"); ax[0].legend()
    ax[1].imshow([[tn, fp], [fn, tp]], cmap="Blues"); ax[1].set_xticks([0, 1], ["pred no", "pred yes"]); ax[1].set_yticks([0, 1], ["not currency", "currency"])
    for (i, j), v in {(0, 0): tn, (0, 1): fp, (1, 0): fn, (1, 1): tp}.items():
        ax[1].text(j, i, v, ha="center", va="center", color="red")
    ax[1].set_title(f"Confusion @ {thr}")
    ax[2].plot(fprs, tprs); ax[2].set_xlabel("false positive rate"); ax[2].set_ylabel("true positive rate"); ax[2].set_title(f"ROC (AUC {auc:.3f})")
    fig.tight_layout(); fig.savefig(config.OUTPUT_DIR / "detector_eval.png", dpi=150); plt.close(fig)

    config.DETECTOR_META.write_text(json.dumps({
        "trained_at": datetime.now().isoformat(timespec="seconds"), "auc": auc, "precision": precision, "recall": recall,
        "false_positive_rate": fpr, "threshold_evaluated": thr, "suggested_threshold": suggested,
        "positives": n_pos, "negatives": n_neg, "composites": len(composites), "backbone": config.DETECTOR_BACKBONE, "img_size": list(config.IMG_SIZE)}, indent=1))
    print(f"\nSaved detector -> {config.DETECTOR_WEIGHTS}")


if __name__ == "__main__":
    main()
