"""Dataset sanity report: classes, counts, corrupt files, image sizes, class imbalance, negative set.

Usage: python check_dataset.py
"""
import statistics
import sys

from PIL import Image

import config
import dataset_utils as du


def scan(files):
    bad, sizes = [], []
    for f in files:
        try:
            with Image.open(f) as im:
                sizes.append(im.size)
                im.verify()
        except Exception:
            bad.append(f)
    return bad, sizes


def size_summary(sizes):
    if not sizes:
        return "n/a"
    w, h = [s[0] for s in sizes], [s[1] for s in sizes]
    return (f"width min/median/max = {min(w)}/{int(statistics.median(w))}/{max(w)}, "
            f"height = {min(h)}/{int(statistics.median(h))}/{max(h)}")


def main():
    groups = du.currency_classes(max_per_class=None)
    if not groups:
        sys.exit(f"No class folders with images in {config.DATASET_DIR}. Run prepare_data.py or make_folders.py.")
    print(f"=== CURRENCY CLASSES ({len(groups)}) ===")
    all_bad, all_sizes = [], []
    for name, files in groups.items():
        bad, sizes = scan(files)
        all_bad += bad; all_sizes += sizes
        print(f"{name:<12}{len(files):>6}" + (f"   ({len(bad)} corrupt)" if bad else ""))
    counts = {k: len(v) for k, v in groups.items()}
    total = sum(counts.values())
    print(f"\nTotal currency images: {total}")
    ratio = max(counts.values()) / max(min(counts.values()), 1)
    smallest = min(counts, key=counts.get)
    print(f"Class imbalance (largest / smallest): {ratio:.1f}x   [largest {max(counts.values())}, smallest {smallest}={counts[smallest]}]")
    if ratio > 3:
        print("  WARNING: strong imbalance - class weights compensate a little; add images to the small classes.")
    print(f"Image sizes (currency): {size_summary(all_sizes)}")
    countries = sorted({n.split('_')[0] for n in groups})
    print(f"Countries: {', '.join(countries)}")

    neg = du.negative_groups()
    print("\n=== NEGATIVE (NOT-CURRENCY) IMAGES ===")
    if not neg:
        print("None found in negative/ -> the currency detector cannot be trained (run prepare_negatives.py, capture_images.py).")
        n_neg = 0
    else:
        n_neg = 0
        for name, files in neg.items():
            bad, sizes = scan(files)
            all_bad += bad
            n_neg += len(files)
            print(f"{name:<16}{len(files):>6}" + (f"   ({len(bad)} corrupt)" if bad else ""))
        print(f"\nNegative images: {n_neg}   (ratio negatives : currency = {n_neg / max(total, 1):.2f})")
        if n_neg < config.DETECTOR_MIN_NEGATIVES:
            print(f"  WARNING: fewer than {config.DETECTOR_MIN_NEGATIVES} negatives - add your own webcam photos of your face, room, objects.")
        if "hard_negatives" not in neg:
            print("  Tip: press N in webcam.py whenever it wrongly sees a note to collect hard negatives.")

    print(f"\nCorrupt images: {len(all_bad)}")
    for f in all_bad:
        print("  corrupt ->", f)
    legacy = config.DATASET_DIR / "NOT_CURRENCY"
    if legacy.exists():
        print(f"\nNOTE: {legacy} is a legacy folder and is ignored. Move its images to negative/random_objects/ or negative/face/.")


if __name__ == "__main__":
    main()
