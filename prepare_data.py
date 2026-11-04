"""Download the India / Turkey / USA banknote datasets from Kaggle and arrange them as
dataset/<CODE>_<VALUE>/ automatically (merges the datasets' own train/test/val folders).

Usage:
    python prepare_data.py                         # download all 3 and organise
    python prepare_data.py --only TRY,USD          # just some countries
    python prepare_data.py --local USD="D:/data/usd_notes"   # use a folder you downloaded manually
    python prepare_data.py --max 600               # max images per note (default 500)
"""
import argparse
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

import config

SOURCES = {   # Kaggle dataset slugs - open the page on Kaggle if one ever fails and fix the slug here
    "INR": "vishalmane109/indian-currency-note-images-dataset-2020",
    "TRY": "baltacifatih/turkish-lira-banknote-dataset",
    "USD": "aishwaryatechie/usd-bill-classification-dataset",
}
VALID = {
    "INR": {10, 20, 50, 100, 200, 500, 2000},
    "TRY": {5, 10, 20, 50, 100, 200},
    "USD": {1, 2, 5, 10, 20, 50, 100},
}
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SKIP = {"train", "test", "val", "valid", "validation", "training", "testing", "dataset", "data"}
WORDS = [("twothousand", 2000), ("onethousand", 1000), ("fivehundred", 500), ("twohundred", 200),
         ("onehundred", 100), ("thousand", 1000), ("hundred", 100), ("fifty", 50), ("twenty", 20),
         ("ten", 10), ("five", 5), ("two", 2)]


def value_from_name(name):
    low = name.lower()
    if low in SKIP:
        return None
    digits = re.findall(r"\d+", low)
    if digits:
        return int(digits[0])
    flat = re.sub(r"[^a-z]", "", low)
    for w, v in WORDS:
        if w in flat:
            return v
    return None


def find_value(file, root, valid):
    """Nearest parent folder (below root) whose name maps to a valid denomination."""
    for parent in file.relative_to(root).parents:
        if parent.name and (v := value_from_name(parent.name)) in valid:
            return v
    return None


def download(code):
    import kagglehub
    try:
        return Path(kagglehub.dataset_download(SOURCES[code]))
    except Exception as e:
        print(f"  ! Could not download {SOURCES[code]}: {e}\n"
              f"    Try kagglehub.login() / check the slug, or download manually and use --local {code}=<folder>")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="INR,TRY,USD")
    ap.add_argument("--local", nargs="*", default=[], help='e.g. USD="D:/path"')
    ap.add_argument("--max", type=int, default=500)
    args = ap.parse_args()
    local = dict(item.split("=", 1) for item in args.local)
    random.seed(config.SEED)
    config.DATASET_DIR.mkdir(exist_ok=True)

    summary = {}
    for code in [c.strip().upper() for c in args.only.split(",")]:
        print(f"\n[{code}] {SOURCES[code]}")
        root = Path(local[code].strip('"')) if code in local else download(code)
        if not root or not root.exists():
            continue
        groups, unmapped = defaultdict(list), 0
        for f in root.rglob("*"):
            if f.suffix.lower() in EXTS:
                v = find_value(f, root, VALID[code])
                if v is None:
                    unmapped += 1
                else:
                    groups[v].append(f)
        folders = sorted({p.name for p in root.rglob("*") if p.is_dir()})[:25]
        print("  folders found:", folders)
        for v, files in sorted(groups.items()):
            if len(files) > args.max:
                files = random.sample(files, args.max)
            dst = config.DATASET_DIR / f"{code}_{v}"
            shutil.rmtree(dst, ignore_errors=True)
            dst.mkdir(parents=True)
            for i, f in enumerate(files):
                shutil.copy2(f, dst / f"{i:05d}{f.suffix.lower()}")
            summary[f"{code}_{v}"] = len(files)
        missing = sorted(VALID[code] - set(groups))
        print(f"  organised {len(groups)} notes | unmapped images: {unmapped} | missing: {missing or 'none'}")
        if unmapped or missing:
            print("  (If folder names are unusual, rename them to the denomination, e.g. '500', and run again.)")

    print("\n=== dataset/ summary ===")
    for k, n in summary.items():
        print(f"{k:>10}: {n}")
    print(f"\nTotal: {sum(summary.values())} images, {len(summary)} classes.  Next: python train.py")


if __name__ == "__main__":
    main()
