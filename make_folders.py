"""Create the empty dataset/ (currency) and negative/ (not-currency) folders.

Usage:
    python make_folders.py                                # India + Turkey + USA presets + negative groups
    python make_folders.py INR:10,20,50 USD:1,5,10,20     # custom currency folders
"""
import sys
from pathlib import Path

import config

PRESET = ["INR:10,20,50,100,200,500,2000", "TRY:5,10,20,50,100,200", "USD:1,2,5,10,20,50,100"]

for spec in (sys.argv[1:] or PRESET):
    code, values = spec.split(":")
    for v in values.split(","):
        d = config.DATASET_DIR / f"{code.upper()}_{v.strip()}"
        d.mkdir(parents=True, exist_ok=True)
        print("created", d.relative_to(config.BASE_DIR))
for g in config.NEGATIVE_GROUPS:
    d = config.NEGATIVE_DIR / g
    d.mkdir(parents=True, exist_ok=True)
    print("created", d.relative_to(config.BASE_DIR))
print("\nFill dataset/* with note photos and negative/* with NON-note photos, then run: python check_dataset.py")
