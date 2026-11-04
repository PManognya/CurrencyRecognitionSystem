"""Classify images from the command line with the full pipeline.

Usage: python predict.py note.jpg [more.jpg ...] [--json]
"""
import argparse
import json

from PIL import Image

from pipeline import CurrencyRecognitionPipeline, PipelineNotReady

ap = argparse.ArgumentParser()
ap.add_argument("images", nargs="+")
ap.add_argument("--json", action="store_true")
args = ap.parse_args()

try:
    pipe = CurrencyRecognitionPipeline()
except PipelineNotReady as e:
    raise SystemExit(f"\n{e}")

for path in args.images:
    r = pipe.analyze_pil(Image.open(path))
    if args.json:
        print(json.dumps({"image": path, **r.to_dict()}, ensure_ascii=False))
        continue
    print(f"\n{path}\n  Status     : {r.message}")
    if r.status == "ok":
        print(f"  Prediction : {r.name}  ({r.confidence:.1%})\n  Description: {r.description}")
    else:
        print(f"  Reason     : {r.reason}")
    if r.detector_score is not None:
        print(f"  Detector   : {r.detector_score:.2f}")
    if r.top3:
        print("  Top-3      :", ", ".join(f"{n} {p:.0%}" for n, p in r.top3), f"| margin {r.margin:.2f} entropy {r.entropy:.2f}")
