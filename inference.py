"""Stage C - ResNet50 denomination classifier with open-set rejection metrics.

`assess()` never trusts the softmax maximum alone: it also checks the margin between the two best classes and the
normalised entropy of the whole distribution, all thresholds coming from config.py.
"""
import json
from dataclasses import dataclass, field

import numpy as np

import config
import preprocessing as pre
from descriptions import display_name, get_description, short_name
from model import build_classifier


@dataclass
class Assessment:
    label: str
    confidence: float
    margin: float
    entropy: float                      # normalised 0..1
    top3: list = field(default_factory=list)
    accepted: bool = False
    failed_checks: list = field(default_factory=list)


def assess_probabilities(probs: np.ndarray, names) -> Assessment:
    probs = np.asarray(probs, dtype="float64")
    order = np.argsort(probs)[::-1]
    top1 = float(probs[order[0]])
    top2 = float(probs[order[1]]) if len(probs) > 1 else 0.0
    entropy = float(-(probs * np.log(probs + 1e-12)).sum() / np.log(max(len(probs), 2)))
    failed = []
    if top1 < config.CONFIDENCE_THRESHOLD:
        failed.append("confidence")
    if top1 - top2 < config.MARGIN_THRESHOLD:
        failed.append("margin")
    if entropy > config.ENTROPY_THRESHOLD:
        failed.append("entropy")
    return Assessment(names[int(order[0])], top1, top1 - top2, entropy,
                      [(names[i], float(probs[i])) for i in order[:3]], not failed, failed)


class DenominationClassifier:
    def __init__(self):
        if not config.CLASSIFIER_WEIGHTS.exists() or not config.CLASSES_PATH.exists():
            raise FileNotFoundError(f"Classifier files missing in {config.MODEL_DIR}. Run `python train.py` "
                                    f"(needs {config.CLASSIFIER_WEIGHTS.name} and {config.CLASSES_PATH.name}).")
        self.names = json.loads(config.CLASSES_PATH.read_text())
        self.model, _ = build_classifier(len(self.names), pretrained=False)
        self.model.load_weights(config.CLASSIFIER_WEIGHTS)          # weights only - no optimizer state

    def probabilities(self, rgb: np.ndarray) -> np.ndarray:
        return self.model(pre.rgb_to_batch(rgb), training=False).numpy()[0]

    def assess(self, rgb: np.ndarray) -> Assessment:
        return assess_probabilities(self.probabilities(rgb), self.names)

    def predict(self, img) -> dict:
        """Legacy helper (classifier only, no detector). Prefer pipeline.CurrencyRecognitionPipeline."""
        a = self.assess(np.asarray(img.convert("RGB")))
        return {"label": a.label if a.accepted else None, "name": display_name(a.label) if a.accepted else None,
                "short": short_name(a.label) if a.accepted else None, "confidence": a.confidence,
                "is_confident": a.accepted, "top3": a.top3,
                "description": get_description(a.label) if a.accepted else "Not sure."}


CurrencyClassifier = DenominationClassifier   # backward-compatible name
