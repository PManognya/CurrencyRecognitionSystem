"""Pipeline decision logic with stub models (no trained weights needed):  python -m unittest discover tests"""
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from inference import Assessment
from note_localizer import NoteLocalizer
from pipeline import CurrencyRecognitionPipeline


class StubDetector:
    def __init__(self, p): self.p = p
    def score_rgb(self, rgb): return self.p


class StubClassifier:
    def __init__(self, label="INR_500", accepted=True):
        self.label, self.accepted = label, accepted
    def assess(self, rgb):
        return Assessment(self.label, 0.95 if self.accepted else 0.4, 0.9 if self.accepted else 0.05, 0.1 if self.accepted else 0.9,
                          [(self.label, 0.95)], self.accepted, [] if self.accepted else ["confidence"])


def make_pipeline(p=0.95, accepted=True):
    pipe = CurrencyRecognitionPipeline.__new__(CurrencyRecognitionPipeline)
    pipe.detector, pipe.classifier = StubDetector(p), StubClassifier(accepted=accepted)
    pipe.localizer, pipe.supported = NoteLocalizer(), ["INR", "TRY", "USD"]
    return pipe


def frame_with_note(pts, size=(480, 640)):
    f = np.full((*size, 3), (90, 100, 95), np.uint8)
    cv2.randn(f, (90, 100, 95), (6, 6, 6))
    cv2.fillConvexPoly(f, np.array(pts, np.int32), (40, 160, 200))
    xs = [p[0] for p in pts]
    for x in range(max(min(xs), 0) + 8, max(xs) - 8, 10):
        cv2.line(f, (x, min(p[1] for p in pts) + 8), (x + 5, max(p[1] for p in pts) - 8), (20, 60, 120), 2)
    return f


GOOD = [[150, 140], [470, 150], [465, 300], [145, 292]]


class TestPipelineDecisions(unittest.TestCase):
    def test_ok(self):
        r = make_pipeline().analyze(frame_with_note(GOOD), mode="live")
        self.assertEqual(r.status, "ok")
        self.assertTrue(r.is_currency)
        self.assertEqual(r.label, "INR_500")
        self.assertIn("500", r.name)

    def test_detector_rejects_non_note(self):
        r = make_pipeline(p=0.10).analyze(frame_with_note(GOOD), mode="live")
        self.assertEqual((r.status, r.is_currency, r.label, r.name), ("no_currency", False, None, "Unknown"))

    def test_no_region_in_live_mode(self):
        r = make_pipeline().analyze(np.full((480, 640, 3), 120, np.uint8), mode="live")
        self.assertEqual(r.status, "no_currency")

    def test_unidentified_when_classifier_unsure(self):
        r = make_pipeline(accepted=False).analyze(frame_with_note(GOOD), mode="live")
        self.assertEqual((r.status, r.is_currency, r.label), ("unidentified", True, None))

    def test_partial_note(self):
        r = make_pipeline().analyze(frame_with_note([[-40, 140], [300, 150], [295, 300], [-45, 292]]), mode="live")
        self.assertEqual(r.status, "partial_note")

    def test_too_far(self):
        r = make_pipeline().analyze(frame_with_note([[300, 200], [460, 204], [458, 280], [298, 276]]), mode="live")
        self.assertEqual(r.status, "too_far")

    def test_blurry_still(self):
        img = cv2.GaussianBlur(frame_with_note(GOOD), (0, 0), 14)
        r = make_pipeline().analyze_pil(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        self.assertEqual(r.status, "too_blurry")

    def test_tiny_image(self):
        self.assertEqual(make_pipeline().analyze(np.zeros((90, 90, 3), np.uint8)).status, "low_resolution")

    def test_result_dict_shape(self):
        d = make_pipeline(p=0.1).analyze(frame_with_note(GOOD)).to_dict()
        for k in ("is_currency", "label", "name", "confidence", "reason"):
            self.assertIn(k, d)


if __name__ == "__main__":
    unittest.main()
