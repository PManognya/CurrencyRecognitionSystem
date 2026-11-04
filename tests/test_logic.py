"""Offline unit tests (no trained models needed):  python -m unittest discover tests"""
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from inference import assess_probabilities
from note_localizer import NoteLocalizer, quality_issue, whole_image_candidate

NAMES = ["INR_10", "INR_20", "TRY_5", "USD_1"]


def synthetic_note_frame(border_touch=False, size=(480, 640)):
    frame = np.full((*size, 3), (90, 100, 95), np.uint8)
    cv2.randn(frame, (90, 100, 95), (6, 6, 6))
    pts = np.array([[150, 140], [470, 150], [465, 300], [145, 292]], np.int32)
    if border_touch:
        pts = np.array([[-40, 140], [300, 150], [295, 300], [-45, 292]], np.int32)
    cv2.fillConvexPoly(frame, pts, (40, 160, 200))
    for x in range(160, 440, 12):                         # texture so it is not 'blurry'
        cv2.line(frame, (x, 160), (x + 6, 280), (20, 60, 120), 2)
    return frame


class TestAssess(unittest.TestCase):
    def test_confident_accepted(self):
        a = assess_probabilities([0.96, 0.02, 0.01, 0.01], NAMES)
        self.assertTrue(a.accepted)

    def test_flat_distribution_rejected(self):
        a = assess_probabilities([0.3, 0.28, 0.22, 0.2], NAMES)
        self.assertFalse(a.accepted)
        self.assertIn("confidence", a.failed_checks)

    def test_high_top1_but_small_margin_rejected(self):
        a = assess_probabilities([0.50, 0.47, 0.02, 0.01], NAMES)
        self.assertFalse(a.accepted)
        self.assertIn("margin", a.failed_checks)


class TestLocalizer(unittest.TestCase):
    def test_finds_note_shaped_region(self):
        c = NoteLocalizer().find_candidates(synthetic_note_frame())
        self.assertTrue(c)
        self.assertGreater(c[0].roi.shape[1], c[0].roi.shape[0])          # landscape ROI
        self.assertFalse(c[0].touches_border)

    def test_blank_frame_has_no_candidate(self):
        blank = np.full((480, 640, 3), 120, np.uint8)
        self.assertEqual(NoteLocalizer().find_candidates(blank), [])

    def test_partial_note_touches_border(self):
        c = NoteLocalizer().find_candidates(synthetic_note_frame(border_touch=True))
        self.assertTrue(c and c[0].touches_border)

    def test_blur_detected(self):
        sharp = whole_image_candidate(synthetic_note_frame()).roi
        blurry = cv2.GaussianBlur(sharp, (0, 0), 12)
        self.assertEqual(quality_issue(blurry)[0], "too_blurry")
        self.assertNotEqual(quality_issue(sharp)[0], "too_blurry")


class TestSmoother(unittest.TestCase):
    def setUp(self):
        from pipeline import RecognitionResult, ResultSmoother
        self.R, self.S = RecognitionResult, ResultSmoother

    def ok(self, label="INR_500"):
        return self.R("ok", True, label, label, "desc", 0.95)

    def test_needs_stable_frames(self):
        s = self.S(window=7, stable=5, absent=4)
        kinds = [s.update(self.ok()).kind for _ in range(5)]
        self.assertEqual(kinds[:4], ["hold_steady"] * 4)
        self.assertEqual(kinds[4], "confirmed")

    def test_flicker_never_confirms(self):
        s = self.S(window=7, stable=5, absent=4)
        kinds = [s.update(self.ok(l)).kind for l in ["INR_500", "TRY_100", "INR_500", "USD_20", "TRY_100", "INR_500"]]
        self.assertNotIn("confirmed", kinds)

    def test_clears_after_misses(self):
        s = self.S(window=7, stable=5, absent=4)
        for _ in range(6):
            s.update(self.ok())
        for _ in range(4):
            state = s.update(self.R("no_currency"))
        self.assertEqual(state.kind, "no_currency")

    def test_guidance_message(self):
        s = self.S()
        for _ in range(3):
            state = s.update(self.R("too_blurry"))
        self.assertEqual((state.kind, state.headline), ("guidance", "Image too blurry"))


if __name__ == "__main__":
    unittest.main()
