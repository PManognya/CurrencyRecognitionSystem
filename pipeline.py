"""CurrencyRecognitionPipeline - the single inference entry point used by webcam.py, app.py and predict.py.

    frame -> [A] note localisation (OpenCV) -> [B] currency / non-currency detector -> geometry + quality checks
          -> [C] ResNet50 denomination classifier -> open-set check (confidence + margin + entropy) -> result

"No currency detected" is a first-class result: it is produced whenever there is no note-shaped region (live mode)
or the learned detector does not accept the region as a banknote. A denomination is only ever reported for
frames that pass every stage.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageOps

import config
from currency_detector import CurrencyDetector
from descriptions import CURRENCIES, display_name, get_description, parse_label, short_name
from inference import DenominationClassifier
from note_localizer import (NoteLocalizer, guide_box_candidate, quality_issue, whole_image_candidate)

STATUS_TEXT = {
    "ok": "Currency detected",
    "no_currency": "No currency detected",
    "unidentified": "Unidentified note",
    "too_blurry": "Image too blurry",
    "too_far": "Move the note closer",
    "partial_note": "Place the complete note inside the box",
    "too_dark": "Too dark - add more light",
    "too_bright": "Too bright - reduce glare",
    "low_resolution": "Image resolution too low",
}


class PipelineNotReady(RuntimeError):
    """Raised when trained model files are missing."""


@dataclass
class RecognitionResult:
    status: str
    is_currency: bool = False
    label: Optional[str] = None
    name: str = "Unknown"
    description: str = ""
    confidence: float = 0.0           # confidence in the reported verdict (see README)
    reason: str = ""
    detector_score: Optional[float] = None
    margin: Optional[float] = None
    entropy: Optional[float] = None
    top3: list = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)
    quad: Optional[np.ndarray] = None
    roi: Optional[np.ndarray] = None

    @property
    def message(self):
        return STATUS_TEXT.get(self.status, self.status)

    @property
    def short(self):
        return short_name(self.label) if self.label else "Unknown"

    def to_dict(self):
        return {"status": self.status, "is_currency": self.is_currency, "label": self.label, "name": self.name,
                "description": self.description, "confidence": round(self.confidence, 4), "reason": self.reason,
                "detector_score": None if self.detector_score is None else round(self.detector_score, 4),
                "margin": self.margin, "entropy": self.entropy, "top3": self.top3, "diagnostics": self.diagnostics,
                "quad": None if self.quad is None else self.quad.round(1).tolist()}


def guide_box(width: int, height: int):
    """Central guide rectangle (x1, y1, x2, y2) shown in the webcam UI."""
    w = int(width * config.GUIDE_WIDTH_FRAC)
    h = int(w / config.GUIDE_ASPECT)
    if h > 0.8 * height:
        h = int(0.8 * height)
        w = int(h * config.GUIDE_ASPECT)
    x1, y1 = (width - w) // 2, (height - h) // 2
    return x1, y1, x1 + w, y1 + h


class CurrencyRecognitionPipeline:
    def __init__(self):
        try:
            self.detector = CurrencyDetector()
            self.classifier = DenominationClassifier()
        except FileNotFoundError as e:
            raise PipelineNotReady(str(e)) from e
        self.localizer = NoteLocalizer()
        self.supported = sorted({parse_label(n)[0] for n in self.classifier.names})

    # ------------------------------------------------------------------ public API
    def analyze_pil(self, img: Image.Image) -> RecognitionResult:
        """Uploaded / captured still image (Streamlit, CLI)."""
        img = ImageOps.exif_transpose(img).convert("RGB")
        bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        return self.analyze(bgr, mode="still")

    def analyze(self, frame_bgr: np.ndarray, mode: str = "live", guide=None) -> RecognitionResult:
        """mode='live': strict - needs a note-shaped region, complete and close enough.
           mode='still': also considers the whole image as the note (uploads that are already cropped)."""
        H, W = frame_bgr.shape[:2]
        if min(H, W) < config.MIN_IMAGE_SIDE:
            return RecognitionResult("low_resolution", reason=f"shorter side {min(H, W)}px < {config.MIN_IMAGE_SIDE}px")

        candidates = self.localizer.find_candidates(frame_bgr)
        options = candidates[:1]
        if mode == "still":
            options.append(whole_image_candidate(frame_bgr))
        elif not options and config.ALLOW_GUIDE_BOX_FALLBACK and guide is not None:
            options.append(guide_box_candidate(frame_bgr, guide))
        if not options:
            return RecognitionResult("no_currency", confidence=0.0, reason="no banknote-shaped region found")

        scored = [(self.detector.score_rgb(c.roi), c) for c in options]
        p_currency, cand = max(scored, key=lambda t: t[0])
        diag = {"area_ratio": round(cand.area_ratio, 3), "aspect": round(cand.aspect, 2), "source": cand.source}
        base = dict(detector_score=p_currency, diagnostics=diag, quad=cand.quad, roi=cand.roi)

        if p_currency < config.DETECTOR_SOFT_THRESHOLD:
            return RecognitionResult("no_currency", confidence=1.0 - p_currency,
                                     reason=f"detector score {p_currency:.2f} < {config.DETECTOR_SOFT_THRESHOLD}", **base)

        issue = None
        if mode == "live":
            if cand.touches_border:
                issue = "partial_note"
            elif cand.area_ratio < config.NOTE_MIN_AREA_RATIO:
                issue = "too_far"
        q_issue, metrics = quality_issue(cand.roi)
        diag.update({k: round(v, 1) for k, v in metrics.items()})
        issue = issue or q_issue
        if issue:
            return RecognitionResult(issue, confidence=p_currency, reason=STATUS_TEXT[issue], **base)

        if p_currency < config.DETECTOR_THRESHOLD:
            return RecognitionResult("no_currency", confidence=1.0 - p_currency,
                                     reason=f"detector score {p_currency:.2f} < {config.DETECTOR_THRESHOLD}", **base)

        a = self.classifier.assess(cand.roi)
        base.update(margin=a.margin, entropy=a.entropy, top3=a.top3)
        if not a.accepted:
            supported = ", ".join(self.supported)
            return RecognitionResult("unidentified", is_currency=True, confidence=a.confidence,
                                     reason=f"classifier rejected ({', '.join(a.failed_checks)}); supported: {supported}",
                                     description=f"A banknote was found but it is not one of the supported notes ({supported}).",
                                     **base)
        return RecognitionResult("ok", is_currency=True, label=a.label, name=display_name(a.label),
                                 description=get_description(a.label), confidence=a.confidence,
                                 reason="all checks passed", **base)


# ============================================================================== temporal smoothing
@dataclass
class DisplayState:
    kind: str                       # searching | no_currency | hold_steady | guidance | unidentified | confirmed
    headline: str
    label: Optional[str] = None
    name: str = ""
    description: str = ""
    confidence: float = 0.0


class ResultSmoother:
    """Stops the display flickering: a denomination is shown only after the same label was seen
    `stable` times within the last `window` analysed frames, and is cleared after `absent` misses."""

    def __init__(self, window=config.SMOOTH_WINDOW, stable=config.SMOOTH_STABLE, absent=config.SMOOTH_ABSENT):
        self.window, self.stable, self.absent = window, stable, absent
        self.history = deque(maxlen=window)
        self.confirmed: Optional[str] = None

    def reset(self):
        self.history.clear()
        self.confirmed = None

    def update(self, r: RecognitionResult) -> DisplayState:
        self.history.append(r)
        recent = list(self.history)
        ok = [x for x in recent if x.status == "ok"]
        counts = Counter(x.label for x in ok)
        tail = recent[-self.absent:]

        if self.confirmed and not any(x.status == "ok" and x.label == self.confirmed for x in tail):
            self.confirmed = None
        if counts:
            best, n = counts.most_common(1)[0]
            if n >= self.stable and best != self.confirmed and n > counts.get(self.confirmed, 0):
                self.confirmed = best

        if self.confirmed:
            hits = [x for x in ok if x.label == self.confirmed]
            last = hits[-1]
            return DisplayState("confirmed", "Currency detected", self.confirmed, last.name, last.description,
                                float(np.mean([x.confidence for x in hits])))
        if any(x.status == "ok" for x in tail):          # recent evidence only - stale frames must not linger
            return DisplayState("hold_steady", "Hold the note steady...")

        statuses = [x.status for x in tail]
        if len(tail) >= self.absent and all(s == "no_currency" for s in statuses):
            return DisplayState("no_currency", STATUS_TEXT["no_currency"])
        last3 = [x.status for x in recent[-3:]]
        if len(last3) == 3 and len(set(last3)) == 1:
            s = last3[0]
            if s == "unidentified":
                return DisplayState("unidentified", STATUS_TEXT[s], description=recent[-1].description)
            if s != "no_currency":
                return DisplayState("guidance", STATUS_TEXT.get(s, s))
        if recent and recent[-1].status == "no_currency":
            return DisplayState("no_currency", STATUS_TEXT["no_currency"])
        return DisplayState("searching", "Looking for a note...")
