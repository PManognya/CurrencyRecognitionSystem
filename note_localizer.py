"""Stage A - classical OpenCV banknote localisation + image-quality checks (no learning involved).

Finds banknote-shaped regions (large, convex, rectangular, ~2:1 aspect) with several edge / threshold maps,
perspective-corrects the best one into a landscape ROI, and reports geometry (area, border contact) used for
guidance messages. This stage only proposes *candidates*: whether a candidate really is a banknote is decided
by the learned detector (currency_detector.py). If no candidate exists, live mode rejects the frame outright.
"""
from dataclasses import dataclass

import cv2
import numpy as np

import config


@dataclass
class NoteCandidate:
    quad: np.ndarray          # (4, 2) float32, frame coordinates, order TL, TR, BR, BL
    roi: np.ndarray           # RGB uint8, rectified landscape crop
    area_ratio: float         # quad area / frame area
    aspect: float             # long side / short side
    short_side_px: float      # short side of the quad in original pixels
    touches_border: bool
    score: float
    source: str               # "contour" | "whole_image" | "guide_box"


def order_quad(pts) -> np.ndarray:
    pts = np.asarray(pts, dtype="float32").reshape(4, 2)
    s, d = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], dtype="float32")


def _side_lengths(q):
    top, bottom = np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[3])
    left, right = np.linalg.norm(q[3] - q[0]), np.linalg.norm(q[2] - q[1])
    return (top + bottom) / 2.0, (left + right) / 2.0


def warp_quad(frame_bgr, quad):
    """Perspective-correct `quad` into a landscape RGB ROI (long side = WARP_LONG_SIDE)."""
    q = order_quad(quad)
    w, h = _side_lengths(q)
    if h > w:                                    # note held vertically -> rotate so the long side is horizontal
        q = np.roll(q, -1, axis=0)
        w, h = h, w
    out_w = config.WARP_LONG_SIDE
    out_h = max(32, int(round(out_w * h / max(w, 1e-6))))
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype="float32")
    matrix = cv2.getPerspectiveTransform(q, dst)
    warped = cv2.warpPerspective(frame_bgr, matrix, (out_w, out_h), flags=cv2.INTER_CUBIC)
    return cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)


class NoteLocalizer:
    def __init__(self):
        self._k3 = np.ones((3, 3), np.uint8)
        self._k9 = np.ones((9, 9), np.uint8)

    def _masks(self, small):
        gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        sat = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[..., 1], (5, 5), 0)
        masks = [cv2.morphologyEx(cv2.dilate(cv2.Canny(gray, 40, 120), self._k3), cv2.MORPH_CLOSE, self._k9)]
        for channel in (gray, sat):
            _, m = cv2.threshold(channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            for mm in (m, 255 - m):
                masks.append(cv2.morphologyEx(mm, cv2.MORPH_OPEN, self._k3))
                masks[-1] = cv2.morphologyEx(masks[-1], cv2.MORPH_CLOSE, self._k9)
        return masks

    def find_candidates(self, frame_bgr, max_candidates=3):
        H, W = frame_bgr.shape[:2]
        scale = min(1.0, 640.0 / max(H, W))
        small = cv2.resize(frame_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else frame_bgr
        sh, sw = small.shape[:2]
        frame_area = float(sh * sw)
        margin = max(2.0, config.BORDER_MARGIN_RATIO * min(sh, sw))
        found = []
        for mask in self._masks(small):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                if not (config.CANDIDATE_MIN_AREA_RATIO * frame_area <= cv2.contourArea(c) <= 0.97 * frame_area):
                    continue
                hull = cv2.convexHull(c)
                hull_area = cv2.contourArea(hull)
                (_, _), (rw, rh), _ = rect = cv2.minAreaRect(hull)
                if min(rw, rh) < 8:
                    continue
                rectangularity = hull_area / (rw * rh)
                if rectangularity < config.RECTANGULARITY_MIN:
                    continue
                approx = cv2.approxPolyDP(hull, 0.03 * cv2.arcLength(hull, True), True)
                quad = approx.reshape(4, 2) if len(approx) == 4 and cv2.isContourConvex(approx) else cv2.boxPoints(rect)
                quad = order_quad(quad)
                a, b = _side_lengths(quad)
                aspect = max(a, b) / max(min(a, b), 1e-6)
                if not (config.NOTE_ASPECT_RANGE[0] <= aspect <= config.NOTE_ASPECT_RANGE[1]):
                    continue
                area_ratio = float(cv2.contourArea(quad) / frame_area)
                if area_ratio < config.CANDIDATE_MIN_AREA_RATIO:
                    continue
                pts = hull.reshape(-1, 2)
                touches = bool(np.any(pts[:, 0] <= margin) or np.any(pts[:, 0] >= sw - 1 - margin)
                               or np.any(pts[:, 1] <= margin) or np.any(pts[:, 1] >= sh - 1 - margin))
                score = rectangularity * np.exp(-(((aspect - 2.2) / 0.8) ** 2)) * np.sqrt(area_ratio)
                found.append((score, quad, area_ratio, aspect, min(a, b), touches))
        found.sort(key=lambda t: t[0], reverse=True)
        picked = []
        for score, quad, area_ratio, aspect, short_side, touches in found:
            center = quad.mean(axis=0)
            if any(np.linalg.norm(center - p[1].mean(axis=0)) < 0.1 * max(sw, sh) and abs(area_ratio - p[2]) < 0.15 for p in picked):
                continue                                          # near-duplicate of a better candidate
            picked.append((score, quad, area_ratio, aspect, short_side, touches))
            if len(picked) >= max_candidates:
                break
        out = []
        for score, quad, area_ratio, aspect, short_side, touches in picked:
            full_quad = (quad / scale).astype("float32")
            out.append(NoteCandidate(full_quad, warp_quad(frame_bgr, full_quad), area_ratio, float(aspect),
                                     float(short_side / scale), touches, float(score), "contour"))
        return out


def whole_image_candidate(frame_bgr, source="whole_image") -> NoteCandidate:
    """Treat the entire image as the ROI (used for uploaded stills that are already cropped notes)."""
    H, W = frame_bgr.shape[:2]
    quad = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype="float32")
    return NoteCandidate(quad, cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), 1.0, max(H, W) / max(min(H, W), 1),
                         float(min(H, W)), False, 0.0, source)


def guide_box_candidate(frame_bgr, box) -> NoteCandidate:
    x1, y1, x2, y2 = box
    crop = frame_bgr[y1:y2, x1:x2]
    H, W = frame_bgr.shape[:2]
    quad = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype="float32")
    return NoteCandidate(quad, cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), float((x2 - x1) * (y2 - y1) / (H * W)),
                         (x2 - x1) / max(y2 - y1, 1), float(min(x2 - x1, y2 - y1)), False, 0.0, "guide_box")


# ----------------------------------------------------------------------------- image quality
def blur_score(rgb) -> float:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    scale = 400.0 / gray.shape[1]
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness(rgb) -> float:
    return float(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).mean())


def quality_issue(rgb):
    """Returns (issue_status | None, metrics dict)."""
    metrics = {"blur": blur_score(rgb), "brightness": brightness(rgb)}
    if metrics["blur"] < config.BLUR_THRESHOLD:
        return "too_blurry", metrics
    if metrics["brightness"] < config.BRIGHTNESS_RANGE[0]:
        return "too_dark", metrics
    if metrics["brightness"] > config.BRIGHTNESS_RANGE[1]:
        return "too_bright", metrics
    return None, metrics
