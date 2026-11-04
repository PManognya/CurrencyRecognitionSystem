"""Drawing helpers shared by webcam.py and app.py (OpenCV + Unicode text via Pillow)."""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = ["C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf",
                   "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/Library/Fonts/Arial Unicode.ttf",
                   "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"]
GREEN, ORANGE, RED, WHITE, BLACK = (60, 200, 60), (0, 165, 255), (60, 60, 230), (255, 255, 255), (0, 0, 0)


class TextRenderer:
    """Draws text with ₹ ₺ symbols when a Unicode font exists; otherwise falls back to ASCII."""

    def __init__(self):
        self.font_path = None
        for p in FONT_CANDIDATES:
            try:
                ImageFont.truetype(p, 12)
                self.font_path = p
                break
            except OSError:
                continue
        self._cache = {}

    def _font(self, size):
        if size not in self._cache:
            self._cache[size] = ImageFont.truetype(self.font_path, size)
        return self._cache[size]

    def draw(self, frame_bgr, lines, x, y):
        """lines: [(text, ascii_fallback, size_px, bgr_color)] drawn top to bottom. Returns the new y."""
        if self.font_path is None:
            for text, ascii_text, size, color in lines:
                cv2.putText(frame_bgr, ascii_text, (x, y + size), cv2.FONT_HERSHEY_SIMPLEX, size / 32.0, color, 2, cv2.LINE_AA)
                y += int(size * 1.4)
            return y
        img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(img)
        for text, _, size, color in lines:
            d.text((x, y), text, font=self._font(size), fill=color[::-1])
            y += int(size * 1.4)
        frame_bgr[:] = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        return y


def draw_guide_box(frame, box, color=WHITE, label="PLACE NOTE HERE"):
    x1, y1, x2, y2 = box
    L = max(18, (x2 - x1) // 12)
    for (cx, cy, dx, dy) in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x2, y2, -1, -1), (x1, y2, 1, -1)]:
        cv2.line(frame, (cx, cy), (cx + dx * L, cy), color, 3)
        cv2.line(frame, (cx, cy), (cx, cy + dy * L), color, 3)
    cv2.putText(frame, label, (x1 + (x2 - x1) // 2 - 95, y1 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)


def draw_quad(frame, quad, color=GREEN, thickness=3):
    if quad is not None:
        cv2.polylines(frame, [np.asarray(quad, dtype=np.int32).reshape(-1, 1, 2)], True, color, thickness, cv2.LINE_AA)


def annotate_still(image_rgb, result):
    """RGB image with the detected note outline (green = accepted, orange = guidance, red = rejected)."""
    out = cv2.cvtColor(np.asarray(image_rgb), cv2.COLOR_RGB2BGR).copy()
    if result.quad is not None and result.diagnostics.get("source") == "contour":
        color = GREEN if result.status == "ok" else ORANGE if result.status != "no_currency" else RED
        draw_quad(out, result.quad, color, max(2, out.shape[1] // 250))
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
