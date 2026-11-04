"""Capture training photos from your webcam straight into the right folder.

Usage:
    python capture_images.py negative/face          # NOT-currency: your face
    python capture_images.py negative/background    # your room, desk, wall ...
    python capture_images.py negative/random_objects
    python capture_images.py INR_500                # real notes of that class (front AND back, different holds)

Keys:  A = start/stop auto-capture (every 0.4 s)   SPACE = one photo   Q = quit
Tips: vary light, angle, distance and background. For negatives also hold paper, cards, a phone, a book -
rectangular things are what most often fool the system.  Aim for 150+ photos per negative group.
"""
import sys
import time

import cv2

import config

name = sys.argv[1] if len(sys.argv) > 1 else "negative/face"
name = name.replace("\\", "/")
out = (config.NEGATIVE_DIR / name.split("/", 1)[1]) if name.startswith("negative/") else (config.DATASET_DIR / name)
out.mkdir(parents=True, exist_ok=True)
n = len(list(out.glob("*.jpg")))
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Could not open webcam.")

auto, last = False, 0.0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    now = time.time()
    save = False
    if auto and now - last > 0.4:
        save, last = True, now
    k = cv2.waitKey(1) & 0xFF
    if k == ord(" "):
        save = True
    elif k == ord("a"):
        auto = not auto
    elif k == ord("q"):
        break
    if save:
        cv2.imwrite(str(out / f"cam_{int(now * 1000)}.jpg"), frame)
        n += 1
    view = frame.copy()
    cv2.putText(view, f"{name}  saved: {n}  auto: {'ON' if auto else 'off'}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    cv2.imshow("Capture (A auto, SPACE one, Q quit)", view)
cap.release(); cv2.destroyAllWindows()
print(f"{n} images in {out}")
