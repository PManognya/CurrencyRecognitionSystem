"""Live webcam recogniser: note localisation -> currency detector -> ResNet50 classifier -> temporal smoothing.

Keys:  Q quit | S save frame | D debug numbers | R reset | N save a false-positive as a hard negative
       (press N whenever the app wrongly sees a note, then re-train the detector to remove that mistake).
"""
import argparse
import time

import cv2

import config
from pipeline import CurrencyRecognitionPipeline, PipelineNotReady, ResultSmoother, guide_box
from viz import GREEN, ORANGE, RED, WHITE, TextRenderer, draw_guide_box, draw_quad

BAR = {"confirmed": GREEN, "hold_steady": ORANGE, "guidance": ORANGE, "unidentified": ORANGE,
       "no_currency": RED, "searching": WHITE}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    try:
        pipeline = CurrencyRecognitionPipeline()
    except PipelineNotReady as e:
        raise SystemExit(f"\n{e}\n\nTrain the models first (see README) and copy them into model/.")
    smoother, text = ResultSmoother(), TextRenderer()
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit("Could not open the webcam (try --camera 1).")

    debug, frame_id, result, state = False, 0, None, None
    flash, flash_until = "", 0.0
    hard_dir = config.NEGATIVE_DIR / "hard_negatives"
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        H, W = frame.shape[:2]
        box = guide_box(W, H)
        if frame_id % config.PROCESS_EVERY_N_FRAMES == 0 or state is None:
            result = pipeline.analyze(frame, mode="live", guide=box)
            state = smoother.update(result)
        frame_id += 1

        raw = frame.copy()
        canvas = frame
        color = BAR.get(state.kind, WHITE)
        draw_guide_box(canvas, box, color=color)
        if result is not None and result.quad is not None and result.diagnostics.get("source") == "contour":
            draw_quad(canvas, result.quad, GREEN if state.kind == "confirmed" else ORANGE)

        # status panel at the bottom
        panel_h = 150 if state.kind == "confirmed" else 70
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, H - panel_h), (W, H), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, canvas, 0.35, 0, canvas)
        lines = [(f"Status: {state.headline}", f"Status: {state.headline}", 26, color)]
        if state.kind == "confirmed":
            from descriptions import short_name, parse_label, CURRENCIES
            code, _ = parse_label(state.label)
            ascii_name = f"{short_name(state.label)} - {CURRENCIES.get(code, ('', code, ''))[1]}"
            lines += [(state.name, ascii_name, 30, WHITE), (f"Confidence: {state.confidence:.0%}", f"Confidence: {state.confidence:.0%}", 24, WHITE)]
        text.draw(canvas, lines, 20, H - panel_h + 10)
        if state.kind == "confirmed":
            cv2.putText(canvas, state.description[:110], (20, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        if debug and result is not None:
            info = [f"detector p={result.detector_score if result.detector_score is not None else float('nan'):.2f}",
                    f"status={result.status}", f"reason={result.reason[:60]}"] + [f"{k}={v}" for k, v in result.diagnostics.items()]
            if result.top3:
                info.append(f"top1={result.top3[0][0]} {result.top3[0][1]:.2f} margin={result.margin:.2f} ent={result.entropy:.2f}")
            for i, s in enumerate(info):
                cv2.putText(canvas, s, (10, 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        if flash and time.time() < flash_until:
            cv2.putText(canvas, flash, (10, H - panel_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Currency Recognition (Q quit, D debug, N hard-negative)", canvas)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        elif k == ord("d"):
            debug = not debug
        elif k == ord("r"):
            smoother.reset()
        elif k == ord("s"):
            name = f"capture_{int(time.time())}.jpg"
            cv2.imwrite(name, raw)
            flash, flash_until = f"saved {name}", time.time() + 2
        elif k == ord("n"):
            hard_dir.mkdir(parents=True, exist_ok=True)
            stamp = int(time.time() * 1000)
            cv2.imwrite(str(hard_dir / f"frame_{stamp}.jpg"), raw)
            if result is not None and result.roi is not None:
                cv2.imwrite(str(hard_dir / f"roi_{stamp}.jpg"), cv2.cvtColor(result.roi, cv2.COLOR_RGB2BGR))
            flash, flash_until = "saved to negative/hard_negatives", time.time() + 2
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
