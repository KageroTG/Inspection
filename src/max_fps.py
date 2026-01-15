import os
import time

import cv2
from ultralytics import YOLO

try:
    import torch
except ImportError:  # pragma: no cover - torch may be missing in some setups
    torch = None


MODEL_PATH = os.getenv("MODEL_PATH", "best.pt")
VIDEO_SOURCE_RAW = os.getenv("VIDEO_SOURCE", "Ipoh to KL - 15minutes.mp4")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.4"))
SHOW_WINDOW = os.getenv("SHOW_WINDOW", "1") == "1"


def _parse_source(raw: str):
    try:
        return int(raw)
    except ValueError:
        return raw


def _select_device() -> str:
    if torch is None:
        return "cpu"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _draw_fps(frame, fps: float) -> None:
    text = f"FPS: {fps:.1f}"
    cv2.putText(
        frame,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    source = _parse_source(VIDEO_SOURCE_RAW)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {source}")

    model = YOLO(MODEL_PATH)
    device = _select_device()
    if device != "cpu":
        try:
            model.to(device)
        except Exception:
            device = "cpu"

    last_time = time.perf_counter()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        _ = model(
            frame,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            device=device,
            verbose=False,
        )

        now = time.perf_counter()
        delta = now - last_time
        if delta > 0:
            fps = 1.0 / delta
        last_time = now

        _draw_fps(frame, fps)

        if SHOW_WINDOW:
            cv2.imshow("Max FPS", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if SHOW_WINDOW:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
