"""Entry point for the inspection pipeline bootstrap."""

import os

import cv2
from ultralytics import YOLO

from config.settings import (
    API_KEY,
    API_URL,
    AWS_ACCESS_KEY,
    AWS_SECRET_KEY,
    CAMERA_HEIGHT_VALUE,
    CAMERA_SOURCE,
    CAMERA_WIDTH_VALUE,
    CONFIDENCE_THRESHOLD,
    FPS_WINDOW,
    IMAGES_DIR,
    IOU_THRESHOLD,
    LOG_DIR,
    MODEL_PATH,
    RECONNECT_DELAY,
    SHOW_WINDOW,
    USE_GPU,
)
from core.camera import CameraManager
from core.fps import FpsTracker
from detection.filter import LabelFilter
from detection.parser import DetectionParser
from detection.visualizer import FrameVisualizer
from upload.pipeline import UploadPipeline
from utils.directories import ensure_directories
from utils.logger import setup_logging
from utils.validators import validate_environment

try:
    import torch
except ImportError:  # pragma: no cover - torch may be missing on some deployments
    torch = None


def main() -> None:
    ensure_directories(LOG_DIR, IMAGES_DIR)
    logger = setup_logging(LOG_DIR)

    if os.getenv("STRICT_ENV_VALIDATION", "0") == "1":
        validate_environment(
            API_URL,
            API_KEY,
            AWS_ACCESS_KEY,
            AWS_SECRET_KEY,
            logger,
        )

    logger.info("Starting camera loop source=%s", CAMERA_SOURCE)
    camera = CameraManager(
        CAMERA_SOURCE,
        CAMERA_WIDTH_VALUE,
        CAMERA_HEIGHT_VALUE,
        RECONNECT_DELAY,
        logger=logger,
    )
    fps_tracker = FpsTracker(FPS_WINDOW)
    show_window = SHOW_WINDOW
    window_name = "Camera Preview"
    parser = DetectionParser()
    label_filter = LabelFilter()
    visualizer = FrameVisualizer()
    uploader = UploadPipeline(logger)

    device = "cpu"
    if USE_GPU and torch is not None:
        if torch.cuda.is_available():
            device = "cuda:0"
            try:
                torch.cuda.set_device(0)
                torch.backends.cudnn.benchmark = True
            except Exception as exc:  # pragma: no cover - protective log
                logger.warning("GPU setup failed (%s). Using CPU instead.", exc)
                device = "cpu"
        else:
            logger.warning("CUDA not available. Using CPU.")
    elif USE_GPU and torch is None:
        logger.warning("PyTorch not installed. Using CPU.")

    model = YOLO(MODEL_PATH)
    try:
        model.to(device)
        logger.info("Model loaded on %s", device)
    except Exception as exc:
        logger.warning("Unable to move model to %s: %s", device, exc)

    frame_index = 0
    try:
        while True:
            frame = camera.read_frame()
            if frame is None:
                if camera.is_finished():
                    logger.info("Input stream finished")
                    break
                logger.warning("No frame retrieved from camera. Retrying...")
                continue

            frame_index += 1
            try:
                results = model.track(
                    frame,
                    persist=True,
                    conf=CONFIDENCE_THRESHOLD,
                    iou=IOU_THRESHOLD,
                    device=device,
                    verbose=False,
                )
            except Exception as exc:
                logger.exception("Model inference failed: %s", exc)
                continue

            filtered = []
            if results:
                detections = parser.parse(results[0], model.names)
                filtered = label_filter.filter(detections)

            if filtered:
                frame = visualizer.draw(frame, filtered, inplace=True)
                uploader.process_frame(frame, filtered, frame_index)

            fps = fps_tracker.update()
            if show_window:
                try:
                    cv2.putText(
                        frame,
                        f"FPS: {fps:.2f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow(window_name, frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        logger.info("Exit requested by user (q)")
                        break
                except cv2.error as exc:
                    logger.warning("OpenCV display unavailable: %s", exc)
                    show_window = False
                    try:
                        cv2.destroyAllWindows()
                    except cv2.error:
                        pass
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        uploader.flush()
        camera.release()
        if show_window:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
        logger.info("Camera loop stopped")


if __name__ == "__main__":
    main()
