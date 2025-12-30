import os
import sys
import time
import threading
import logging
import json
import random
from datetime import timezone, timedelta, datetime
from typing import Optional, Dict, Any, List, Callable
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
import cv2
import boto3
import requests
from ultralytics import YOLO
from colorama import init, Fore

try:
    import torch
except ImportError:  # pragma: no cover - torch may be missing on some deployments
    torch = None


load_dotenv()
init(autoreset=True)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LOG_DIR = "logs"
IMAGES_DIR = "images"
MODEL_PATH = os.getenv("MODEL_PATH", "best.pt")
CAMERA_SOURCE_RAW = os.getenv("CAMERA_SOURCE", "Ipoh to KL - 15minutes.mp4")
CAMERA_WIDTH = os.getenv("CAMERA_WIDTH")
CAMERA_HEIGHT = os.getenv("CAMERA_HEIGHT")
RECONNECT_DELAY = float(os.getenv("CAMERA_RECONNECT_DELAY", "1.5"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.4"))
UPLOAD_WORKERS = max(1, int(os.getenv("UPLOAD_WORKERS", "4")))
CRACK_UPLOAD_DELAY = float(os.getenv("CRACK_UPLOAD_DELAY", "5.0"))
PERF_LOG_INTERVAL = float(os.getenv("PERF_LOG_INTERVAL", "5.0"))
SHOW_WINDOW = os.getenv("SHOW_WINDOW", "1") == "1"

# Detection configuration
COMPANY_ID = os.getenv("ROBOCOMPANY_ID", "1005")
LOCATION = os.getenv("ROBOLOCATION", "1111110000.1551331")
DETECTION_TYPE = os.getenv("ROBODETECTION_TYPE", "RoAd")


def _parse_label_set(raw: Optional[str], fallback: List[str]) -> set:
    if not raw:
        return {item.strip().lower() for item in fallback if item.strip()}
    return {item.strip().lower() for item in raw.split(",") if item.strip()}


IMMEDIATE_UPLOAD_LABELS = _parse_label_set(
    os.getenv("PRIORITY_CLASSES"),
    ["pothole", "potholes", "raveling", "stagnant_water"],
)
CRACK_LABELS = _parse_label_set(
    os.getenv("CRACK_CLASSES"),
    ["crack", "cracks"],
)
INTERESTING_LABELS = IMMEDIATE_UPLOAD_LABELS | CRACK_LABELS


# API / AWS configuration
API_URL = os.getenv("ROBOLYZE_API_URL")
API_KEY = os.getenv("ROBOLYZE_API_KEY")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_BUCKET = os.getenv("AWS_BUCKET", "robolyzedatamy")
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-5")
AWS_BASE_URL = os.getenv(
    "AWS_BASE_URL",
    f"https://{AWS_BUCKET}.s3.{AWS_REGION}.amazonaws.com",
)


# GPU configuration
USE_GPU = os.getenv("USE_GPU", "1") == "1"
DEVICE: Optional[str]
if USE_GPU and torch is not None:
    if torch.cuda.is_available():
        DEVICE = "cuda:0"
        try:
            torch.cuda.set_device(0)
            torch.backends.cudnn.benchmark = True
            print(Fore.GREEN + f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
        except Exception as exc:  # pragma: no cover - protective log
            print(Fore.YELLOW + f"GPU setup failed ({exc}). Using CPU instead.")
            DEVICE = "cpu"
    else:
        print(Fore.YELLOW + "CUDA not available. Using CPU.")
        DEVICE = "cpu"
else:
    if USE_GPU and torch is None:
        print(Fore.YELLOW + "PyTorch not installed. Using CPU.")
    DEVICE = "cpu"

print(Fore.CYAN + f"Device set to: {DEVICE}\n")


# ---------------------------------------------------------------------------
# Logging setup and directory creation
# ---------------------------------------------------------------------------
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, f"edge_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")),
        logging.StreamHandler(sys.stdout),
    ],
)
LOGGER = logging.getLogger("edge_inference")


# ---------------------------------------------------------------------------
# Environment validation
# ---------------------------------------------------------------------------
if not API_URL or not API_KEY:
    LOGGER.error("Missing ROBOLYZE_API_URL or ROBOLYZE_API_KEY in environment")
    sys.exit(1)

if not AWS_ACCESS_KEY or not AWS_SECRET_KEY:
    LOGGER.error("Missing AWS credentials in environment")
    sys.exit(1)


# ---------------------------------------------------------------------------
# AWS client + time helpers
# ---------------------------------------------------------------------------
try:
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION,
    )
    LOGGER.info("✓ S3 client ready for bucket %s", AWS_BUCKET)
except Exception as exc:
    LOGGER.error("Failed to create S3 client: %s", exc)
    sys.exit(1)

MYT = timezone(timedelta(hours=8))


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _parse_camera_source(raw: str):
    try:
        return int(raw)
    except ValueError:
        return raw


def _parse_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


CAMERA_SOURCE = _parse_camera_source(CAMERA_SOURCE_RAW)
CAMERA_WIDTH_VALUE = _parse_int(CAMERA_WIDTH)
CAMERA_HEIGHT_VALUE = _parse_int(CAMERA_HEIGHT)


class UploadManager:
    """Prevent duplicate uploads for the same tracked object."""

    def __init__(self) -> None:
        self._seen = set()
        self._lock = threading.Lock()

    def should_upload(self, track_id: Optional[int]) -> bool:
        if track_id is None:
            return False
        with self._lock:
            if track_id in self._seen:
                LOGGER.debug("Duplicate suppressed track_id=%s", track_id)
                return False
            self._seen.add(track_id)
            return True


upload_manager = UploadManager()


class CameraStream:
    """Thin wrapper around cv2.VideoCapture that supports files and live cameras."""

    def __init__(self, source, width: Optional[int], height: Optional[int], reconnect_delay: float):
        self.source = source
        self.width = width
        self.height = height
        self.reconnect_delay = reconnect_delay
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_video_file = isinstance(source, str) and os.path.isfile(source)
        self._finished = False
        self._open()

    def _open(self) -> None:
        if self.cap:
            self.cap.release()
        self._finished = False
        self.cap = cv2.VideoCapture(self.source)
        if self.width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to open camera source: {self.source}")
        LOGGER.info(
            "Stream ready source=%s resolution=%sx%s",
            self.source,
            self.width or "auto",
            self.height or "auto",
        )

    def read(self) -> Optional[Any]:
        if self.cap is None:
            self._open()
        if self._finished:
            return None
        ret, frame = self.cap.read()
        if ret:
            return frame
        if self.is_video_file:
            LOGGER.info("Video source %s finished", self.source)
            self._finished = True
            return None
        LOGGER.warning("Camera frame missing. Reconnecting in %.1fs", self.reconnect_delay)
        time.sleep(self.reconnect_delay)
        try:
            self._open()
        except Exception as exc:
            LOGGER.error("Failed to reopen camera: %s", exc)
            return None
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self) -> None:
        if self.cap:
            self.cap.release()
            self.cap = None
            LOGGER.info("Camera stream released")

    def is_finished(self) -> bool:
        return self._finished


class PerformanceMonitor:
    """Track FPS and log periodically."""

    def __init__(self, log_interval: float):
        self.total_frames = 0
        self.current_fps = 0.0
        self._window_frames = 0
        self._last_fps_time = time.time()
        self._last_log_time = time.time()
        self.log_interval = log_interval

    def next_frame_index(self) -> int:
        return self.total_frames + 1

    def update(self, detections: int) -> None:
        self.total_frames += 1
        self._window_frames += 1
        now = time.time()
        elapsed = now - self._last_fps_time
        if elapsed >= 1.0:
            self.current_fps = self._window_frames / elapsed
            self._window_frames = 0
            self._last_fps_time = now
        if now - self._last_log_time >= self.log_interval:
            LOGGER.info(
                "PERF frames=%s fps=%.2f detections_this_frame=%s",
                self.total_frames,
                self.current_fps,
                detections,
            )
            self._last_log_time = now


class CrackBatcher:
    """Hold latest crack detection for a short delay before uploading."""

    def __init__(self, delay_seconds: float, upload_callback: Callable[[Any, Dict[str, Any], int, bool], None]):
        self.delay_seconds = delay_seconds
        self._upload_callback = upload_callback
        self._lock = threading.Lock()
        self._timer: Optional[threading.Timer] = None
        self._payload: Optional[tuple] = None

    def add_detection(self, frame, detection: Dict[str, Any], frame_count: int) -> None:
        with self._lock:
            self._payload = (frame.copy(), detection.copy(), frame_count)
            if self._timer:
                self._timer.cancel()
            self._timer = threading.Timer(self.delay_seconds, self._flush)
            self._timer.daemon = True
            self._timer.start()
            LOGGER.info(
                "CRACK_BUFFER label=%s track_id=%s delay=%.1fs",
                detection.get("class_name"),
                detection.get("track_id"),
                self.delay_seconds,
            )

    def _flush(self) -> None:
        with self._lock:
            payload = self._payload
            self._payload = None
            self._timer = None
        if not payload:
            return
        frame, detection, frame_count = payload
        LOGGER.info(
            "CRACK_FLUSH frame=%s label=%s track_id=%s",
            frame_count,
            detection.get("class_name"),
            detection.get("track_id"),
        )
        self._upload_callback(frame, detection, frame_count, True)

    def shutdown(self) -> None:
        with self._lock:
            if self._timer:
                self._timer.cancel()
                self._timer = None
            payload = self._payload
            self._payload = None
        if payload:
            frame, detection, frame_count = payload
            LOGGER.info(
                "CRACK_SHUTDOWN_FLUSH frame=%s label=%s track_id=%s",
                frame_count,
                detection.get("class_name"),
                detection.get("track_id"),
            )
            self._upload_callback(frame, detection, frame_count, True)


def normalize_label(name: Optional[str]) -> str:
    return (name or "").strip().lower()


def delete_file_safely(path: Optional[str]) -> None:
    if not path:
        return
    try:
        if os.path.exists(path):
            os.remove(path)
            LOGGER.debug("Deleted temporary file %s", path)
    except Exception as exc:
        LOGGER.warning("Failed to delete %s: %s", path, exc)


def persist_frame_to_s3(frame, class_name: str, frame_count: int) -> Optional[Dict[str, str]]:
    now = datetime.now(MYT)
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    unique_suffix = f"{now.microsecond}_{random.randint(1000, 9999)}"
    filename = f"{class_name}-frame{frame_count}-{timestamp}-{unique_suffix}.jpg"
    local_path = os.path.join(IMAGES_DIR, filename)
    if not cv2.imwrite(local_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85]):
        LOGGER.error("Failed to save frame locally for %s", class_name)
        return None
    try:
        s3_client.upload_file(local_path, AWS_BUCKET, filename)
        LOGGER.info("S3_UPLOAD label=%s key=%s", class_name, filename)
    except Exception as exc:
        LOGGER.error("S3 upload failed for %s: %s", filename, exc)
        delete_file_safely(local_path)
        return None
    return {
        "local_path": local_path,
        "s3_key": filename,
        "url": f"{AWS_BASE_URL}/{filename}",
    }


def post_to_dashboard(
    payload: Dict[str, Any],
    track_id: Optional[int],
    local_path: Optional[str],
    retries: int = 3,
) -> bool:
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY,
    }
    for attempt in range(1, retries + 1):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=6)
            if response.status_code in (200, 201):
                LOGGER.info(
                    "API_POST success attempt=%s track_id=%s status=%s",
                    attempt,
                    track_id,
                    response.status_code,
                )
                delete_file_safely(local_path)
                return True
            LOGGER.warning(
                "API_POST unexpected status attempt=%s status=%s track_id=%s",
                attempt,
                response.status_code,
                track_id,
            )
        except Exception as exc:
            LOGGER.warning(
                "API_POST attempt=%s failed track_id=%s error=%s",
                attempt,
                track_id,
                exc,
            )
            time.sleep(1)
    error_file = f"failed_upload_{int(time.time())}.json"
    with open(error_file, "w", encoding="utf-8") as handle:
        json.dump({"payload": payload, "track_id": track_id}, handle)
    LOGGER.error("API_POST giving up track_id=%s saved=%s", track_id, error_file)
    return False


def build_payload(class_name: str, image_url: str, frame_count: int, track_id: Optional[int]) -> Dict[str, Any]:
    detection_id = str(track_id) if track_id is not None else str(frame_count)
    return {
        "did": detection_id,
        "type": DETECTION_TYPE,
        "detect": class_name,
        "image": image_url,
        "location": LOCATION,
        "company_id": COMPANY_ID,
    }


def upload_detection(frame, detection: Dict[str, Any], frame_count: int) -> None:
    class_name = detection.get("class_name", "unknown")
    track_id = detection.get("track_id")
    if not upload_manager.should_upload(track_id):
        return
    image_info = persist_frame_to_s3(frame, class_name, frame_count)
    if not image_info:
        return
    payload = build_payload(class_name, image_info["url"], frame_count, track_id)
    post_to_dashboard(payload, track_id, image_info["local_path"])


def submit_upload(
    executor: ThreadPoolExecutor,
    frame,
    detection: Dict[str, Any],
    frame_count: int,
    already_copied: bool = False,
) -> None:
    if executor is None:
        upload_detection(frame, detection, frame_count)
        return
    payload_frame = frame if already_copied else frame.copy()
    payload_detection = detection if already_copied else detection.copy()
    executor.submit(_upload_worker, payload_frame, payload_detection, frame_count)


def _upload_worker(frame, detection: Dict[str, Any], frame_count: int) -> None:
    try:
        upload_detection(frame, detection, frame_count)
    except Exception as exc:
        LOGGER.exception(
            "UPLOAD_WORKER_ERROR label=%s track_id=%s error=%s",
            detection.get("class_name"),
            detection.get("track_id"),
            exc,
        )


def extract_detections(result, names: Dict[int, str]) -> List[Dict[str, Any]]:
    detections: List[Dict[str, Any]] = []
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.cls is None:
        return detections
    xyxy_list = boxes.xyxy
    cls_list = boxes.cls
    conf_list = getattr(boxes, "conf", None)
    track_list = getattr(boxes, "id", None)
    for idx in range(len(cls_list)):
        class_id = int(cls_list[idx])
        class_name = names.get(class_id, str(class_id))
        detection = {
            "class_name": class_name,
            "confidence": float(conf_list[idx]) if conf_list is not None else 0.0,
            "bbox": [float(v) for v in xyxy_list[idx]],
        }
        if track_list is not None and idx < len(track_list) and track_list[idx] is not None:
            detection["track_id"] = int(track_list[idx])
        detections.append(detection)
    return detections


def draw_detection(frame, detection: Dict[str, Any]) -> None:
    bbox = detection.get("bbox")
    if not bbox or len(bbox) != 4:
        return
    x1, y1, x2, y2 = [int(b) for b in bbox]
    x1 = max(0, min(frame.shape[1] - 1, x1))
    y1 = max(0, min(frame.shape[0] - 1, y1))
    x2 = max(0, min(frame.shape[1] - 1, x2))
    y2 = max(0, min(frame.shape[0] - 1, y2))
    color = (0, 255, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
    class_name = detection.get("class_name", "object")
    confidence = detection.get("confidence")
    track_id = detection.get("track_id")
    label_text = class_name
    if confidence is not None:
        label_text += f" {confidence:.2f}"
    if track_id is not None:
        label_text += f" #{track_id}"
    (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    top_left = (x1, max(y1 - text_h - baseline - 4, 0))
    bottom_right = (x1 + text_w + 6, top_left[1] + text_h + baseline + 4)
    cv2.rectangle(frame, top_left, bottom_right, color, -1)
    cv2.putText(
        frame,
        label_text,
        (top_left[0] + 3, top_left[1] + text_h),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )


def run_inference():
    print(Fore.CYAN + "Starting edge inference pipeline\n")
    LOGGER.info("Starting edge inference pipeline")

    camera = CameraStream(CAMERA_SOURCE, CAMERA_WIDTH_VALUE, CAMERA_HEIGHT_VALUE, RECONNECT_DELAY)
    model = YOLO(MODEL_PATH)
    try:
        model.to(DEVICE)
        LOGGER.info("Model loaded on %s", DEVICE)
    except Exception as exc:
        LOGGER.warning("Unable to move model to %s: %s", DEVICE, exc)

    executor = ThreadPoolExecutor(max_workers=max(2, UPLOAD_WORKERS))
    perf = PerformanceMonitor(PERF_LOG_INTERVAL)
    crack_batcher = CrackBatcher(
        CRACK_UPLOAD_DELAY,
        lambda frame, detection, frame_idx, copied: submit_upload(
            executor, frame, detection, frame_idx, already_copied=copied
        ),
    )
    show_window = SHOW_WINDOW
    window_name = "Edge Detection"

    try:
        while True:
            frame = camera.read()
            if frame is None:
                if camera.is_finished():
                    LOGGER.info("Input stream finished")
                    break
                LOGGER.warning("No frame retrieved from camera. Retrying...")
                continue

            frame_index = perf.next_frame_index()
            try:
                results = model.track(
                    frame,
                    persist=True,
                    conf=CONFIDENCE_THRESHOLD,
                    iou=IOU_THRESHOLD,
                    device=DEVICE,
                    verbose=False,
                )
            except Exception as exc:
                LOGGER.exception("Model inference failed: %s", exc)
                time.sleep(0.05)
                continue

            detections = extract_detections(results[0], model.names)
            relevant_count = 0
            frame_for_uploads = None

            for detection in detections:
                label = normalize_label(detection.get("class_name"))
                if label not in INTERESTING_LABELS:
                    continue
                track_id = detection.get("track_id")
                if track_id is None:
                    LOGGER.debug(
                        "Skipping %s on frame %s because track id unavailable",
                        label,
                        frame_index,
                    )
                    continue
                if frame_for_uploads is None:
                    frame_for_uploads = frame.copy()
                draw_detection(frame_for_uploads, detection)
                relevant_count += 1
                confidence = detection.get("confidence", 0.0)
                LOGGER.info(
                    "DETECTION frame=%s label=%s conf=%.2f track_id=%s",
                    frame_index,
                    label,
                    confidence,
                    track_id,
                )
                if label in CRACK_LABELS:
                    crack_batcher.add_detection(frame_for_uploads, detection, frame_index)
                else:
                    submit_upload(executor, frame_for_uploads, detection, frame_index)

            perf.update(relevant_count)

            frame_to_show = frame_for_uploads if frame_for_uploads is not None else frame
            if show_window:
                try:
                    cv2.imshow(window_name, frame_to_show)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        LOGGER.info("Exit requested by user (q)")
                        break
                except cv2.error as exc:
                    LOGGER.warning("OpenCV display unavailable: %s", exc)
                    show_window = False
                    try:
                        cv2.destroyAllWindows()
                    except cv2.error:
                        pass

    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user")
    finally:
        crack_batcher.shutdown()
        executor.shutdown(wait=True)
        camera.release()
        if show_window:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
        print(Fore.GREEN + "\nEdge inference pipeline stopped\n")
        LOGGER.info("Shutdown complete")


if __name__ == "__main__":
    run_inference()
