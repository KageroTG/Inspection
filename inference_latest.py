import os
import sys
import time
import queue
import threading
import logging
import json
import concurrent.futures
import random
from datetime import timezone, timedelta, datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from dotenv import load_dotenv

import cv2
import boto3  # type: ignore
import requests  # type: ignore
from ultralytics import YOLO
from colorama import init, Fore

try:
    import torch
except ImportError:
    torch = None


load_dotenv()

print("ENV CHECK START")
print("AWS_KEY =", os.environ.get("AWS_ACCESS_KEY_ID"))
print("AWS_SECRET =", os.environ.get("AWS_SECRET_ACCESS_KEY"))
print("ENV CHECK END")



init(autoreset=True)

# Configuration
VIDEO_PATH = "Ipoh to KL - 15minutes.mp4"  # Change to your video path
MODEL_PATH = "best.pt"  # Change to your model path
RESIZE_PERCENT = 65  # Resize frames for faster processing
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.4
LOG_DIR = "logs"
RESULTS_DIR = "results"
SHOW_WINDOW = True  # Set to False to disable video display (headless mode)
SAVE_OUTPUT_VIDEO = False  # Set to True to save annotated video
SAVE_METADATA = True  # Set to True to save detection metadata as JSON

# GPU Configuration
USE_GPU = True  # Set to False to force CPU usage
DEVICE = None  # Will be set automatically based on GPU availability

# Check GPU availability
if USE_GPU:
    if torch is None:
        print(Fore.YELLOW + "Warning: PyTorch not installed. Falling back to CPU.")
        DEVICE = "cpu"
    elif not torch.cuda.is_available():
        print(Fore.YELLOW + "Warning: CUDA not available. Falling back to CPU.")
        DEVICE = "cpu"
    else:
        DEVICE = "cuda:0"
        try:
            torch.cuda.set_device(0)
            print(Fore.GREEN + f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
            # Enable cuDNN benchmark for better performance
            torch.backends.cudnn.benchmark = True
        except Exception as e:
            print(Fore.YELLOW + f"Warning: GPU setup failed ({e}). Using CPU.")
            DEVICE = "cpu"
else:
    DEVICE = "cpu"
    print(Fore.YELLOW + "GPU usage disabled. Using CPU.")

print(Fore.CYAN + f"Device set to: {DEVICE}\n")

# Create directories first (before logging setup)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(f'{LOG_DIR}/test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
LOGGER = logging.getLogger("video_tester")

# Upload/API configuration
ENABLE_UPLOADS = os.environ.get("ENABLE_UPLOADS", "1").strip().lower() in ("1", "true", "yes")
AWS_BUCKET = os.environ.get("AWS_BUCKET", "robolyzedatamy")
AWS_BASE_URL = os.environ.get(
    "AWS_BASE_URL", "https://robolyzedatamy.s3.ap-southeast-5.amazonaws.com"
)
API_URL = os.environ.get(
    "API_URL", "https://mizube.oud.ai/api/users/DemoDashboard"
)
CLASS_TYPE_MAP = {"crack": "005"}
UPLOAD_WORKERS = int(os.environ.get("UPLOAD_WORKERS", "2"))

if ENABLE_UPLOADS:
    s3_client_kwargs: Dict[str, Optional[str]] = {
        "aws_access_key_id": os.environ.get("AWS_ACCESS_KEY_ID"),
        "aws_secret_access_key": os.environ.get("AWS_SECRET_ACCESS_KEY"),
    }
    aws_region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if aws_region:
        s3_client_kwargs["region_name"] = aws_region
    s3_client = boto3.client("s3", **{k: v for k, v in s3_client_kwargs.items() if v})
    upload_executor = concurrent.futures.ThreadPoolExecutor(max_workers=UPLOAD_WORKERS)
else:
    s3_client = None
    upload_executor = None


class UploadManager:
    """Prevent duplicate uploads for the same label/track."""

    def __init__(self):
        self._seen = set()
        self._lock = threading.Lock()

    def should_upload(self, label: Optional[str], track_id: Optional[int]) -> bool:
        if track_id is None:
            return True
        key = (label or "", int(track_id))
        with self._lock:
            # Check if timer is still activate INSIDE the lock
            if key in self._seen:
                LOGGER.warning(

                    f"DUPLOCATE_PREVENTED labels={label} track_id={track_id} (already uploaded)"
                )
                return False
            self._seen.add(key)
            LOGGER.info(f"UPLOAD_ALLOWED label={label} track_id={track_id}")
            return True


upload_manager = UploadManager()


MYT = timezone(timedelta(hours=8))
IMAGES_DIR = "images"
CRACK_LABELS = {"crack", "cracks"}


def normalize_label(name: Optional[str]) -> str:
    return (name or "").strip().lower()


def build_payload(
    class_name: str,
    image_url: str,
    gps_coords: Optional[Tuple[float, float]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    latitude = f"{gps_coords[0]:.6f}" if gps_coords else "N/A"
    longitude = f"{gps_coords[1]:.6f}" if gps_coords else "N/A"
    normalized = normalize_label(class_name)

    payload: Dict[str, Any] = {
        "typ": CLASS_TYPE_MAP.get(normalized, "000"),
        "detect": class_name,
        "img": image_url,
        "cam_loc": [latitude, longitude],
    }

    if extra:
        payload.update(extra)
    return payload


def delete_file_safely(path: Optional[str]) -> None:
    if not path:
        return
    try:
        if os.path.exists(path):
            os.remove(path)
            LOGGER.info("LOCAL_CLEANUP_SUCCESS path=%s", path)
    except Exception as exc:
        LOGGER.warning("LOCAL_CLEANUP_FAILED path=%s error=%s", path, exc)


def persist_frame_to_s3(frame, class_name: str) -> Optional[Dict[str, str]]:
    if s3_client is None:
        return None

    now = datetime.now(MYT)
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    unique_suffix = f"{now.microsecond}_{random.randint(1000, 9999)}"
    filename = f"{class_name}-{timestamp}-{unique_suffix}.jpg"
    local_path = os.path.join(IMAGES_DIR, filename)
    os.makedirs(IMAGES_DIR, exist_ok=True)

    if not cv2.imwrite(local_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85]):
        LOGGER.error("LOCAL_SAVE_FAILED path=%s", local_path)
        return None

    try:
        s3_client.upload_file(local_path, AWS_BUCKET, filename)
        LOGGER.info("S3_UPLOAD_SUCCESS bucket=%s key=%s", AWS_BUCKET, filename)
    except Exception as exc:
        LOGGER.exception("S3_UPLOAD_FAILED bucket=%s key=%s error=%s", AWS_BUCKET, filename, exc)
        delete_file_safely(local_path)
        return None

    return {
        "local_path": local_path,
        "s3_key": filename,
        "url": f"{AWS_BASE_URL}/{filename}",
    }


def post_to_dashboard(
    payload: Dict[str, Any],
    track_id: Optional[int] = None,
    local_path: Optional[str] = None,
    retries: int = 3,
) -> bool:
    for attempt in range(1, retries + 1):
        try:
            LOGGER.info(
                "API_POST attempt=%s track_id=%s url=%s payload=%s",
                attempt,
                track_id if track_id is not None else "NA",
                API_URL,
                payload,
            )
            response = requests.post(API_URL, json=payload, timeout=6)
            response.raise_for_status()
            LOGGER.info(
                "API_POST_SUCCESS attempt=%s status_code=%s track_id=%s",
                attempt,
                response.status_code,
                track_id if track_id is not None else "NA",
            )
            delete_file_safely(local_path)
            return True
        except Exception as exc:
            LOGGER.warning(
                "API_POST_FAILED attempt=%s track_id=%s error=%s",
                attempt,
                track_id if track_id is not None else "NA",
                exc,
            )
            time.sleep(2)

    error_file = f"failed_upload_{int(time.time())}.json"
    try:
        with open(error_file, "w") as handle:
            json.dump({"payload": payload, "track_id": track_id}, handle)
        LOGGER.error("API_POST_GAVE_UP track_id=%s saved=%s", track_id, error_file)
    except Exception as exc:
        LOGGER.exception("API_POST_SAVE_FAILED file=%s error=%s", error_file, exc)
    return False


def upload_immediate_detection(frame, detection: Dict[str, Any]) -> None:
    image_info = persist_frame_to_s3(frame, detection["class_name"])
    if not image_info:
        return

    payload = build_payload(detection["class_name"], image_info["url"])
    post_to_dashboard(payload, detection.get("track_id"), image_info["local_path"])


class CrackBatcher:
    """Aggregate crack detections for batched dashboard uploads."""

    def __init__(self, delay_seconds: float = 20.0):
        self.delay_seconds = delay_seconds
        self._lock = threading.Lock()
        self._timer: Optional[threading.Timer] = None
        self._pending_payload: Optional[Dict[str, Any]] = None
        self._pending_local_path: Optional[str] = None
        self._pending_track_id: Optional[int] = None
        self._count = 0

    def add_detection(self, frame, detection: Dict[str, Any]) -> None:
        image_info = persist_frame_to_s3(frame, detection["class_name"])
        if not image_info:
            return

        payload = build_payload(
            detection["class_name"],
            image_info["url"],
        )
        track_id = detection.get("track_id")

        with self._lock:
            self._count += 1
            payload["count"] = self._count

            if self._pending_local_path and self._pending_local_path != image_info["local_path"]:
                delete_file_safely(self._pending_local_path)

            self._pending_payload = payload
            self._pending_local_path = image_info["local_path"]
            self._pending_track_id = track_id

            if self._timer:
                self._timer.cancel()

            self._timer = threading.Timer(self.delay_seconds, self._flush)
            self._timer.daemon = True
            self._timer.start()

            LOGGER.info(
                "CRACK_BUFFER_UPDATED count=%s delay=%ss",
                self._count,
                self.delay_seconds,
            )

    def _flush(self) -> None:
        payload: Optional[Dict[str, Any]] = None
        local_path: Optional[str] = None
        track_id: Optional[int] = None

        with self._lock:
            payload = self._pending_payload
            local_path = self._pending_local_path
            track_id = self._pending_track_id
            self._pending_payload = None
            self._pending_local_path = None
            self._pending_track_id = None
            self._count = 0
            self._timer = None

        if not payload:
            return

        LOGGER.info("CRACK_BATCH_SENDING payload=%s", payload)
        post_to_dashboard(payload, track_id, local_path)

    def shutdown(self) -> None:
        with self._lock:
            if self._timer:
                self._timer.cancel()
                self._timer = None
            payload = self._pending_payload
            local_path = self._pending_local_path
            track_id = self._pending_track_id
            self._pending_payload = None
            self._pending_local_path = None
            self._pending_track_id = None
            self._count = 0

        if payload:
            LOGGER.info("CRACK_BATCH_FLUSH_ON_SHUTDOWN payload=%s", payload)
            post_to_dashboard(payload, track_id, local_path)


crack_batcher = CrackBatcher(delay_seconds=20.0)


def schedule_uploads(frame, detections: List[Dict[str, Any]]) -> None:
    if not ENABLE_UPLOADS or upload_executor is None:
        return

    for det in detections:
        label = normalize_label(det["class_name"])
        track_id = det["track_id"]

        if not upload_manager.should_upload(label, track_id):
            continue

        if label in CRACK_LABELS:
            upload_executor.submit(crack_batcher.add_detection, frame.copy(), det)
        else:
            upload_executor.submit(upload_immediate_detection, frame.copy(), det)


class VideoReader:
    """
    Multi-threaded video frame reader.
    Continuously reads frames in background thread and stores latest frame.
    """
    
    def __init__(self, video_path, resize_percent=65):
        """
        Initialize video reader with threading support.
        
        Args:
            video_path: Path to video file
            resize_percent: Percentage to resize frames (for performance)
        """
        self.video_path = video_path
        self.resize_percent = resize_percent
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            LOGGER.error(f"Failed to open video: {video_path}")
            raise RuntimeError(f"Cannot open video file: {video_path}")
        
        # Get video properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        LOGGER.info(f"Video loaded: {video_path}")
        LOGGER.info(f"Properties: {self.width}x{self.height} @ {self.fps}FPS, {self.total_frames} frames")
        
        # Threading components
        self.frame_queue = queue.Queue(maxsize=1)  # Only keep latest frame
        self.stopped = False
        self.frame_count = 0
        
        # Start reader thread
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()
        LOGGER.info("Video reader thread started")
    
    def _reader(self):
        """Background thread that continuously reads frames."""
        while not self.stopped:
            ret, frame = self.cap.read()
            
            if not ret:
                LOGGER.info("End of video reached")
                self.stopped = True
                break
            
            # Resize frame for performance
            if self.resize_percent and self.resize_percent != 100:
                frame = self._resize_frame(frame, self.resize_percent)
            
            self.frame_count += 1
            
            # Update queue with latest frame (drop old frame if queue is full)
            try:
                self.frame_queue.put(frame, timeout=0.1)
            except queue.Full:
                # Remove old frame and add new one
                try:
                    _ = self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
                self.frame_queue.put(frame)
    
    def _resize_frame(self, frame, percent):
        """Resize frame by percentage."""
        width = int(frame.shape[1] * percent / 100)
        height = int(frame.shape[0] * percent / 100)
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    
    def get_frame(self, timeout=1.0):
        """
        Get the latest frame from the queue.
        
        Args:
            timeout: Max time to wait for frame (seconds)
            
        Returns:
            Frame or None if timeout/stopped
        """
        try:
            frame = self.frame_queue.get(timeout=timeout)
            return frame
        except queue.Empty:
            if self.stopped:
                LOGGER.info("Video reader stopped")
                return None
            LOGGER.warning(f"Frame retrieval timeout ({timeout}s)")
            return None
    
    def is_alive(self):
        """Check if reader thread is still running."""
        return self.thread.is_alive()
    
    def is_finished(self):
        """Check if video has finished."""
        return self.stopped
    
    def stop(self):
        """Stop the reader thread and release resources."""
        self.stopped = True
        self.thread.join(timeout=1.0)
        self.cap.release()
        LOGGER.info("Video reader stopped and resources released")


class PerformanceTracker:
    """Track and display performance metrics."""
    
    def __init__(self):
        self.start_time = time.time()
        self.frame_count = 0
        self.fps_start = time.time()
        self.fps_frame_count = 0
        self.current_fps = 0
        self.detections_count = 0
        
        # Metadata storage
        self.all_detections = []  # List of all detections with metadata
        self.class_counts = {}  # Count per class
        
    def update(self, num_detections=0):
        """Update metrics with new frame."""
        self.frame_count += 1
        self.fps_frame_count += 1
        self.detections_count += num_detections
        
        # Calculate FPS every 20 frames
        if self.fps_frame_count >= 20:
            elapsed = time.time() - self.fps_start
            self.current_fps = self.fps_frame_count / elapsed
            self.fps_start = time.time()
            self.fps_frame_count = 0
    
    def add_detection(self, frame_number, class_name, confidence, bbox, track_id=None):
        """Add a detection to metadata."""
        detection_timestamp = datetime.fromtimestamp(time.time()).strftime('%H:%M:%S')
        
        detection = {
            "frame": frame_number,
            "timestamp": detection_timestamp,
            "class": class_name,
            "confidence": round(float(confidence), 4),
            "bbox": {
                "x1": int(bbox[0]),
                "y1": int(bbox[1]),
                "x2": int(bbox[2]),
                "y2": int(bbox[3])
            }
        }
        
        if track_id is not None:
            detection["track_id"] = int(track_id)
        
        self.all_detections.append(detection)
        
        # Update class counts
        if class_name not in self.class_counts:
            self.class_counts[class_name] = 0
        self.class_counts[class_name] += 1
    
    def get_stats(self):
        """Get current performance statistics."""
        elapsed = time.time() - self.start_time
        avg_fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        return {
            'current_fps': self.current_fps,
            'avg_fps': avg_fps,
            'total_frames': self.frame_count,
            'total_detections': self.detections_count,
            'elapsed_time': elapsed
        }
    
    def print_summary(self):
        """Print final performance summary."""
        stats = self.get_stats()
        print("\n" + "="*60)
        print(Fore.CYAN + "PERFORMANCE SUMMARY")
        print("="*60)
        print(f"Total Frames Processed: {stats['total_frames']}")
        print(f"Total Detections: {stats['total_detections']}")
        print(f"Average FPS: {stats['avg_fps']:.2f}")
        print(f"Total Time: {stats['elapsed_time']:.2f}s")
        
        if self.class_counts:
            print("\nDetections by Class:")
            for class_name, count in sorted(self.class_counts.items()):
                print(f"  - {class_name}: {count}")
        
        print("="*60 + "\n")
        
        LOGGER.info(f"Performance Summary - Frames: {stats['total_frames']}, "
                   f"Detections: {stats['total_detections']}, "
                   f"Avg FPS: {stats['avg_fps']:.2f}, "
                   f"Time: {stats['elapsed_time']:.2f}s")


def draw_detections(frame, result, names, fps=0):
    """
    Draw bounding boxes and labels on frame.
    
    Args:
        frame: Input frame
        result: YOLO detection result
        names: Class names dictionary
        fps: Current FPS for display
        
    Returns:
        Annotated frame
    """
    img = frame.copy()
    boxes = result.boxes
    
    # Draw FPS counter
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(img, fps_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    
    if boxes is None or boxes.xyxy is None:
        return img
    
    xyxy_list = boxes.xyxy
    cls_list = boxes.cls
    conf_list = getattr(boxes, "conf", None)
    
    # Draw each detection
    for i in range(len(cls_list)):
        x1, y1, x2, y2 = [int(float(v)) for v in xyxy_list[i]]
        cls_id = int(cls_list[i])
        label = names[cls_id]
        
        # Add confidence to label
        if conf_list is not None:
            conf = float(conf_list[i])
            label = f"{label} {conf:.2f}"
        
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label background
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        text_height_total = text_height + baseline + 4
        y_text = max(y1, text_height_total + 2)
        
        cv2.rectangle(img, 
                     (x1, y_text - text_height_total), 
                     (x1 + text_width + 6, y_text), 
                     (0, 255, 0), -1)
        
        # Draw label text
        cv2.putText(img, label, 
                   (x1 + 3, y_text - baseline - 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    
    return img


def save_metadata_json(perf, video_path, model_path, output_path):
    """Save all detection metadata to JSON file."""
    stats = perf.get_stats()
    
    metadata = {
        "test_info": {
            "video_path": video_path,
            "model_path": model_path,
            "test_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "device": DEVICE,
            "resize_percent": RESIZE_PERCENT,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "iou_threshold": IOU_THRESHOLD
        },
        "performance": {
            "total_frames": stats['total_frames'],
            "total_detections": stats['total_detections'],
            "average_fps": round(stats['avg_fps'], 2),
            "processing_time_seconds": round(stats['elapsed_time'], 2)
        },
        "class_summary": perf.class_counts,
        "detections": perf.all_detections
    }
    
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    LOGGER.info(f"Metadata saved to {output_path}")
    print(Fore.GREEN + f"✓ Metadata saved to {output_path}")
    
    # Also save a summary without individual detections for quick viewing
    summary_path = output_path.replace('.json', '_summary.json')
    summary = {
        "test_info": metadata["test_info"],
        "performance": metadata["performance"],
        "class_summary": metadata["class_summary"]
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(Fore.GREEN + f"✓ Summary saved to {summary_path}")


def main():
    """Main testing loop with multi-threading."""
    
    print(Fore.CYAN + "\n" + "="*60)
    print(Fore.CYAN + "YOLO MODEL PERFORMANCE TESTER (Multi-threaded)")
    print(Fore.CYAN + "="*60 + "\n")
    
    LOGGER.info("="*60)
    LOGGER.info("Starting YOLO model performance test")
    LOGGER.info("="*60)
    
    video_reader = None
    video_writer = None
    show_window = SHOW_WINDOW  # Local copy to modify if GUI fails
    
    try:
        # Initialize video reader (multi-threaded)
        print(Fore.YELLOW + f"Loading video: {VIDEO_PATH}")
        video_reader = VideoReader(VIDEO_PATH, resize_percent=RESIZE_PERCENT)
        
        if not video_reader.is_alive():
            raise RuntimeError("Video reader thread failed to start")
        
        print(Fore.GREEN + "✓ Video reader initialized")
        print(Fore.YELLOW + f"Loading model: {MODEL_PATH}")
        
        # Load YOLO model
        model = YOLO(MODEL_PATH)
        
        # Move model to GPU if available
        try:
            model.to(DEVICE)
            print(Fore.GREEN + f"✓ Model loaded on {DEVICE}: {', '.join(model.names.values())}")
        except Exception as e:
            print(Fore.YELLOW + f"Warning: Could not move model to {DEVICE} ({e})")
            print(Fore.GREEN + f"✓ Model loaded: {', '.join(model.names.values())}")
        
        LOGGER.info(f"Model loaded with classes: {', '.join(model.names.values())}")
        LOGGER.info(f"Using device: {DEVICE}")
        
        # Setup video writer if saving output
        if SAVE_OUTPUT_VIDEO:
            output_path = os.path.join(RESULTS_DIR, f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = video_reader.fps if video_reader.fps > 0 else 30
            frame_size = (int(video_reader.width * RESIZE_PERCENT / 100), 
                         int(video_reader.height * RESIZE_PERCENT / 100))
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
            print(Fore.GREEN + f"✓ Output video will be saved to: {output_path}")
            LOGGER.info(f"Output video path: {output_path}")
        
        # Initialize performance tracker
        perf = PerformanceTracker()
        
        if show_window:
            print(Fore.CYAN + "\nStarting inference... Press 'q' to quit\n")
        else:
            print(Fore.CYAN + "\nStarting inference (headless mode)...\n")
        LOGGER.info("Entering main detection loop")
        
        # Main detection loop
        while True:
            # Get frame from threaded reader
            frame = video_reader.get_frame(timeout=1.0)
            
            if frame is None:
                if video_reader.is_finished():
                    print(Fore.YELLOW + "\nVideo finished")
                    LOGGER.info("Video playback completed")
                    break
                else:
                    print(Fore.RED + "Failed to get frame")
                    continue
            
            # Run inference with explicit device
            results = model.track(frame, persist=True, 
                                conf=CONFIDENCE_THRESHOLD, 
                                iou=IOU_THRESHOLD,
                                device=DEVICE,
                                verbose=False)
            
            # Count detections and store metadata
            num_detections = 0
            upload_candidates: List[Dict[str, Any]] = []
            if results[0].boxes is not None and results[0].boxes.cls is not None:
                boxes = results[0].boxes
                num_detections = len(boxes.cls)
                xyxy_list = boxes.xyxy
                cls_list = boxes.cls
                conf_list = getattr(boxes, "conf", None)
                track_list = getattr(boxes, "id", None)

                for i in range(num_detections):
                    class_id = int(cls_list[i])
                    class_name = model.names[class_id]
                    confidence = float(conf_list[i]) if conf_list is not None else 0.0
                    bbox = [float(v) for v in xyxy_list[i]]
                    track_id = int(track_list[i]) if track_list is not None and i < len(track_list) else None

                    upload_candidates.append(
                        {
                            "class_name": class_name,
                            "confidence": confidence,
                            "track_id": track_id,
                            "bbox": bbox,
                        }
                    )

                    # Store detection metadata if enabled
                    if SAVE_METADATA:
                        perf.add_detection(
                            frame_number=perf.frame_count,
                            class_name=class_name,
                            confidence=confidence,
                            bbox=bbox,
                            track_id=track_id
                        )
            
            # Update performance metrics
            perf.update(num_detections)
            stats = perf.get_stats()
            
            # Draw detections
            annotated_frame = draw_detections(frame, results[0], model.names, 
                                             fps=stats['current_fps'])

            if upload_candidates and ENABLE_UPLOADS:
                schedule_uploads(
                    annotated_frame,
                    upload_candidates,
                )
            
            # Save to output video if enabled
            if SAVE_OUTPUT_VIDEO and video_writer is not None:
                video_writer.write(annotated_frame)
            
            # Display progress
            progress = (perf.frame_count / video_reader.total_frames) * 100
            print(f"\rFrame {perf.frame_count}/{video_reader.total_frames} "
                  f"({progress:.1f}%) | FPS: {stats['current_fps']:.1f} | "
                  f"Detections: {num_detections}", end='')
            
            # Show frame (with error handling for headless environments)
            if show_window:
                try:
                    cv2.imshow("YOLO Model Testing", annotated_frame)
                    
                    # Check for quit
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print(Fore.YELLOW + "\n\nUser requested exit")
                        LOGGER.info("User interrupted testing (q pressed)")
                        break
                except cv2.error as e:
                    # OpenCV GUI not available (headless environment)
                    print(Fore.YELLOW + f"\n\nWarning: Display not available ({str(e)[:50]}...)")
                    print(Fore.YELLOW + "Continuing in headless mode (no video display)\n")
                    LOGGER.warning("OpenCV GUI not available; continuing without display")
                    show_window = False  # Disable further display attempts
                    try:
                        cv2.destroyAllWindows()
                    except:
                        pass
        
        # Print final results
        perf.print_summary()
        
        # Save metadata JSON if enabled
        if SAVE_METADATA:
            metadata_file = os.path.join(RESULTS_DIR, 
                                        f"metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            save_metadata_json(perf, VIDEO_PATH, MODEL_PATH, metadata_file)
        
        print(Fore.GREEN + f"✓ Testing complete!")
        
    except KeyboardInterrupt:
        print(Fore.YELLOW + "\n\nInterrupted by user (Ctrl+C)")
        LOGGER.info("Testing interrupted by user (KeyboardInterrupt)")
    
    except Exception as e:
        print(Fore.RED + f"\nError: {e}")
        LOGGER.exception(f"Unexpected error: {e}")
    
    finally:
        # Cleanup
        if video_reader is not None:
            video_reader.stop()
        if video_writer is not None:
            video_writer.release()
            print(Fore.GREEN + f"\n✓ Output video saved")
        if show_window:
            try:
                cv2.destroyAllWindows()
            except:
                pass

        if ENABLE_UPLOADS:
            crack_batcher.shutdown()
            LOGGER.info("Crack batcher shut down")

        if upload_executor is not None:
            upload_executor.shutdown(wait=True)
        print(Fore.GREEN + "\n✓ Resources released")
        LOGGER.info("Testing session ended")


if __name__ == "__main__":
    main()
