"""Draw detections on frames."""

from __future__ import annotations

from typing import Dict, Iterable


import cv2


class FrameVisualizer:
    """Render bounding boxes and labels onto frames."""

    def draw(self, frame, detections: Iterable[Dict], inplace: bool = True):
        canvas = frame if inplace else frame.copy()
        for detection in detections:
            self.draw_detection(canvas, detection)
        return canvas

    def draw_detection(self, frame, detection: Dict) -> None:
        bbox = detection.get("bbox")
        if not bbox or len(bbox) != 4:
            return
        x1, y1, x2, y2 = [int(b) for b in bbox]
        x1 = max(0, min(frame.shape[1] - 1, x1))
        y1 = max(0, min(frame.shape[0] - 1, y1))
        x2 = max(0, min(frame.shape[1] - 1, x2))
        y2 = max(0, min(frame.shape[0] - 1, y2))
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        class_name = detection.get("class_name", "object")
        confidence = detection.get("confidence")
        track_id = detection.get("track_id")
        label_text = class_name
        if confidence is not None:
            label_text += f" {confidence:.2f}"
        if track_id is not None:
            label_text += f" #{track_id}"
        (text_w, text_h), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
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
