"""Parse YOLO results into normalized detection dictionaries."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Union


class DetectionParser:
    """Convert model outputs into a list of detection dictionaries."""

    def parse(
        self,
        result: Any,
        names: Union[Dict[int, str], Sequence[str]],
    ) -> List[Dict[str, Any]]:
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
            class_name = _lookup_class_name(names, class_id)
            detection: Dict[str, Any] = {
                "class_name": class_name,
                "confidence": float(conf_list[idx]) if conf_list is not None else 0.0,
                "bbox": [float(v) for v in xyxy_list[idx]],
            }
            if (
                track_list is not None
                and idx < len(track_list)
                and track_list[idx] is not None
            ):
                detection["track_id"] = int(track_list[idx])
            detections.append(detection)
        return detections


def _lookup_class_name(
    names: Union[Dict[int, str], Sequence[str]], class_id: int
) -> str:
    if isinstance(names, dict):
        return names.get(class_id, str(class_id))
    if class_id < 0 or class_id >= len(names):
        return str(class_id)
    return str(names[class_id])
