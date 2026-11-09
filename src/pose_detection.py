"""YOLO pose inference with simple, explicit result conversion."""
from __future__ import annotations
import numpy as np

KEYPOINT_NAMES = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
                  "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                  "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
                  "right_knee", "left_ankle", "right_ankle"]

class PoseDetector:
    def __init__(self, model_name: str = "yolo11n-pose.pt", confidence: float = 0.35,
                 image_size: int = 640):
        from ultralytics import YOLO
        self.model = YOLO(model_name)  # Ultralytics downloads official weights when absent.
        self.confidence = confidence
        self.image_size = image_size

    def detect(self, frame: np.ndarray, image_size: int | None = None,
               confidence: float | None = None) -> list[dict]:
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Expected a non-empty image frame.")
        # Ultralytics accepts OpenCV frames in BGR order and converts internally.
        result = self.model.predict(frame, conf=self.confidence if confidence is None else confidence,
                                    imgsz=image_size or self.image_size,
                                    max_det=10, verbose=False)[0]
        people = []
        if result.keypoints is None or result.keypoints.xy is None:
            return people
        xy = result.keypoints.xy.cpu().numpy()
        # Missing confidence values are unknown, not proof that every joint is
        # visible. Downstream posture logic will correctly reject these points.
        conf = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else np.zeros(xy.shape[:2])
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.zeros((len(xy), 4))
        box_conf_tensor = getattr(result.boxes, "conf", None) if result.boxes is not None else None
        box_conf = box_conf_tensor.cpu().numpy() if box_conf_tensor is not None else np.zeros(len(xy))
        for points, scores, box, person_confidence in zip(xy, conf, boxes, box_conf):
            people.append({"keypoints": {name: (float(p[0]), float(p[1]), float(s))
                           for name, p, s in zip(KEYPOINT_NAMES, points, scores)},
                           "bbox": tuple(float(v) for v in box),
                           "confidence": float(person_confidence)})
        return people
