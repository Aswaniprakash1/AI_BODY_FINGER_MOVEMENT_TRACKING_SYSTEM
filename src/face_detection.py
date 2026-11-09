"""CPU-friendly MediaPipe face detection for webcam frames."""
from __future__ import annotations

import cv2
import numpy as np


class FaceDetector:
    def __init__(self, confidence: float = 0.4):
        import mediapipe as mp

        if not hasattr(mp, "solutions") or not hasattr(mp.solutions, "face_detection"):
            raise RuntimeError("This MediaPipe installation does not include Face Detection solutions.")
        self._detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=confidence
        )

    def detect(self, frame_bgr: np.ndarray) -> list[dict]:
        if frame_bgr is None or not isinstance(frame_bgr, np.ndarray) or frame_bgr.size == 0:
            raise ValueError("Expected a non-empty webcam frame.")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return self.detect_rgb(frame_rgb)

    def detect_rgb(self, frame_rgb: np.ndarray) -> list[dict]:
        if frame_rgb is None or not isinstance(frame_rgb, np.ndarray) or frame_rgb.size == 0:
            raise ValueError("Expected a non-empty RGB webcam frame.")
        height, width = frame_rgb.shape[:2]
        result = self._detector.process(frame_rgb)
        faces = []
        for detection in result.detections or []:
            relative = detection.location_data.relative_bounding_box
            x1 = max(0, min(width, int(relative.xmin * width)))
            y1 = max(0, min(height, int(relative.ymin * height)))
            x2 = max(0, min(width, int((relative.xmin + relative.width) * width)))
            y2 = max(0, min(height, int((relative.ymin + relative.height) * height)))
            if x2 > x1 and y2 > y1:
                score = float(detection.score[0]) if detection.score else 0.0
                faces.append({"bbox": (x1, y1, x2, y2), "confidence": score})
        return faces

    def close(self) -> None:
        self._detector.close()
