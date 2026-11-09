"""MediaPipe Hands wrapper; outputs pixel coordinates for all 21 landmarks."""
from __future__ import annotations
import cv2
import numpy as np

class HandTracker:
    def __init__(self, max_hands: int = 2, confidence: float = 0.4):
        import mediapipe as mp
        if not hasattr(mp, "solutions"):
            raise RuntimeError(
                f"MediaPipe {mp.__version__} no longer includes the legacy solutions API. "
                "Install this project's pinned version with: "
                "python -m pip install -r requirements.txt"
            )
        self._mp = mp
        self._hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=max_hands,
            min_detection_confidence=confidence, min_tracking_confidence=max(0.35, confidence))

    def detect(self, frame: np.ndarray) -> list[dict]:
        if frame is None or frame.size == 0:
            raise ValueError("Expected a non-empty image frame.")
        # OpenCV supplies BGR for both image and webcam modes; MediaPipe Hands
        # requires RGB. Keep the conversion here so every caller uses one path.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.detect_rgb(rgb_frame)

    def detect_rgb(self, rgb_frame: np.ndarray) -> list[dict]:
        if rgb_frame is None or rgb_frame.size == 0:
            raise ValueError("Expected a non-empty RGB image frame.")
        h, w = rgb_frame.shape[:2]
        result = self._hands.process(rgb_frame)
        hands = []
        if result.multi_hand_landmarks:
            handedness = result.multi_handedness or []
            for index, landmarks in enumerate(result.multi_hand_landmarks):
                points = [(p.x*w, p.y*h, p.z*w) for p in landmarks.landmark]
                handed = handedness[index] if index < len(handedness) else None
                classifications = getattr(handed, "classification", []) if handed else []
                classification = classifications[0] if classifications else None
                hands.append({"landmarks": points,
                              "label": classification.label if classification else "Unknown",
                              "confidence": float(classification.score) if classification else None})
        return hands

    def close(self):
        self._hands.close()
