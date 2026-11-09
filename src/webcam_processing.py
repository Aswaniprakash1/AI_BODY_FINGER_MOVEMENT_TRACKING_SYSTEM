"""OpenCV webcam lifecycle and per-frame body/hand analysis."""
from __future__ import annotations

import atexit
import threading
import time
import weakref

import cv2
import numpy as np

from .movement_analysis import KEYPOINT_CONFIDENCE_THRESHOLD, TemporalAnalyzer
from .person_tracking import PersonTracker
from .visualization import draw_faces, draw_hands, draw_people

_ACTIVE_CAMERA_LOCK = threading.RLock()
_ACTIVE_CAMERA = None


def _release_process_camera():
    with _ACTIVE_CAMERA_LOCK:
        camera = _ACTIVE_CAMERA() if _ACTIVE_CAMERA is not None else None
    if camera is not None:
        camera.stop()


atexit.register(_release_process_camera)


class OpenCVCamera:
    """Own exactly one capture handle until Stop or a camera read error."""

    def __init__(self, camera_index: int = 0, backend: int | None = None,
                 backend_name: str = "OpenCV default"):
        self.camera_index = camera_index
        self.backend = backend
        self.backend_name = backend_name
        self.capture = None
        self._first_frame = None
        self._lock = threading.RLock()

    @property
    def is_open(self) -> bool:
        if self.capture is None:
            return False
        try:
            return bool(self.capture.isOpened())
        except (cv2.error, AttributeError, RuntimeError):
            return False

    def start(self) -> None:
        global _ACTIVE_CAMERA
        with self._lock:
            if self.is_open:
                return
            self.stop()
            with _ACTIVE_CAMERA_LOCK:
                active_camera = _ACTIVE_CAMERA() if _ACTIVE_CAMERA is not None else None
                if active_camera is not None and active_camera is not self:
                    if active_camera.is_open:
                        raise RuntimeError(
                            "The webcam is already open in another Live Detection session. Stop that camera first."
                        )
                    # A dead capture can leave its owner alive in Streamlit's
                    # session state. Release it and remove the false lock so a
                    # new Start attempt can recover normally.
                    active_camera.stop()
                elif _ACTIVE_CAMERA is not None and active_camera is None:
                    # The weakly referenced Streamlit session has disappeared.
                    _ACTIVE_CAMERA = None
                # Reserve a single process-wide camera before opening it so two
                # Streamlit sessions cannot race and create competing handles.
                _ACTIVE_CAMERA = weakref.ref(self)
            # Create one handle for the user-selected camera/backend. Do not
            # silently cycle through devices and leave a competing handle open.
            try:
                cap = (cv2.VideoCapture(self.camera_index, self.backend)
                       if self.backend is not None else cv2.VideoCapture(self.camera_index))
            except cv2.error as exc:
                self.stop()
                raise RuntimeError(
                    f"OpenCV could not create camera index {self.camera_index} "
                    f"with {self.backend_name}: {exc}"
                ) from exc
            if not cap.isOpened():
                cap.release()
                self.stop()
                raise RuntimeError(
                    f"Could not open camera index {self.camera_index} with {self.backend_name}. "
                    "Try another camera index or backend, allow desktop camera access in Windows "
                    "Privacy settings, and close apps that may be using the camera."
                )

            self.capture = cap
            try:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                last_read = "no read attempted"
                for attempt in range(3):
                    ok, frame = cap.read()
                    if ok and frame is not None and frame.size:
                        self._first_frame = frame
                        return
                    last_read = f"cap.read() returned ok={ok}, frame_empty={frame is None or frame.size == 0}"
                    if attempt < 2:
                        time.sleep(0.05)
                raise RuntimeError(
                    f"Camera index {self.camera_index} opened with {self.backend_name}, but no frame was "
                    f"read after 3 attempts ({last_read}). Check permissions, camera use by other apps, "
                    "and the camera connection."
                )
            except Exception:
                self.stop()
                raise

    def read(self) -> np.ndarray:
        with self._lock:
            if not self.is_open:
                raise RuntimeError("Camera is not open. Press Start Camera to begin.")
            if self._first_frame is not None:
                frame, self._first_frame = self._first_frame, None
                return frame
            try:
                ok, frame = self.capture.read()
            except cv2.error as exc:
                self.stop()
                raise RuntimeError(
                    f"OpenCV cap.read() failed for camera index {self.camera_index} "
                    f"using {self.backend_name}: {exc}"
                ) from exc
            if not ok or frame is None or frame.size == 0:
                self.stop()
                raise RuntimeError(
                    f"cap.read() failed for camera index {self.camera_index} using {self.backend_name} "
                    f"(ok={ok}, frame_empty={frame is None or frame.size == 0}). The camera may have "
                    "been disconnected, permission revoked, or taken by another application."
                )
            return frame

    def stop(self) -> None:
        global _ACTIVE_CAMERA
        with self._lock:
            cap, self.capture = self.capture, None
            self._first_frame = None
            if cap is not None:
                try:
                    cap.release()
                except (cv2.error, AttributeError, RuntimeError):
                    # The device may already have been disconnected/released.
                    pass
            with _ACTIVE_CAMERA_LOCK:
                active_camera = _ACTIVE_CAMERA() if _ACTIVE_CAMERA is not None else None
                if active_camera is self:
                    _ACTIVE_CAMERA = None

    def __del__(self):
        # Best-effort cleanup when a Streamlit session disappears unexpectedly.
        self.stop()


class WebcamProcessor:
    """Run existing pose and hand models on each OpenCV frame."""

    def __init__(self, pose_detector, hand_tracker, inference_lock, face_detector=None):
        self.pose_detector = pose_detector
        self.hand_tracker = hand_tracker
        self.face_detector = face_detector
        self.inference_lock = inference_lock
        self.person_tracker = PersonTracker()
        self.temporal_by_id = {}
        self.started_at = time.perf_counter()
        self.frame_count = 0
        self._last_display_detections = {"people": [], "hands": [], "faces": []}
        self._display_misses = {"people": 0, "hands": 0, "faces": 0}

    def _persist_display_detections(self, name: str, detections: list[dict], max_missed: int = 3):
        """Bridge brief detector dropouts so live overlays do not flash off."""
        if detections:
            self._last_display_detections[name] = detections
            self._display_misses[name] = 0
            return detections
        if self._last_display_detections[name] and self._display_misses[name] < max_missed:
            self._display_misses[name] += 1
            return self._last_display_detections[name]
        self._last_display_detections[name] = []
        self._display_misses[name] = 0
        return []

    def process_bgr(self, frame: np.ndarray) -> tuple[np.ndarray, dict, list[dict], float, dict]:
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Camera returned an empty frame.")

        conversion_started = time.perf_counter()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        conversion_ms = (time.perf_counter() - conversion_started) * 1000
        with self.inference_lock:
            # YOLO Pose gets the same OpenCV BGR frame as image processing. A
            # 512 input keeps the lightweight pose model practical on CPU.
            pose_started = time.perf_counter()
            people = self.pose_detector.detect(frame, image_size=512, confidence=0.25)
            pose_ms = (time.perf_counter() - pose_started) * 1000
            hands_started = time.perf_counter()
            hands = self.hand_tracker.detect_rgb(frame_rgb)
            hands_ms = (time.perf_counter() - hands_started) * 1000
            faces_started = time.perf_counter()
            faces = self.face_detector.detect_rgb(frame_rgb) if self.face_detector is not None else []
            face_ms = (time.perf_counter() - faces_started) * 1000
        inference_ms = pose_ms + hands_ms + face_ms

        people = self.person_tracker.update(people)
        person_analyses = []
        for person in people:
            track_id = person["track_id"]
            analyzer = self.temporal_by_id.setdefault(track_id, TemporalAnalyzer())
            result = analyzer.update([person])
            person_analyses.append({"track_id": track_id, **result})
        active_ids = set(self.person_tracker.active_ids)
        self.temporal_by_id = {track_id: analyzer for track_id, analyzer in self.temporal_by_id.items()
                               if track_id in active_ids}

        keypoint_scores = [point[2] for person in people
                           for point in person.get("keypoints", {}).values()
                           if point and len(point) >= 3 and np.isfinite(point[2])]
        visible_scores = [score for score in keypoint_scores
                          if score >= KEYPOINT_CONFIDENCE_THRESHOLD]
        debug = {
            "person_count": len(people),
            "tracked_person_count": len(person_analyses),
            "active_track_count": len(self.person_tracker.active_ids),
            "track_ids": [person["track_id"] for person in people],
            "person_analyses": person_analyses,
            "face_count": len(faces),
            "face_confidences": [face["confidence"] for face in faces],
            "keypoint_count": len(visible_scores),
            "keypoint_total": len(keypoint_scores),
            "keypoint_confidence": float(np.mean(visible_scores)) if visible_scores else 0.0,
            "hand_count": len(hands),
            "width": int(frame.shape[1]),
            "height": int(frame.shape[0]),
            "inference_ms": inference_ms,
            "pose_ms": pose_ms,
            "hands_ms": hands_ms,
            "face_ms": face_ms,
            "color_conversion_ms": conversion_ms,
        }

        self.frame_count += 1
        elapsed = max(time.perf_counter() - self.started_at, 1e-6)
        fps = self.frame_count / elapsed
        body = person_analyses[0] if person_analyses else {
            "posture": "Unknown", "movement": "No person detected", "hand_raised": False
        }
        # Draw every result on this exact camera frame; app.py then converts
        # the completed BGR image to RGB once for Streamlit's image renderer.
        display_people = self._persist_display_detections("people", people)
        display_hands = self._persist_display_detections("hands", hands)
        display_faces = self._persist_display_detections("faces", faces)
        annotated = draw_hands(
            draw_people(draw_faces(frame.copy(), display_faces), display_people), display_hands
        )

        return annotated, body, hands, fps, debug
