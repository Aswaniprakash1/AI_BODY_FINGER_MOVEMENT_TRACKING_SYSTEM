import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import cv2

from src.webcam_processing import OpenCVCamera, WebcamProcessor
from src.hand_tracking import HandTracker


class FakePose:
    def __init__(self):
        self.calls = 0
        self.image_size = None
        self.confidence = None

    def detect(self, frame, image_size=None, confidence=None):
        self.calls += 1
        self.image_size = image_size
        self.confidence = confidence
        return [{"keypoints": {
            "left_shoulder": (25, 20, .95), "right_shoulder": (40, 20, .95),
            "left_hip": (27, 42, .95), "right_hip": (38, 42, .95),
            "left_knee": (27, 65, .95), "right_knee": (38, 65, .95),
            "left_ankle": (27, 88, .95), "right_ankle": (38, 88, .95),
            "left_wrist": (18, 0, .95), "right_wrist": (47, 0, .95),
        }, "bbox": (10, 5, 55, 95)}]


class FakeHands:
    def __init__(self):
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        points = [(20., 20., 0.) for _ in range(21)]
        return [{"landmarks": points, "label": "Left"}]

    def detect_rgb(self, frame):
        self.rgb = frame.copy()
        return self.detect(frame)


class FakeFaces:
    def detect(self, frame):
        return [{"bbox": (50, 10, 90, 50), "confidence": .9}]

    def detect_rgb(self, frame):
        self.rgb = frame.copy()
        return self.detect(frame)


class FakeCapture:
    instances = []

    def __init__(self, *_args):
        self.released = False
        self.args = _args
        self.reads = 0
        FakeCapture.instances.append(self)

    def isOpened(self):
        return not self.released

    def set(self, *_args):
        return True

    def read(self):
        self.reads += 1
        return True, np.zeros((100, 100, 3), dtype=np.uint8)

    def release(self):
        self.released = True


class WebcamPipelineTests(unittest.TestCase):
    def test_hand_tracker_converts_opencv_bgr_to_mediapipe_rgb(self):
        class FakeMediaPipeHands:
            def process(self, rgb):
                self.rgb = rgb.copy()
                landmarks = [SimpleNamespace(x=i / 20, y=i / 20, z=0.0) for i in range(21)]
                return SimpleNamespace(
                    multi_hand_landmarks=[SimpleNamespace(landmark=landmarks)],
                    multi_handedness=[],
                )

        tracker = HandTracker.__new__(HandTracker)
        tracker._hands = FakeMediaPipeHands()
        bgr = np.zeros((32, 48, 3), dtype=np.uint8)
        bgr[0, 0] = (10, 20, 30)

        hands = tracker.detect(bgr)

        self.assertEqual(tuple(tracker._hands.rgb[0, 0]), (30, 20, 10))
        self.assertEqual(len(hands), 1)
        self.assertEqual(len(hands[0]["landmarks"]), 21)
        self.assertEqual(hands[0]["label"], "Unknown")

    def test_live_frame_runs_both_detectors_and_draws_results(self):
        pose, hands = FakePose(), FakeHands()
        faces = FakeFaces()
        processor = WebcamProcessor(pose, hands, threading.Lock(), faces)
        original = np.zeros((100, 100, 3), dtype=np.uint8)
        original[0, 0] = (10, 20, 30)
        output, body, detected_hands, fps, debug = processor.process_bgr(original)

        self.assertEqual(pose.calls, 1)
        self.assertEqual(hands.calls, 1)
        self.assertEqual(output.shape, original.shape)
        self.assertFalse(np.array_equal(output, original))
        self.assertTrue(body["hand_raised"])
        self.assertEqual(len(detected_hands), 1)
        self.assertGreaterEqual(fps, 0)
        self.assertEqual(pose.image_size, 512)
        self.assertEqual(pose.confidence, .25)
        self.assertEqual(debug["person_count"], 1)
        self.assertEqual(debug["keypoint_count"], 10)
        self.assertEqual(debug["hand_count"], 1)
        self.assertEqual(debug["face_count"], 1)
        self.assertEqual(tuple(hands.rgb[0, 0]), (30, 20, 10))
        self.assertEqual(tuple(faces.rgb[0, 0]), (30, 20, 10))
        self.assertTupleEqual(tuple(output[99, 99]), (0, 0, 0))  # no opaque debug panel over the video
        self.assertEqual(debug["track_ids"], [1])
        self.assertEqual(debug["person_analyses"][0]["track_id"], 1)
        self.assertEqual((debug["width"], debug["height"]), (100, 100))
        self.assertGreaterEqual(debug["inference_ms"], 0)
        self.assertGreaterEqual(debug["color_conversion_ms"], 0)
        _output, _body, _hands, _fps, next_debug = processor.process_bgr(original)
        self.assertEqual(next_debug["track_ids"], [1])

    def test_camera_reads_and_releases_handle(self):
        with patch("src.webcam_processing.cv2.VideoCapture", FakeCapture):
            FakeCapture.instances.clear()
            camera = OpenCVCamera(camera_index=1, backend=cv2.CAP_DSHOW, backend_name="DirectShow")
            camera.start()
            cap = camera.capture
            frame = camera.read()
            self.assertEqual(cap.reads, 1)  # start frame is cached, no second capture is opened/read
            camera.start()  # rerun/start is idempotent while the handle is open
            self.assertEqual(len(FakeCapture.instances), 1)
            camera.stop()
        self.assertEqual(frame.shape, (100, 100, 3))
        self.assertTrue(cap.released)
        self.assertEqual(cap.args, (1, cv2.CAP_DSHOW))

    def test_only_one_camera_session_can_own_the_device(self):
        with patch("src.webcam_processing.cv2.VideoCapture", FakeCapture):
            FakeCapture.instances.clear()
            first = OpenCVCamera(0)
            second = OpenCVCamera(1)
            first.start()
            with self.assertRaisesRegex(RuntimeError, "already open in another"):
                second.start()
            first.stop()
            second.start()
            second.stop()
        self.assertEqual(len(FakeCapture.instances), 2)

    def test_stale_closed_camera_lock_is_released_for_new_start(self):
        with patch("src.webcam_processing.cv2.VideoCapture", FakeCapture):
            FakeCapture.instances.clear()
            stale = OpenCVCamera(0)
            replacement = OpenCVCamera(0)
            stale.start()
            stale_capture = stale.capture
            stale_capture.release()  # Simulate a disconnected/released device.

            replacement.start()

            self.assertFalse(stale.is_open)
            self.assertIsNone(stale.capture)
            self.assertTrue(stale_capture.released)
            self.assertTrue(replacement.is_open)
            replacement.stop()
        self.assertEqual(len(FakeCapture.instances), 2)

    def test_camera_unavailable_has_actionable_error(self):
        class UnavailableCapture(FakeCapture):
            def isOpened(self):
                return False

        with patch("src.webcam_processing.cv2.VideoCapture", UnavailableCapture):
            UnavailableCapture.instances.clear()
            with self.assertRaisesRegex(RuntimeError, "Privacy settings"):
                OpenCVCamera().start()
        self.assertTrue(UnavailableCapture.instances[0].released)

    def test_camera_open_without_frames_reports_read_failure_and_releases(self):
        class NoFramesCapture(FakeCapture):
            def read(self):
                self.reads += 1
                return False, None

        with patch("src.webcam_processing.cv2.VideoCapture", NoFramesCapture):
            NoFramesCapture.instances.clear()
            camera = OpenCVCamera(1, cv2.CAP_MSMF, "Media Foundation")
            with self.assertRaisesRegex(RuntimeError, r"opened.*cap.read\(\) returned ok=False"):
                camera.start()
        self.assertEqual(NoFramesCapture.instances[0].reads, 3)
        self.assertTrue(NoFramesCapture.instances[0].released)


if __name__ == "__main__":
    unittest.main()
