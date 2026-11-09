import unittest
from types import SimpleNamespace

import numpy as np

from src.pose_detection import PoseDetector


class TensorLike:
    def __init__(self, value):
        self.value = value

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class PoseDetectionTests(unittest.TestCase):
    def test_extracts_pose_keypoints_confidence_and_uses_webcam_image_size(self):
        class Model:
            def predict(self, frame, **kwargs):
                self.frame = frame
                self.kwargs = kwargs
                xy = np.zeros((1, 17, 2), dtype=np.float32)
                xy[0, 5] = (25, 30)
                confidence = np.full((1, 17), .8, dtype=np.float32)
                boxes = np.array([[1, 2, 50, 60]], dtype=np.float32)
                return [SimpleNamespace(
                    keypoints=SimpleNamespace(xy=TensorLike(xy), conf=TensorLike(confidence)),
                    boxes=SimpleNamespace(xyxy=TensorLike(boxes)),
                )]

        detector = PoseDetector.__new__(PoseDetector)
        detector.model = Model()
        detector.confidence = .35
        detector.image_size = 640
        frame = np.zeros((64, 96, 3), dtype=np.uint8)

        people = detector.detect(frame, image_size=512, confidence=.25)

        self.assertEqual(len(people), 1)
        shoulder = people[0]["keypoints"]["left_shoulder"]
        self.assertEqual(shoulder[:2], (25., 30.))
        self.assertAlmostEqual(shoulder[2], .8)
        self.assertEqual(detector.model.kwargs["conf"], .25)
        self.assertEqual(detector.model.kwargs["imgsz"], 512)
        self.assertEqual(detector.model.kwargs["max_det"], 10)
        self.assertIs(detector.model.frame, frame)

    def test_missing_keypoint_confidence_is_not_promoted_to_valid(self):
        class Model:
            def predict(self, frame, **_kwargs):
                return [SimpleNamespace(
                    keypoints=SimpleNamespace(xy=TensorLike(np.ones((1, 17, 2)),), conf=None),
                    boxes=SimpleNamespace(xyxy=TensorLike(np.ones((1, 4))),),
                )]

        detector = PoseDetector.__new__(PoseDetector)
        detector.model = Model()
        detector.confidence = .25
        detector.image_size = 640
        person = detector.detect(np.zeros((64, 96, 3), dtype=np.uint8))[0]
        self.assertTrue(all(point[2] == 0.0 for point in person["keypoints"].values()))


if __name__ == "__main__":
    unittest.main()
