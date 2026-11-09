import unittest
from types import SimpleNamespace

import numpy as np

from src.face_detection import FaceDetector


class FaceDetectionTests(unittest.TestCase):
    def test_converts_bgr_to_rgb_and_returns_pixel_box_with_confidence(self):
        class FakeMediaPipeDetector:
            def process(self, rgb):
                self.rgb = rgb.copy()
                relative = SimpleNamespace(xmin=.25, ymin=.2, width=.5, height=.5)
                detection = SimpleNamespace(
                    location_data=SimpleNamespace(relative_bounding_box=relative), score=[.91]
                )
                return SimpleNamespace(detections=[detection])

        detector = FaceDetector.__new__(FaceDetector)
        detector._detector = FakeMediaPipeDetector()
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        frame[0, 0] = (3, 20, 200)

        faces = detector.detect(frame)

        self.assertEqual(tuple(detector._detector.rgb[0, 0]), (200, 20, 3))
        self.assertEqual(faces, [{"bbox": (50, 20, 150, 70), "confidence": .91}])


if __name__ == "__main__":
    unittest.main()
