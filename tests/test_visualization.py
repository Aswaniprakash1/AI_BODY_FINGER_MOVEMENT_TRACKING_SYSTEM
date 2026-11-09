import unittest

import numpy as np

from src.pose_detection import KEYPOINT_NAMES
from src.visualization import BODY_EDGES, draw_faces, draw_hands, draw_people


class VisualizationTests(unittest.TestCase):
    def test_draws_full_coco_skeleton_person_box_and_highlighted_nose(self):
        frame = np.zeros((120, 120, 3), dtype=np.uint8)
        keypoints = {name: (10 + index * 4, 10 + index * 3, .9)
                     for index, name in enumerate(KEYPOINT_NAMES)}
        keypoints["nose"] = (20, 20, .9)
        keypoints["left_eye"] = (4, 4, .9)
        keypoints["right_eye"] = (8, 4, .9)
        keypoints["left_ear"] = (2, 8, .9)
        keypoints["right_ear"] = (12, 8, .9)
        person = {"keypoints": keypoints, "bbox": (40, 40, 100, 110), "confidence": .93}

        result = draw_people(frame, [person])

        self.assertEqual(len(BODY_EDGES), 16)
        self.assertIn((0, 1), BODY_EDGES)
        self.assertIn((0, 2), BODY_EDGES)
        self.assertTupleEqual(tuple(result[20, 20]), (0, 0, 255))
        self.assertTrue(np.any(np.all(result == (255, 100, 0), axis=2)))

    def test_draws_face_box_and_all_21_hand_landmarks(self):
        frame = np.zeros((120, 120, 3), dtype=np.uint8)
        draw_faces(frame, [{"bbox": (30, 20, 70, 65), "confidence": .88}])
        points = [(80 + index % 4, 20 + index // 4, 0) for index in range(21)]
        result = draw_hands(frame, [{"landmarks": points}])

        self.assertTrue(np.any(np.all(result == (255, 255, 0), axis=2)))
        self.assertTrue(np.any(np.all(result == (255, 0, 180), axis=2)))


if __name__ == "__main__":
    unittest.main()
