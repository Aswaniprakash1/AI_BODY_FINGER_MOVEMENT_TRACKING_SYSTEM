import unittest
from src.movement_analysis import analyze_posture, TemporalAnalyzer
from src.gesture_analysis import count_extended_fingers, recognize_gesture

def person(points):
    names = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
    return {"keypoints": {n: (*points.get(n, (0, 0)), .99) for n in names}, "bbox": (0, 0, 100, 200)}

class AnalysisTests(unittest.TestCase):
    def test_standing_and_raise(self):
        p = person({"left_shoulder": (40, 50), "right_shoulder": (60, 50), "left_hip": (42, 100), "right_hip": (58, 100), "left_knee": (42, 150), "right_knee": (58, 150), "left_ankle": (42, 195), "right_ankle": (58, 195), "left_wrist": (30, 20)})
        result = analyze_posture(p)
        self.assertEqual(result["posture"], "Standing")
        self.assertTrue(result["hand_raised"])
    def test_missing_keypoints_safe(self):
        self.assertEqual(analyze_posture({"keypoints": {}})["posture"], "Insufficient landmarks")
        self.assertIsNone(count_extended_fingers([])["count"])
        self.assertEqual(TemporalAnalyzer().update([])["movement"], "No person detected")
    def test_upper_body_landmarks_are_enough_for_torso_posture(self):
        p = {"keypoints": {
            "left_shoulder": (40, 30, .9), "right_shoulder": (60, 30, .9),
            "left_hip": (42, 80, .9), "right_hip": (58, 80, .9),
        }, "bbox": (20, 10, 80, 100)}
        self.assertEqual(analyze_posture(p)["posture"], "Upright torso")
        self.assertEqual(TemporalAnalyzer().update([p])["movement"], "Upright torso")
    def test_zero_confidence_keypoints_are_not_a_valid_pose(self):
        p = {"keypoints": {"left_shoulder": (40, 30, 0.0), "right_shoulder": (60, 30, 0.0)},
             "bbox": (20, 10, 80, 100)}
        self.assertEqual(TemporalAnalyzer().update([p])["movement"], "Insufficient landmarks")
    def test_invalid_hand_and_fist(self):
        self.assertEqual(recognize_gesture([]), "Unknown")
        points = [(0., 0., 0.) for _ in range(21)]
        self.assertEqual(count_extended_fingers(points)["count"], 0)
        self.assertEqual(recognize_gesture(points), "Fist")

if __name__ == "__main__":
    unittest.main()
