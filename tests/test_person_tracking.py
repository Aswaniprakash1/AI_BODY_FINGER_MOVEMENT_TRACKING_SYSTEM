import unittest

from src.person_tracking import PersonTracker


class PersonTrackerTests(unittest.TestCase):
    def test_keeps_ids_when_boxes_move_and_detection_order_changes(self):
        tracker = PersonTracker()
        first = [
            {"bbox": (10, 10, 50, 90)},
            {"bbox": (150, 10, 190, 90)},
        ]
        tracker.update(first)
        left_id, right_id = first[0]["track_id"], first[1]["track_id"]

        next_frame = [
            {"bbox": (145, 12, 185, 92)},
            {"bbox": (15, 10, 55, 90)},
        ]
        tracker.update(next_frame)

        self.assertEqual(next_frame[0]["track_id"], right_id)
        self.assertEqual(next_frame[1]["track_id"], left_id)

    def test_track_expires_after_configured_missed_frames(self):
        tracker = PersonTracker(max_missed_frames=1)
        first = {"bbox": (10, 10, 50, 90)}
        tracker.update([first])
        first_id = first["track_id"]
        tracker.update([])
        tracker.update([])
        next_person = {"bbox": (10, 10, 50, 90)}
        tracker.update([next_person])
        self.assertNotEqual(next_person["track_id"], first_id)


if __name__ == "__main__":
    unittest.main()
