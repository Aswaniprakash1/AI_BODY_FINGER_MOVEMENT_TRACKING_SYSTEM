import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.video_processing import _valid_fps, _verify_browser_mp4, process_video


class NoPose:
    def detect(self, frame):
        return []


class NoHands:
    def detect(self, frame):
        return []


class VideoProcessingTests(unittest.TestCase):
    def test_fps_fallback_for_missing_or_invalid_source_rate(self):
        class Capture:
            def __init__(self, value):
                self.value = value

            def get(self, _property):
                return self.value

        self.assertEqual(_valid_fps(Capture(0)), 25.0)
        self.assertEqual(_valid_fps(Capture(float("nan"))), 25.0)
        self.assertEqual(_valid_fps(Capture(29.97)), 29.97)

    def test_rejects_non_mp4_output_path(self):
        with self.assertRaisesRegex(ValueError, r"\.mp4"):
            process_video("missing.mp4", "output.avi", NoPose(), NoHands())

    def test_processes_and_prepares_browser_playable_h264_mp4(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.mp4"
            output = Path(folder) / "annotated.mp4"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (96, 64))
            self.assertTrue(writer.isOpened(), "OpenCV MP4V source writer unavailable")
            for index in range(5):
                frame = np.full((64, 96, 3), (index * 20, 40, 80), dtype=np.uint8)
                writer.write(frame)
            writer.release()

            diagnostics = {}
            frame_count = process_video(source, output, NoPose(), NoHands(), diagnostics=diagnostics)

            self.assertEqual(frame_count, 5)
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)
            self.assertEqual(diagnostics["opencv"]["readable_frames"], 5)
            self.assertTrue(diagnostics["opencv"]["valid"])
            self.assertEqual(diagnostics["ffmpeg"]["codec"], "h264")
            self.assertEqual(diagnostics["ffmpeg"]["pixel_format"], "yuv420p")
            self.assertTrue(diagnostics["ffmpeg"]["faststart"])
            self.assertGreater(diagnostics["file_size_mb"], 0)
            verification = _verify_browser_mp4(output, 5, 12.0, (96, 64))
            self.assertEqual(verification["codec"], "h264")
            self.assertEqual(verification["pixel_format"], "yuv420p")
            self.assertEqual(verification["frames"], 5)
            self.assertAlmostEqual(verification["fps"], 12.0)
            self.assertGreater(verification["duration"], 0)
            self.assertTrue(verification["faststart"])
            capture = cv2.VideoCapture(str(output))
            try:
                self.assertTrue(capture.isOpened())
                self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 5)
                first_fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
                codec = "".join(chr((first_fourcc >> (8 * i)) & 0xFF) for i in range(4)).lower()
                self.assertIn(codec, ("avc1", "h264"))
                ok, frame = capture.read()
                self.assertTrue(ok)
                self.assertEqual(frame.shape[:2], (64, 96))
            finally:
                capture.release()


if __name__ == "__main__":
    unittest.main()
