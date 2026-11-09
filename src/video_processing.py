"""Encode annotated video safely, then prepare a browser-compatible H.264 MP4."""
from __future__ import annotations

from contextlib import nullcontext
import math
import os
from pathlib import Path
import re
import subprocess

import cv2
import numpy as np

from .gesture_analysis import count_extended_fingers, recognize_gesture
from .movement_analysis import TemporalAnalyzer
from .visualization import draw_hands, draw_people


class _DetectionPersistence:
    """Keep overlays through very short detector misses in consecutive video frames."""

    def __init__(self, max_missed_frames: int = 4):
        self.max_missed_frames = max_missed_frames
        self._last = {"people": [], "hands": []}
        self._missed = {"people": 0, "hands": 0}

    def update(self, key: str, detections: list[dict]) -> list[dict]:
        if detections:
            self._last[key] = detections
            self._missed[key] = 0
            return detections
        if self._last[key] and self._missed[key] < self.max_missed_frames:
            self._missed[key] += 1
            return self._last[key]
        self._last[key] = []
        self._missed[key] = 0
        return []


def _valid_fps(capture) -> float:
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    return fps if math.isfinite(fps) and 0 < fps <= 240 else 25.0


def _as_bgr(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim != 3:
        raise ValueError("Video decoder returned an invalid frame shape.")
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    if frame.shape[2] != 3:
        raise ValueError("Video frames must have 3 BGR color channels.")
    return frame


def _annotate(frame, pose, hands, analyzer, inference_lock, persistence=None):
    with inference_lock if inference_lock is not None else nullcontext():
        people = pose.detect(frame)
        detected_hands = hands.detect(frame)
    analysis = analyzer.update(people)
    display_people = persistence.update("people", people) if persistence else people
    display_hands = persistence.update("hands", detected_hands) if persistence else detected_hands
    draw_people(frame, display_people)
    draw_hands(frame, display_hands)
    movement = analysis["movement"] if people else "Tracking briefly"
    cv2.putText(frame, movement, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 255), 2)
    if display_hands:
        hand = display_hands[0]
        finger_count = count_extended_fingers(hand["landmarks"])["count"]
        label = f'{hand["label"]}: {recognize_gesture(hand["landmarks"])} ({finger_count} fingers)'
        cv2.putText(frame, label[:80], (15, 65), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 0), 2)
    return frame


def _h264_encoders(ffmpeg: str) -> list[str]:
    result = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Could not inspect FFmpeg encoders: {result.stderr[-1200:]}")
    available = result.stdout
    return [encoder for encoder in ("libx264", "libopenh264", "h264_mf", "h264_nvenc", "h264_amf", "h264_qsv")
            if re.search(rf"\b{re.escape(encoder)}\b", available)]


def _transcode_h264(source: Path, destination: Path) -> None:
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError(f"H.264 playback preparation is unavailable: {exc}. Install imageio-ffmpeg and retry.") from exc

    encoders = _h264_encoders(ffmpeg)
    if not encoders:
        raise RuntimeError("FFmpeg is installed, but it has no H.264 encoder. Install an FFmpeg build with libx264 or H.264 support.")
    temporary = destination.with_name(destination.stem + "_h264_tmp.mp4")
    errors = []
    try:
        for encoder in encoders:
            if temporary.exists():
                temporary.unlink()
            command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
                       "-an", "-c:v", encoder]
            command += ["-preset", "veryfast", "-crf", "23"] if encoder == "libx264" else ["-b:v", "4M"]
            # H.264 yuv420p requires even dimensions for broad browser compatibility.
            command += ["-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", "-f", "mp4", str(temporary)]
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            if result.returncode == 0 and temporary.exists() and temporary.stat().st_size > 0:
                os.replace(temporary, destination)
                return encoder
            errors.append(f"{encoder}: {result.stderr[-700:] or f'exit code {result.returncode}'}")
        raise RuntimeError("All available H.264 encoders failed: " + "; ".join(errors))
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_video(path: Path, expected_frames: int, expected_size: tuple[int, int]) -> dict:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError("The encoded MP4 cannot be reopened by OpenCV.")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if (width, height) != expected_size:
            raise RuntimeError(f"OpenCV reports dimensions {width}x{height}; expected {expected_size[0]}x{expected_size[1]}.")
        if not math.isfinite(fps) or fps <= 0:
            raise RuntimeError(f"OpenCV reports an invalid output FPS: {fps}.")

        readable_frames = 0
        first = None
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None or frame.size == 0:
                raise RuntimeError(f"OpenCV returned an empty video frame at index {readable_frames}.")
            if first is None:
                first = frame
            readable_frames += 1
        if readable_frames == 0 or first is None:
            raise RuntimeError("The encoded MP4 contains no decodable frames.")
        if readable_frames != expected_frames:
            raise RuntimeError(f"OpenCV decoded {readable_frames} frames; expected {expected_frames}.")
        reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if reported_frames and reported_frames != expected_frames:
            raise RuntimeError(f"Encoded video reports {reported_frames} frames; expected {expected_frames}.")
        size = (int(first.shape[1]), int(first.shape[0]))
        if size != expected_size:
            raise RuntimeError(f"Encoded dimensions are {size}; expected {expected_size}.")
        return {"valid": True, "readable_frames": readable_frames, "width": width, "height": height,
                "fps": fps, "duration_seconds": readable_frames / fps}
    finally:
        capture.release()


def _mp4_atoms(path: Path) -> list[tuple[str, int]]:
    """Read top-level ISO BMFF boxes to confirm moov is before media data."""
    size = path.stat().st_size
    atoms = []
    offset = 0
    with path.open("rb") as video_file:
        while offset + 8 <= size:
            video_file.seek(offset)
            header = video_file.read(8)
            box_size = int.from_bytes(header[:4], "big")
            box_type = header[4:8].decode("ascii", errors="replace")
            header_size = 8
            if box_size == 1:
                extended_size = video_file.read(8)
                if len(extended_size) != 8:
                    break
                box_size = int.from_bytes(extended_size, "big")
                header_size = 16
            elif box_size == 0:
                box_size = size - offset
            if box_size < header_size or offset + box_size > size:
                raise RuntimeError(f"Invalid MP4 {box_type!r} atom size.")
            atoms.append((box_type, offset))
            offset += box_size
    return atoms


def _verify_browser_mp4(path: Path, expected_frames: int, expected_fps: float,
                        expected_size: tuple[int, int]) -> dict:
    """Probe and decode the final file with FFmpeg; verify browser playback traits."""
    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [ffmpeg, "-hide_banner", "-loglevel", "info", "-i", str(path),
               "-map", "0:v:0", "-f", "null", "-", "-progress", "pipe:1", "-nostats"]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        detail = result.stderr[-1200:] or f"FFmpeg exited with code {result.returncode}."
        raise RuntimeError(f"FFmpeg could not decode the final annotated video: {detail}")

    video_line = next((line for line in result.stderr.splitlines() if "Video:" in line), "")
    if not re.search(r"\bVideo:\s*h264\b", video_line, re.IGNORECASE):
        raise RuntimeError(f"Final video codec is not H.264. FFmpeg reported: {video_line or 'no video stream'}")
    if not re.search(r"\byuv420p\b", video_line, re.IGNORECASE):
        raise RuntimeError(f"Final video is not yuv420p. FFmpeg reported: {video_line}")

    dimensions = re.search(r"\b(\d{2,5})x(\d{2,5})\b", video_line)
    if not dimensions:
        raise RuntimeError(f"FFmpeg could not read final video dimensions: {video_line}")
    width, height = int(dimensions.group(1)), int(dimensions.group(2))
    if (width, height) != expected_size:
        raise RuntimeError(f"Final dimensions are {width}x{height}; expected {expected_size[0]}x{expected_size[1]}.")

    fps_match = re.search(r",\s*([\d.]+)\s+fps\b", video_line)
    duration_match = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", result.stderr)
    frame_matches = re.findall(r"(?m)^frame=(\d+)\s*$", result.stdout)
    if not fps_match or not duration_match or not frame_matches:
        raise RuntimeError("FFmpeg could not verify the output FPS, duration, or decoded frame count.")
    fps = float(fps_match.group(1))
    hours, minutes, seconds = duration_match.groups()
    duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    decoded_frames = int(frame_matches[-1])
    if fps <= 0 or abs(fps - expected_fps) > max(0.05, expected_fps * 0.01):
        raise RuntimeError(f"Final FPS is {fps:g}; expected about {expected_fps:g}.")
    if duration <= 0 or abs(duration - expected_frames / expected_fps) > max(0.05, 2 / expected_fps):
        raise RuntimeError(f"Final duration {duration:.3f}s is invalid for {expected_frames} frames at {expected_fps:g} FPS.")
    if decoded_frames != expected_frames:
        raise RuntimeError(f"FFmpeg decoded {decoded_frames} frames; expected {expected_frames}.")

    atoms = _mp4_atoms(path)
    atom_types = [kind for kind, _offset in atoms]
    moov = next((offset for kind, offset in atoms if kind == "moov"), None)
    mdat = next((offset for kind, offset in atoms if kind == "mdat"), None)
    if "ftyp" not in atom_types or moov is None or mdat is None or moov > mdat:
        raise RuntimeError("MP4 faststart verification failed: moov must be placed before mdat.")

    return {"codec": "h264", "pixel_format": "yuv420p", "width": width, "height": height,
            "fps": fps, "duration": duration, "frames": decoded_frames, "faststart": True,
            "ffmpeg_path": ffmpeg, "ffmpeg_probe_succeeded": True}


def process_video(input_path, output_path, pose, hands, progress=None, inference_lock=None, diagnostics=None):
    """Annotate every input frame, encode H.264 MP4, and validate browser output.

    Returns the number of annotated frames. Raises a clear error for invalid input,
    encoding failure, or an output that cannot be reopened for playback.
    """
    source = Path(input_path)
    output = Path(output_path)
    if output.suffix.lower() != ".mp4":
        raise ValueError("Annotated video output must use a .mp4 extension.")
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_output = output.with_name(output.stem + "_opencv.mp4")
    for stale in (raw_output, output):
        if stale.exists():
            stale.unlink()

    capture = cv2.VideoCapture(str(source))
    writer = None
    written = 0
    analyzer = TemporalAnalyzer()
    persistence = _DetectionPersistence()
    try:
        if not capture.isOpened():
            raise ValueError("Could not open the uploaded video. Check that its format is supported and the file is not damaged.")
        fps = _valid_fps(capture)
        ok, first_frame = capture.read()
        if not ok or first_frame is None or first_frame.size == 0:
            raise ValueError("The uploaded video contains no readable frames.")
        first_frame = _as_bgr(first_frame)
        height, width = first_frame.shape[:2]
        if width <= 0 or height <= 0:
            raise ValueError("Video has invalid frame dimensions.")
        browser_size = (width // 2 * 2, height // 2 * 2)

        writer = cv2.VideoWriter(str(raw_output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height), True)
        if not writer.isOpened():
            raise RuntimeError("OpenCV could not create its temporary MP4V video. Check the OpenCV video encoder installation.")

        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame = first_frame
        while True:
            frame = _as_bgr(frame)
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
            annotated = _annotate(frame, pose, hands, analyzer, inference_lock, persistence)
            if annotated.shape[:2] != (height, width) or annotated.dtype != np.uint8:
                raise RuntimeError("Annotated frame dimensions or color depth changed during encoding.")
            writer.write(annotated)
            written += 1
            if progress and total > 0:
                progress(min(written / total, 1.0))

            ok, frame = capture.read()
            if not ok:
                break
        # Flush MP4 headers and close the input before FFmpeg opens the file.
        writer.release()
        writer = None
        capture.release()
        if written == 0 or not raw_output.exists() or raw_output.stat().st_size == 0:
            raise RuntimeError("Video encoding produced an empty file; no annotated frames were written.")
        # OpenCV MP4V output is not supported by all browsers, so always convert to
        # H.264 and fail visibly if a usable encoder is not present.
        h264_encoder = _transcode_h264(raw_output, output)
        opencv_report = _validate_video(output, written, browser_size)
        ffmpeg_report = _verify_browser_mp4(output, written, fps, browser_size)
        if diagnostics is not None:
            diagnostics.update({
                "path": str(output.resolve()),
                "file_size_bytes": output.stat().st_size,
                "file_size_mb": output.stat().st_size / (1024 * 1024),
                "opencv": opencv_report,
                "ffmpeg": ffmpeg_report,
                "encoder": h264_encoder,
            })
        return written
    except Exception as exc:
        output.unlink(missing_ok=True)
        if isinstance(exc, (ValueError, RuntimeError)) and str(exc).startswith(("Could not open", "The uploaded", "Video has", "OpenCV could not", "Video encoding")):
            raise
        raise RuntimeError(f"Could not prepare a playable H.264 MP4: {exc}") from exc
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if raw_output.exists():
            raw_output.unlink()
