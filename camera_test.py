"""Probe camera indexes/backends without importing Streamlit or vision models."""
from __future__ import annotations

import platform
import time

import cv2


def backends_to_test():
    if platform.system() == "Windows":
        return [
            ("DirectShow", cv2.CAP_DSHOW),
            ("Media Foundation", cv2.CAP_MSMF),
            ("OpenCV default", cv2.CAP_ANY),
        ]
    return [("OpenCV default", cv2.CAP_ANY)]


def test_camera(index: int, backend_name: str, backend: int) -> bool:
    # Exactly one VideoCapture exists at a time; release it before probing the
    # next index/backend so failed probes cannot keep the device busy.
    capture = cv2.VideoCapture(index, backend)
    try:
        if not capture.isOpened():
            print(f"FAIL index={index} backend={backend_name}: cap.isOpened() is false")
            return False

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        last = None
        for attempt in range(3):
            ok, frame = capture.read()
            if ok and frame is not None and frame.size:
                actual_backend = capture.getBackendName()
                print(
                    f"SUCCESS index={index} backend={backend_name} actual_backend={actual_backend} "
                    f"resolution={frame.shape[1]}x{frame.shape[0]}"
                )
                return True
            last = f"cap.read() returned ok={ok}, empty={frame is None or frame.size == 0}"
            if attempt < 2:
                time.sleep(0.05)
        print(f"FAIL index={index} backend={backend_name}: {last}")
        return False
    except cv2.error as exc:
        print(f"FAIL index={index} backend={backend_name}: OpenCV error: {exc}")
        return False
    finally:
        capture.release()


def main() -> int:
    successes = []
    for index in (0, 1):
        for backend_name, backend in backends_to_test():
            if test_camera(index, backend_name, backend):
                successes.append((index, backend_name))

    if successes:
        print("\nWorking camera configurations:")
        for index, backend_name in successes:
            print(f"  index={index}, backend={backend_name}")
        return 0
    print(
        "\nNo tested camera configuration returned a frame. Check Windows camera privacy "
        "permissions, close other camera apps, and verify the webcam is connected."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
