"""Test OpenCV webcam capture and MediaPipe face detection outside Streamlit/YOLO."""
from __future__ import annotations

import argparse
import os

import cv2

from src.face_detection import FaceDetector


BACKENDS = {
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
    "default": cv2.CAP_ANY,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, default=0, help="OpenCV camera index (default: 0)")
    parser.add_argument("--backend", choices=BACKENDS, default="dshow" if os.name == "nt" else "default")
    args = parser.parse_args()

    capture = None
    face_detector = None
    try:
        capture = cv2.VideoCapture(args.index, BACKENDS[args.backend])
        opened = capture.isOpened()
        print(f"Camera index={args.index}, backend={args.backend}, cap.isOpened()={opened}")
        if not opened:
            return 1

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        ok, frame = capture.read()
        if not ok or frame is None or frame.size == 0:
            print(f"Frame capture failed: cap.read() returned ok={ok}, empty={frame is None or frame.size == 0}")
            return 1
        print(f"First frame captured: {frame.shape[1]}x{frame.shape[0]}")
        try:
            print("OpenCV actual backend:", capture.getBackendName())
        except cv2.error:
            pass

        face_detector = FaceDetector()
        last_status = None
        while True:
            if frame is None:
                ok, frame = capture.read()
                if not ok or frame is None or frame.size == 0:
                    raise RuntimeError("cap.read() stopped returning valid frames.")

            faces = face_detector.detect(frame)  # BGR input; module converts once to RGB for MediaPipe.
            preview = frame.copy()
            for face in faces:
                x1, y1, x2, y2 = face["bbox"]
                cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(preview, f"Face {face['confidence']:.2f}", (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2, cv2.LINE_AA)
            status = f"Face detected: {len(faces)}" if faces else "No face detected"
            cv2.putText(preview, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        .7, (0, 255, 0) if faces else (0, 0, 255), 2, cv2.LINE_AA)
            if status != last_status:
                confidence_text = ", ".join(f"{face['confidence']:.2f}" for face in faces) or "none"
                print(f"{status}; confidence={confidence_text}")
                last_status = status
            cv2.imshow("Webcam + MediaPipe Face Detection Test (Q/Esc to quit)", preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                return 0
            ok, frame = capture.read()
            if not ok or frame is None or frame.size == 0:
                raise RuntimeError("cap.read() failed during face detection.")
    except Exception as exc:
        print(f"Webcam/face test failed: {exc}")
        return 1
    finally:
        if face_detector is not None:
            face_detector.close()
        if capture is not None:
            capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
