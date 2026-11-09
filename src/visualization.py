"""Drawing helpers for COCO body pose, face boxes, and MediaPipe Hands."""
import cv2

from .movement_analysis import KEYPOINT_CONFIDENCE_THRESHOLD
from .pose_detection import KEYPOINT_NAMES

# COCO 17-keypoint pose topology used by the pretrained YOLO Pose model.
BODY_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),                  # nose, eyes, ears
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),        # shoulders, arms
    (5, 11), (6, 12), (11, 12),                      # torso
    (11, 13), (13, 15), (12, 14), (14, 16),         # legs
]
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


def draw_faces(frame, faces):
    """Draw actual face detector boxes and confidence (distinct from YOLO nose)."""
    for index, face in enumerate(faces, start=1):
        x1, y1, x2, y2 = map(int, face["bbox"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(frame, f"Face {index}: {face['confidence']:.2f}",
                    (x1, max(20, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 0), 2, cv2.LINE_AA)
    return frame


def draw_people(frame, people):
    height, width = frame.shape[:2]
    for index, person in enumerate(people, start=1):
        x1, y1, x2, y2 = person.get("bbox", (0, 0, 0, 0))
        x1, x2 = max(0, min(width - 1, int(x1))), max(0, min(width - 1, int(x2)))
        y1, y2 = max(0, min(height - 1, int(y1))), max(0, min(height - 1, int(y2)))
        if x2 > x1 and y2 > y1:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 2)
            score = person.get("confidence")
            track_id = person.get("track_id", index)
            label = f"ID {track_id}" + (f": {score:.2f}" if score is not None else "")
            cv2.putText(frame, label, (x1, max(20, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 100, 0), 2, cv2.LINE_AA)

        keypoints = person.get("keypoints", {})
        for first, second in BODY_EDGES:
            p = keypoints.get(KEYPOINT_NAMES[first])
            q = keypoints.get(KEYPOINT_NAMES[second])
            if (p and q and p[2] >= KEYPOINT_CONFIDENCE_THRESHOLD
                    and q[2] >= KEYPOINT_CONFIDENCE_THRESHOLD):
                cv2.line(frame, tuple(map(int, p[:2])), tuple(map(int, q[:2])), (0, 220, 0), 2, cv2.LINE_AA)

        for name, point in keypoints.items():
            if not point or len(point) < 3 or point[2] < KEYPOINT_CONFIDENCE_THRESHOLD:
                continue
            x, y = map(int, point[:2])
            if name == "nose":
                # Large, high-contrast ring makes the YOLO nose keypoint obvious.
                cv2.circle(frame, (x, y), 10, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(frame, (x, y), 4, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.putText(frame, "Nose", (x + 9, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (0, 255, 255), 2, cv2.LINE_AA)
            elif name in {"left_eye", "right_eye", "left_ear", "right_ear"}:
                cv2.circle(frame, (x, y), 5, (255, 0, 255), -1, cv2.LINE_AA)
            else:
                cv2.circle(frame, (x, y), 4, (0, 100, 255), -1, cv2.LINE_AA)
    return frame


def draw_hands(frame, hands):
    for hand in hands:
        points = hand.get("landmarks", [])
        if len(points) != 21:
            continue
        for first, second in HAND_EDGES:
            cv2.line(frame, tuple(map(int, points[first][:2])),
                     tuple(map(int, points[second][:2])), (255, 180, 0), 2, cv2.LINE_AA)
        for point in points:
            cv2.circle(frame, tuple(map(int, point[:2])), 3, (255, 0, 180), -1, cv2.LINE_AA)
        wrist_x, wrist_y = map(int, points[0][:2])
        label = hand.get("label", "Hand")
        confidence = hand.get("confidence")
        if confidence is not None:
            label += f" {confidence:.2f}"
        cv2.putText(frame, label, (wrist_x + 5, wrist_y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 180, 0), 2, cv2.LINE_AA)
    return frame
