"""Lightweight IoU and center-distance tracker for webcam person boxes."""
from __future__ import annotations

import math


def _center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def _iou(first, second):
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = width * height
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _normalized_center_distance(first, second):
    ax, ay = _center(first)
    bx, by = _center(second)
    distance = math.hypot(ax - bx, ay - by)
    diagonal_a = math.hypot(first[2] - first[0], first[3] - first[1])
    diagonal_b = math.hypot(second[2] - second[0], second[3] - second[1])
    return distance / max(1.0, (diagonal_a + diagonal_b) / 2)


class PersonTracker:
    """Assign persistent IDs by matching each frame's boxes to recent boxes."""

    def __init__(self, max_missed_frames: int = 8, min_iou: float = 0.05,
                 max_center_distance: float = 0.75):
        self.max_missed_frames = max_missed_frames
        self.min_iou = min_iou
        self.max_center_distance = max_center_distance
        self._tracks = {}
        self._next_id = 1

    @property
    def active_ids(self):
        return tuple(sorted(self._tracks))

    def update(self, people: list[dict]) -> list[dict]:
        boxes = [tuple(float(value) for value in person.get("bbox", (0, 0, 0, 0)))
                 for person in people]
        pairs = []
        for track_id, track in self._tracks.items():
            for index, box in enumerate(boxes):
                overlap = _iou(track["bbox"], box)
                distance = _normalized_center_distance(track["bbox"], box)
                if overlap >= self.min_iou or distance <= self.max_center_distance:
                    pairs.append((overlap - 0.1 * distance, track_id, index))
        pairs.sort(reverse=True)
        matched_tracks, matched_people = set(), set()
        for _score, track_id, index in pairs:
            if track_id in matched_tracks or index in matched_people:
                continue
            self._tracks[track_id] = {"bbox": boxes[index], "missed": 0}
            people[index]["track_id"] = track_id
            matched_tracks.add(track_id)
            matched_people.add(index)

        for track_id in list(self._tracks):
            if track_id not in matched_tracks:
                self._tracks[track_id]["missed"] += 1
                if self._tracks[track_id]["missed"] > self.max_missed_frames:
                    del self._tracks[track_id]

        for index, person in enumerate(people):
            if index not in matched_people:
                track_id = self._next_id
                self._next_id += 1
                self._tracks[track_id] = {"bbox": boxes[index], "missed": 0}
                person["track_id"] = track_id
        return people
