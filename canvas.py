"""Drawing canvas: strokes, object grouping, hit-testing and rendering.

The canvas keeps two layers:

* **objects** - committed drawings. Each object owns one or more strokes and an
  ``offset`` that shifts the whole object when it is moved by a hand.
* **live strokes** - the stroke currently being drawn by each hand, rendered
  but not yet committed.

Nothing here depends on MediaPipe or a camera, so it is fully unit-testable.
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Optional

import numpy as np

import config

_OBJECT_IDS = itertools.count(1)


def _bbox(points: np.ndarray):
    """Return ``(min_x, min_y, max_x, max_y)`` for an (N, 2) array."""
    return (
        float(points[:, 0].min()),
        float(points[:, 1].min()),
        float(points[:, 0].max()),
        float(points[:, 1].max()),
    )


def _bbox_gap(a, b) -> float:
    """Shortest gap between two axis-aligned boxes (0 if they overlap)."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    dx = max(bx0 - ax1, ax0 - bx1, 0.0)
    dy = max(by0 - ay1, ay0 - by1, 0.0)
    return float(np.hypot(dx, dy))


class Stroke:
    """A single continuous drawn line, stored in object-local coordinates."""

    def __init__(self, points: List[np.ndarray], color, thickness: int) -> None:
        self.points: List[np.ndarray] = [np.asarray(p, dtype=float) for p in points]
        self.color = tuple(color)
        self.thickness = int(thickness)

    def as_array(self) -> np.ndarray:
        return np.asarray(self.points, dtype=float)

    def translate(self, dx: float, dy: float) -> None:
        delta = np.array([dx, dy], dtype=float)
        self.points = [p + delta for p in self.points]


class DrawObject:
    """One or more strokes that move together as a single drawn object."""

    def __init__(self, stroke: Stroke) -> None:
        self.id = next(_OBJECT_IDS)
        self.strokes: List[Stroke] = [stroke]
        self.offset = np.zeros(2, dtype=float)

    # -- geometry -----------------------------------------------------------
    def _world_points(self) -> np.ndarray:
        chunks = [s.as_array() + self.offset for s in self.strokes]
        return np.vstack(chunks)

    def bbox(self):
        return _bbox(self._world_points())

    def centroid(self) -> np.ndarray:
        return self._world_points().mean(axis=0)

    def distance_to(self, point: np.ndarray) -> float:
        """Gap from a point to the object's bounding box (0 if inside)."""
        x, y = float(point[0]), float(point[1])
        x0, y0, x1, y1 = self.bbox()
        dx = max(x0 - x, x - x1, 0.0)
        dy = max(y0 - y, y - y1, 0.0)
        return float(np.hypot(dx, dy))

    # -- mutation -----------------------------------------------------------
    def add_stroke(self, stroke: Stroke) -> None:
        # Store the new stroke relative to this object's current offset so the
        # whole object can still be translated by a single offset.
        stroke.translate(-self.offset[0], -self.offset[1])
        self.strokes.append(stroke)

    def translate(self, dx: float, dy: float) -> None:
        self.offset += np.array([dx, dy], dtype=float)

    def render(self, frame: np.ndarray, highlight: bool = False) -> None:
        import cv2

        for stroke in self.strokes:
            pts = np.asarray(stroke.points, dtype=float) + self.offset
            if len(pts) >= 2:
                cv2.polylines(
                    frame,
                    [pts.astype(np.int32)],
                    False,
                    stroke.color,
                    stroke.thickness,
                    cv2.LINE_AA,
                )
            elif len(pts) == 1:
                cv2.circle(frame, tuple(pts[0].astype(int)), stroke.thickness // 2, stroke.color, -1)

        if highlight:
            x0, y0, x1, y1 = self.bbox()
            pad = 8
            cv2.rectangle(
                frame,
                (int(x0) - pad, int(y0) - pad),
                (int(x1) + pad, int(y1) + pad),
                config.COLOR_GRAB_HIGHLIGHT,
                2,
                cv2.LINE_AA,
            )


class Canvas:
    """Holds committed objects and the in-progress stroke of each hand."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.objects: List[DrawObject] = []
        self._live: Dict[str, dict] = {}

    # -- live stroke lifecycle ---------------------------------------------
    def begin_stroke(self, hand_label: str, point: np.ndarray, color, thickness: int) -> None:
        self._live[hand_label] = {
            "points": [np.asarray(point, dtype=float)],
            "color": tuple(color),
            "thickness": int(thickness),
        }

    def add_point(self, hand_label: str, point: np.ndarray) -> None:
        state = self._live.get(hand_label)
        if state is None:
            return
        point = np.asarray(point, dtype=float)
        last = state["points"][-1]
        if np.linalg.norm(point - last) >= config.MIN_POINT_DISTANCE:
            state["points"].append(point)

    def is_drawing(self, hand_label: str) -> bool:
        return hand_label in self._live

    def live_labels(self) -> List[str]:
        """Labels of hands currently mid-stroke."""
        return list(self._live.keys())

    def end_stroke(self, hand_label: str) -> Optional[DrawObject]:
        """Commit the hand's live stroke into an object and return it."""
        state = self._live.pop(hand_label, None)
        if state is None or len(state["points"]) < 2:
            return None
        stroke = Stroke(state["points"], state["color"], state["thickness"])
        return self._commit(stroke)

    def cancel_stroke(self, hand_label: str) -> None:
        self._live.pop(hand_label, None)

    # -- object management --------------------------------------------------
    def _commit(self, stroke: Stroke) -> DrawObject:
        """Add a stroke, merging it into a nearby object or starting a new one."""
        sb = _bbox(stroke.as_array())
        best: Optional[DrawObject] = None
        best_gap = config.GROUP_DISTANCE
        for obj in self.objects:
            gap = _bbox_gap(sb, obj.bbox())
            if gap <= best_gap:
                best, best_gap = obj, gap
        if best is None:
            best = DrawObject(stroke)
            self.objects.append(best)
        else:
            best.add_stroke(stroke)
        return best

    def find_object(self, point: np.ndarray, max_distance: float = 60.0) -> Optional[DrawObject]:
        """Nearest object to a point, within ``max_distance`` (0 gap wins first)."""
        point = np.asarray(point, dtype=float)
        best: Optional[DrawObject] = None
        best_dist = max_distance
        for obj in self.objects:
            d = obj.distance_to(point)
            if d <= best_dist:
                best, best_dist = obj, d
        return best

    def undo(self) -> bool:
        if self.objects:
            self.objects.pop()
            return True
        return False

    def clear(self) -> None:
        self.objects.clear()
        self._live.clear()

    # -- rendering ----------------------------------------------------------
    def render(self, frame: np.ndarray, highlighted: Optional[DrawObject] = None) -> None:
        import cv2

        for obj in self.objects:
            obj.render(frame, highlight=(obj is highlighted))

        # Live strokes on top.
        for state in self._live.values():
            pts = np.asarray(state["points"], dtype=float).astype(np.int32)
            if len(pts) >= 2:
                cv2.polylines(frame, [pts], False, state["color"], state["thickness"], cv2.LINE_AA)
            cv2.circle(frame, tuple(pts[-1]), state["thickness"] // 2 + 2, (255, 255, 255), -1, cv2.LINE_AA)

    def render_transparent(self) -> np.ndarray:
        """Render just the drawings on a transparent BGRA image (for export)."""
        import cv2

        layer = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        bgr = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        for obj in self.objects:
            obj.render(bgr, highlight=False)
        drawn = bgr.any(axis=2)
        mask[drawn] = 255
        layer[:, :, :3] = bgr
        layer[:, :, 3] = mask
        return layer
