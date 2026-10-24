"""Per-hand movement analytics.

Tracks, for each hand, a rolling window of motion so the app can report:

* current and average speed (px/s),
* total path length and straight-line efficiency (how direct the motion is),
* steadiness (1 - normalised speed variance; higher == smoother),
* how often each gesture is used and the dominant gesture.

Pure Python + NumPy, no MediaPipe, no camera.
"""

from __future__ import annotations

from collections import Counter, deque
from typing import Deque, Dict, Optional, Tuple

import numpy as np

import config


class HandStats:
    """Rolling statistics for one hand."""

    def __init__(self, window: int = config.ANALYSIS_WINDOW) -> None:
        self.window = window
        self.positions: Deque[np.ndarray] = deque(maxlen=window)
        self.times: Deque[float] = deque(maxlen=window)
        self.speeds: Deque[float] = deque(maxlen=window)
        self.path_length = 0.0
        self.frames = 0
        self.gesture_counts: Counter = Counter()
        self.last_gesture = None
        self.last_position: Optional[np.ndarray] = None
        self.last_time: Optional[float] = None

    def update(self, position: np.ndarray, gesture, timestamp: float) -> None:
        position = np.asarray(position, dtype=float)

        if self.last_position is not None and self.last_time is not None:
            dt = timestamp - self.last_time
            if dt > 1e-6:
                step = float(np.linalg.norm(position - self.last_position))
                self.path_length += step
                self.speeds.append(step / dt)

        self.positions.append(position)
        self.times.append(timestamp)
        self.frames += 1
        self.last_position = position
        self.last_time = timestamp

        if gesture is not None:
            key = getattr(gesture, "value", gesture)
            self.gesture_counts[str(key)] += 1
            self.last_gesture = gesture

    # -- derived metrics ----------------------------------------------------
    def current_speed(self) -> float:
        return self.speeds[-1] if self.speeds else 0.0

    def average_speed(self) -> float:
        return float(np.mean(self.speeds)) if self.speeds else 0.0

    def efficiency(self) -> float:
        """Net displacement / path travelled over the window. 1.0 == perfectly straight."""
        if len(self.positions) < 2:
            return 0.0
        net = float(np.linalg.norm(self.positions[-1] - self.positions[0]))
        path = 0.0
        for a, b in zip(self.positions, list(self.positions)[1:]):
            path += float(np.linalg.norm(b - a))
        return float(net / path) if path > 1e-6 else 0.0

    def steadiness(self) -> float:
        """1 - coefficient of variation of speed, clipped to [0, 1]."""
        if len(self.speeds) < 2:
            return 1.0
        mean = float(np.mean(self.speeds))
        if mean < 1e-6:
            return 1.0
        cv = float(np.std(self.speeds)) / mean
        return float(np.clip(1.0 - cv, 0.0, 1.0))

    def dominant_gesture(self) -> str:
        if not self.gesture_counts:
            return "none"
        return self.gesture_counts.most_common(1)[0][0]

    def summary(self) -> Dict[str, object]:
        return {
            "speed_now": self.current_speed(),
            "speed_avg": self.average_speed(),
            "path_length": self.path_length,
            "efficiency": self.efficiency(),
            "steadiness": self.steadiness(),
            "dominant_gesture": self.dominant_gesture(),
            "gesture_counts": dict(self.gesture_counts),
        }


class MovementAnalyzer:
    """Dispatches movement updates to per-hand :class:`HandStats`."""

    def __init__(self, window: int = config.ANALYSIS_WINDOW) -> None:
        self.window = window
        self.hands: Dict[str, HandStats] = {}

    def update(self, hand_label: str, position: np.ndarray, gesture, timestamp: float) -> None:
        stats = self.hands.get(hand_label)
        if stats is None:
            stats = HandStats(self.window)
            self.hands[hand_label] = stats
        stats.update(position, gesture, timestamp)

    def summary(self, hand_label: str) -> Optional[Dict[str, object]]:
        stats = self.hands.get(hand_label)
        return stats.summary() if stats else None

    def labels(self):
        return list(self.hands.keys())

    def reset(self) -> None:
        self.hands.clear()
