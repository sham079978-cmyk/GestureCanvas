"""Rule-based gesture classification from hand landmarks.

Deliberately free of any MediaPipe dependency: it operates on a plain
``(21, 2)`` array of normalised landmark coordinates, which makes it easy to
unit-test with synthetic hands.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict

import numpy as np

import config


# Landmark indices (mirrors hand_tracker but kept local so this module is standalone).
WRIST = 0
THUMB_MCP, THUMB_IP, THUMB_TIP = 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_MCP, RING_PIP, RING_TIP = 13, 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20


class Gesture(str, Enum):
    NONE = "none"
    PINCH = "pinch"
    POINT = "point"
    PEACE = "peace"
    OPEN_PALM = "open_palm"
    FIST = "fist"


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


class GestureRecognizer:
    """Classifies a hand into one of :class:`Gesture`.

    Pinch detection uses hysteresis: it takes a tighter pinch to *enter* the
    pinch state than to stay in it, which stops the drawing stroke from
    flickering on and off at the threshold.
    """

    def __init__(
        self,
        pinch_ratio: float = config.PINCH_RATIO,
        release_ratio: float = config.PINCH_RELEASE_RATIO,
        extend_ratio: float = config.FINGER_EXTEND_RATIO,
    ) -> None:
        self.pinch_ratio = pinch_ratio
        self.release_ratio = release_ratio
        self.extend_ratio = extend_ratio
        self._pinching: Dict[str, bool] = {}

    # -- geometry helpers ---------------------------------------------------
    @staticmethod
    def hand_scale(landmarks: np.ndarray) -> float:
        """Reference length (wrist -> middle knuckle), used to normalise distances."""
        scale = _dist(landmarks[WRIST], landmarks[MIDDLE_MCP])
        return scale if scale > 1e-6 else 1e-6

    def pinch_strength(self, landmarks: np.ndarray) -> float:
        """Normalised thumb-tip/index-tip distance. Small value == pinching."""
        return _dist(landmarks[THUMB_TIP], landmarks[INDEX_TIP]) / self.hand_scale(landmarks)

    def fingers_up(self, landmarks: np.ndarray) -> Dict[str, bool]:
        """Return which fingers are extended.

        Fingers 2-5: a finger is up when its tip is further from the wrist than
        its PIP joint. Thumb: up when the tip has moved away from the index
        knuckle relative to the IP joint.
        """
        up = {}
        for name, tip, pip in (
            ("index", INDEX_TIP, INDEX_PIP),
            ("middle", MIDDLE_TIP, MIDDLE_PIP),
            ("ring", RING_TIP, RING_PIP),
            ("pinky", PINKY_TIP, PINKY_PIP),
        ):
            up[name] = _dist(landmarks[tip], landmarks[WRIST]) > self.extend_ratio * _dist(
                landmarks[pip], landmarks[WRIST]
            )
        up["thumb"] = _dist(landmarks[THUMB_TIP], landmarks[INDEX_MCP]) > self.extend_ratio * _dist(
            landmarks[THUMB_IP], landmarks[INDEX_MCP]
        )
        return up

    # -- classification -----------------------------------------------------
    def classify(self, hand_label: str, landmarks: np.ndarray) -> Gesture:
        landmarks = np.asarray(landmarks, dtype=float)

        # Pinch with hysteresis, tracked per hand. A pinch only counts if the
        # index finger is reaching forward - otherwise a closed fist (where the
        # thumb tip also rests near the index tip) would read as a pinch.
        strength = self.pinch_strength(landmarks)
        index_forward = _dist(landmarks[INDEX_TIP], landmarks[WRIST]) > _dist(
            landmarks[INDEX_MCP], landmarks[WRIST]
        )
        was_pinching = self._pinching.get(hand_label, False)
        threshold = self.release_ratio if was_pinching else self.pinch_ratio
        pinching = index_forward and strength < threshold
        self._pinching[hand_label] = pinching
        if pinching:
            return Gesture.PINCH

        up = self.fingers_up(landmarks)
        four = [up["index"], up["middle"], up["ring"], up["pinky"]]
        n_up = sum(four)

        if n_up == 0 and not up["thumb"]:
            return Gesture.FIST
        if up["index"] and up["middle"] and not up["ring"] and not up["pinky"]:
            return Gesture.PEACE
        if up["index"] and n_up == 1:
            return Gesture.POINT
        if n_up == 4:
            return Gesture.OPEN_PALM
        return Gesture.NONE

    def reset(self, hand_label: str | None = None) -> None:
        if hand_label is None:
            self._pinching.clear()
        else:
            self._pinching.pop(hand_label, None)
