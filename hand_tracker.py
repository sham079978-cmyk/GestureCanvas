"""MediaPipe-based dual-hand tracking wrapper.

This is the only module that talks to MediaPipe. It converts raw MediaPipe
results into a small, plain ``Hand`` dataclass so the rest of the application
(gestures, canvas, analytics) never has to import MediaPipe and can be tested
with synthetic landmarks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

try:  # MediaPipe is optional at import time so the logic modules stay testable.
    import mediapipe as mp

    _MP_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only when mediapipe is absent
    mp = None
    _MP_AVAILABLE = False

import config


# Landmark indices, re-exported for convenience.
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

# Pairs of landmarks that form the hand skeleton, used for drawing.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (5, 9), (9, 10), (10, 11), (11, 12),       # middle
    (9, 13), (13, 14), (14, 15), (15, 16),     # ring
    (13, 17), (17, 18), (18, 19), (19, 20),    # pinky
    (0, 17),                                    # palm
]


@dataclass
class Hand:
    """A single detected hand.

    Attributes:
        label: ``"Left"`` or ``"Right"`` for the *user's* hand. The frame is
            mirrored before detection, so MediaPipe's handedness is correct.
        score: handedness confidence in [0, 1].
        landmarks: (21, 2) float array of normalised [0, 1] coordinates.
        pixels: (21, 2) int array of pixel coordinates in the frame.
    """

    label: str
    score: float
    landmarks: np.ndarray
    pixels: np.ndarray

    def point(self, index: int) -> np.ndarray:
        """Return the pixel coordinate of a single landmark."""
        return self.pixels[index]

    @property
    def index_tip(self) -> np.ndarray:
        return self.pixels[INDEX_TIP]

    @property
    def center(self) -> np.ndarray:
        """Rough centre of the palm, used for placing HUD labels."""
        return self.pixels[[WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]].mean(axis=0)


class HandTracker:
    """Thin wrapper around ``mediapipe.solutions.hands``."""

    def __init__(
        self,
        max_hands: int = config.MAX_NUM_HANDS,
        model_complexity: int = config.MODEL_COMPLEXITY,
        min_detection_confidence: float = config.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = config.MIN_TRACKING_CONFIDENCE,
    ) -> None:
        if not _MP_AVAILABLE:
            raise RuntimeError(
                "mediapipe is not installed. Run `pip install -r requirements.txt`."
            )

        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame_bgr: np.ndarray) -> List[Hand]:
        """Detect hands in a BGR frame and return a list of :class:`Hand`."""
        h, w = frame_bgr.shape[:2]
        rgb = frame_bgr[:, :, ::-1]  # BGR -> RGB without an extra copy
        results = self._hands.process(rgb)

        hands: List[Hand] = []
        if not results.multi_hand_landmarks:
            return hands

        handedness = results.multi_handedness or []
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            landmarks = np.array(
                [[lm.x, lm.y] for lm in hand_landmarks.landmark], dtype=np.float64
            )
            pixels = np.column_stack(
                (landmarks[:, 0] * w, landmarks[:, 1] * h)
            ).astype(np.int32)

            label = "Right"
            score = 0.0
            if i < len(handedness):
                classification = handedness[i].classification[0]
                label = classification.label
                score = float(classification.score)

            hands.append(Hand(label=label, score=score, landmarks=landmarks, pixels=pixels))

        return hands

    def close(self) -> None:
        self._hands.close()

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def draw_skeleton(frame_bgr: np.ndarray, hand: Hand, color=(0, 255, 0)) -> None:
    """Draw the hand skeleton and joints onto a frame in place."""
    import cv2

    for a, b in HAND_CONNECTIONS:
        pa = tuple(int(v) for v in hand.pixels[a])
        pb = tuple(int(v) for v in hand.pixels[b])
        cv2.line(frame_bgr, pa, pb, color, 2, cv2.LINE_AA)

    for x, y in hand.pixels:
        cv2.circle(frame_bgr, (int(x), int(y)), 3, (255, 255, 255), -1, cv2.LINE_AA)
