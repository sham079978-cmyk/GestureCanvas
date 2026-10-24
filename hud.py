"""On-screen heads-up display: mode bar, per-hand labels and analytics panel."""

from __future__ import annotations

from typing import Dict

import cv2
import numpy as np

import config

FONT = cv2.FONT_HERSHEY_SIMPLEX
AA = cv2.LINE_AA
TEXT = (235, 235, 235)
MUTED = (170, 170, 170)
ACCENT_DRAW = (90, 210, 110)
ACCENT_MOVE = (0, 205, 255)


def _panel(frame: np.ndarray, x: int, y: int, w: int, h: int, alpha: float = 0.55) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (18, 18, 18), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def _bar(frame, x, y, w, h, frac, color) -> None:
    frac = float(np.clip(frac, 0.0, 1.0))
    cv2.rectangle(frame, (x, y), (x + w, y + h), (70, 70, 70), -1)
    cv2.rectangle(frame, (x, y), (x + int(w * frac), y + h), color, -1)


def draw_top_bar(frame: np.ndarray, mode: str, fps: float) -> None:
    h, w = frame.shape[:2]
    _panel(frame, 0, 0, w, 56, alpha=0.6)

    color = ACCENT_MOVE if mode == "MOVE" else ACCENT_DRAW
    cv2.putText(frame, f"MODE: {mode}", (18, 38), FONT, 0.9, color, 2, AA)
    cv2.putText(
        frame,
        "m / hold peace: switch   pinch: draw & move   u: undo   c: clear   a: stats   s: save   q: quit",
        (250, 36),
        FONT,
        0.52,
        MUTED,
        1,
        AA,
    )
    cv2.putText(frame, f"{fps:5.1f} FPS", (w - 150, 36), FONT, 0.7, TEXT, 2, AA)


def draw_hand_labels(frame: np.ndarray, hands, gestures: Dict[str, str]) -> None:
    for hand in hands:
        label = hand.label
        cx, cy = hand.center
        color = config.COLOR_RIGHT if label == "Right" else config.COLOR_LEFT
        gesture = gestures.get(label, "")
        text = f"{label}: {gesture}"
        (tw, _), _ = cv2.getTextSize(text, FONT, 0.6, 2)
        cv2.rectangle(frame, (int(cx) - 6, int(cy) - 34), (int(cx) + tw + 6, int(cy) - 8), (0, 0, 0), -1)
        cv2.putText(frame, text, (int(cx), int(cy) - 14), FONT, 0.6, color, 2, AA)


def draw_analysis(frame: np.ndarray, analyzer, gestures: Dict[str, str]) -> None:
    labels = analyzer.labels()
    if not labels:
        return

    h, w = frame.shape[:2]
    pw = 340
    ph = 60 + 150 * len(labels)
    x0 = w - pw - 12
    y0 = 70
    _panel(frame, x0, y0, pw, ph, alpha=0.62)
    cv2.putText(frame, "HAND MOVEMENT ANALYSIS", (x0 + 14, y0 + 30), FONT, 0.6, ACCENT_MOVE, 2, AA)

    for i, label in enumerate(labels):
        s = analyzer.summary(label)
        if s is None:
            continue
        top = y0 + 52 + i * 150
        color = config.COLOR_RIGHT if label == "Right" else config.COLOR_LEFT

        cv2.putText(frame, f"{label}  ({gestures.get(label, '-')})", (x0 + 14, top + 16), FONT, 0.62, color, 2, AA)

        rows = [
            ("speed", s["speed_now"] / 1500.0, f"{s['speed_now']:6.0f} px/s"),
            ("avg", s["speed_avg"] / 1500.0, f"{s['speed_avg']:6.0f} px/s"),
            ("direct", s["efficiency"], f"{s['efficiency'] * 100:5.0f} %"),
            ("smooth", s["steadiness"], f"{s['steadiness'] * 100:5.0f} %"),
        ]
        for j, (name, frac, value) in enumerate(rows):
            ry = top + 34 + j * 22
            cv2.putText(frame, name, (x0 + 14, ry + 12), FONT, 0.45, MUTED, 1, AA)
            _bar(frame, x0 + 78, ry, 150, 12, frac, color)
            cv2.putText(frame, value, (x0 + 238, ry + 12), FONT, 0.45, TEXT, 1, AA)

        cv2.putText(
            frame,
            f"path {s['path_length']:8.0f} px   dominant: {s['dominant_gesture']}",
            (x0 + 14, top + 138),
            FONT,
            0.45,
            MUTED,
            1,
            AA,
        )


def draw_mode_hint(frame: np.ndarray, mode: str) -> None:
    h, w = frame.shape[:2]
    text = (
        "PINCH to draw"
        if mode == "DRAW"
        else "PINCH on an object to grab & move it"
    )
    cv2.putText(frame, text, (24, h - 24), FONT, 0.7, TEXT, 2, AA)
