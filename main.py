"""GestureCanvas - draw and move objects in the air with two hands.

Run:
    python main.py

Controls:
    pinch (thumb + index)   draw a stroke  (DRAW mode) / grab & move an object (MOVE mode)
    m  or hold a peace sign  switch between DRAW and MOVE
    u                        undo last object
    c                        clear the canvas
    a                        toggle the movement-analysis panel
    s                        save a snapshot
    q / Esc                  quit
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

import config
import hud
from canvas import Canvas, DrawObject
from gestures import Gesture, GestureRecognizer
from hand_tracker import HandTracker, draw_skeleton

MODE_DRAW = "DRAW"
MODE_MOVE = "MOVE"


class GestureCanvasApp:
    def __init__(self, camera_index: int = config.CAMERA_INDEX) -> None:
        self.camera_index = camera_index
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open camera {camera_index}. "
                "Check that a webcam is connected and not in use by another app."
            )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

        self.tracker = HandTracker()
        self.recognizer = GestureRecognizer()
        self.canvas = Canvas(config.FRAME_WIDTH, config.FRAME_HEIGHT)
        from movement_analyzer import MovementAnalyzer

        self.analyzer = MovementAnalyzer()

        self.mode = MODE_DRAW
        self.show_analysis = config.SHOW_ANALYSIS
        self.grabbed: Dict[str, Tuple[DrawObject, np.ndarray]] = {}
        self.gestures: Dict[str, str] = {}

        self._peace_start: Optional[float] = None
        self._peace_fired = False

        self._prev_time = time.time()
        self.fps = 0.0

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _color_for(label: str):
        return config.COLOR_RIGHT if label == "Right" else config.COLOR_LEFT

    def _toggle_mode(self) -> None:
        self.mode = MODE_MOVE if self.mode == MODE_DRAW else MODE_DRAW
        # Cancel any in-progress interaction when switching modes.
        for label in self.canvas.live_labels():
            self.canvas.end_stroke(label)
        self.grabbed.clear()

    # -- per-frame logic ----------------------------------------------------
    def _handle_hands(self, hands, now: float) -> None:
        active = set()
        peace_any = False

        for hand in hands:
            label = hand.label
            active.add(label)
            gesture = self.recognizer.classify(label, hand.landmarks)
            self.gestures[label] = gesture.value
            point = hand.index_tip
            self.analyzer.update(label, point, gesture, now)

            if gesture == Gesture.PEACE:
                peace_any = True

            if self.mode == MODE_DRAW:
                self._handle_draw(label, gesture, point)
            else:
                self._handle_move(label, gesture, point)

        # A hand that left the frame should release its stroke / grab.
        for label in list(self.canvas.live_labels()):
            if label not in active:
                self.canvas.end_stroke(label)
        for label in list(self.grabbed):
            if label not in active:
                self.grabbed.pop(label, None)
        for label in list(self.gestures):
            if label not in active:
                self.gestures.pop(label, None)
                self.recognizer.reset(label)

        self._handle_peace_toggle(peace_any, now)

    def _handle_draw(self, label: str, gesture: Gesture, point: np.ndarray) -> None:
        if gesture == Gesture.PINCH:
            if not self.canvas.is_drawing(label):
                self.canvas.begin_stroke(label, point, self._color_for(label), config.BRUSH_THICKNESS)
            self.canvas.add_point(label, point)
        elif self.canvas.is_drawing(label):
            self.canvas.end_stroke(label)

    def _handle_move(self, label: str, gesture: Gesture, point: np.ndarray) -> None:
        if gesture == Gesture.PINCH:
            if label not in self.grabbed:
                obj = self.canvas.find_object(point)
                if obj is not None:
                    self.grabbed[label] = (obj, np.asarray(point, dtype=float))
            else:
                obj, anchor = self.grabbed[label]
                delta = np.asarray(point, dtype=float) - anchor
                obj.translate(float(delta[0]), float(delta[1]))
                self.grabbed[label] = (obj, np.asarray(point, dtype=float))
        else:
            self.grabbed.pop(label, None)

    def _handle_peace_toggle(self, peace_any: bool, now: float) -> None:
        if peace_any:
            if self._peace_start is None:
                self._peace_start = now
                self._peace_fired = False
            elif not self._peace_fired and now - self._peace_start >= config.PEACE_HOLD_SECONDS:
                self._toggle_mode()
                self._peace_fired = True
        else:
            self._peace_start = None
            self._peace_fired = False

    # -- rendering ----------------------------------------------------------
    def _render(self, frame: np.ndarray, hands) -> None:
        for hand in hands:
            draw_skeleton(frame, hand, color=self._color_for(hand.label))
            tip = tuple(int(v) for v in hand.index_tip)
            cv2.circle(frame, tip, 10, self._color_for(hand.label), 2, cv2.LINE_AA)

        highlighted = next((obj for obj, _ in self.grabbed.values()), None)
        self.canvas.render(frame, highlighted)

        hud.draw_top_bar(frame, self.mode, self.fps)
        hud.draw_mode_hint(frame, self.mode)
        if self.show_analysis:
            hud.draw_analysis(frame, self.analyzer, self.gestures)
        else:
            hud.draw_hand_labels(frame, hands, self.gestures)

    # -- main loop ----------------------------------------------------------
    def run(self) -> None:
        cv2.namedWindow("GestureCanvas", cv2.WINDOW_NORMAL)
        try:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    print("Camera frame grab failed; stopping.")
                    break
                if config.FLIP_HORIZONTAL:
                    frame = cv2.flip(frame, 1)
                frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))

                now = time.time()
                self._tick_fps(now)

                hands = self.tracker.process(frame)
                self._handle_hands(hands, now)
                self._render(frame, hands)

                cv2.imshow("GestureCanvas", frame)
                key = cv2.waitKey(1) & 0xFF
                if not self._handle_key(key, frame):
                    break
        finally:
            self.close()

    def _tick_fps(self, now: float) -> None:
        dt = now - self._prev_time
        self._prev_time = now
        if dt > 1e-6:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)

    def _handle_key(self, key: int, frame: np.ndarray) -> bool:
        if key in (27, ord("q")):
            return False
        if key == ord("m"):
            self._toggle_mode()
        elif key == ord("c"):
            self.canvas.clear()
            self.grabbed.clear()
        elif key == ord("u"):
            self.canvas.undo()
        elif key == ord("a"):
            self.show_analysis = not self.show_analysis
        elif key == ord("s"):
            self._save_snapshot(frame)
        return True

    def _save_snapshot(self, frame: np.ndarray) -> None:
        os.makedirs("snapshots", exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        annotated = os.path.join("snapshots", f"canvas-{stamp}.png")
        art = os.path.join("snapshots", f"art-{stamp}.png")
        cv2.imwrite(annotated, frame)
        cv2.imwrite(art, self.canvas.render_transparent())
        print(f"Saved {annotated} and {art}")

    def close(self) -> None:
        self.cap.release()
        self.tracker.close()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GestureCanvas - air drawing with two hands.")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX, help="camera index")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        app = GestureCanvasApp(camera_index=args.camera)
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return
    app.run()


if __name__ == "__main__":
    main()
