"""Offline tests for the camera-free logic: gestures, canvas and analytics.

These run without MediaPipe or a webcam, using synthetic hand landmarks, so the
core behaviour can be verified on any machine:

    python test_logic.py
"""

from __future__ import annotations

import numpy as np

from canvas import Canvas
from gestures import Gesture, GestureRecognizer
from movement_analyzer import MovementAnalyzer


# --------------------------------------------------------------------------
# Synthetic hand builder
# --------------------------------------------------------------------------
def make_hand(extended=("index", "middle", "ring", "pinky"), thumb=False, pinch=False):
    """Build a plausible (21, 2) landmark array.

    ``extended`` names the fingers held straight; ``thumb``/``pinch`` control the
    thumb and whether the thumb tip meets the index tip.
    """
    lm = np.zeros((21, 2), dtype=float)
    lm[0] = (0.50, 0.95)   # wrist
    lm[1] = (0.46, 0.82)   # thumb cmc
    lm[2] = (0.42, 0.72)   # thumb mcp
    lm[5] = (0.44, 0.62)   # index mcp
    lm[9] = (0.50, 0.60)   # middle mcp
    lm[13] = (0.56, 0.62)  # ring mcp
    lm[17] = (0.62, 0.66)  # pinky mcp

    chains = {"index": (5, 6, 7, 8), "middle": (9, 10, 11, 12),
              "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}
    for name, (mcp, pip, dip, tip) in chains.items():
        base = lm[mcp].copy()
        if name in extended:
            lm[pip] = base + (0.00, -0.10)
            lm[dip] = base + (0.00, -0.16)
            lm[tip] = base + (0.00, -0.22)
        else:
            lm[pip] = base + (0.00, -0.06)
            lm[dip] = base + (0.01, -0.02)
            lm[tip] = base + (0.02, 0.02)

    # Thumb.
    if pinch:
        lm[3] = (0.42, 0.66)
        lm[4] = lm[8] + (0.01, 0.02)   # meet the index tip
    elif thumb:
        lm[3] = (0.36, 0.68)
        lm[4] = (0.30, 0.64)
    else:
        lm[3] = (0.40, 0.70)
        lm[4] = (0.42, 0.68)
    return lm


# --------------------------------------------------------------------------
# Gesture tests
# --------------------------------------------------------------------------
def test_gestures():
    r = GestureRecognizer()
    assert r.classify("Right", make_hand(extended=("index", "middle", "ring", "pinky"), thumb=True)) == Gesture.OPEN_PALM
    assert r.classify("Right", make_hand(extended=())) == Gesture.FIST
    assert r.classify("Right", make_hand(extended=("index", "middle"))) == Gesture.PEACE
    assert r.classify("Right", make_hand(extended=("index",))) == Gesture.POINT
    assert r.classify("Left", make_hand(pinch=True)) == Gesture.PINCH
    # Hysteresis: a hand that is pinching stays pinched at a mid distance.
    mid = make_hand(pinch=True)
    mid[4] = mid[8] + (0.11, 0.02)   # ratio ~ 0.34 -> still under release threshold
    assert r.classify("Left", mid) == Gesture.PINCH
    print("gestures: OK")


# --------------------------------------------------------------------------
# Canvas / object tests
# --------------------------------------------------------------------------
def _draw(canvas, label, pts, color=(255, 0, 0)):
    canvas.begin_stroke(label, np.array(pts[0]), color, 8)
    for p in pts[1:]:
        canvas.add_point(label, np.array(p))
    return canvas.end_stroke(label)


def test_canvas():
    c = Canvas(640, 480)
    _draw(c, "Right", [(100, 100), (140, 120), (180, 100)])
    assert len(c.objects) == 1

    # A far-away stroke starts a new object.
    _draw(c, "Left", [(400, 400), (440, 420)])
    assert len(c.objects) == 2

    # A stroke near the first object merges into it.
    _draw(c, "Right", [(120, 140), (150, 150)])
    assert len(c.objects) == 2, len(c.objects)

    # Hit-testing finds the right object and respects distance.
    assert c.find_object(np.array([150.0, 110.0])).id == c.objects[0].id
    assert c.find_object(np.array([900.0, 900.0])) is None

    # Moving an object shifts its bounding box by exactly the delta.
    obj = c.objects[0]
    before = obj.bbox()
    obj.translate(50, -20)
    after = obj.bbox()
    assert abs((after[0] - before[0]) - 50) < 1e-6
    assert abs((after[1] - before[1]) + 20) < 1e-6

    # Strokes that joined an object move with it.
    assert len(obj.strokes) == 2

    assert c.undo() and len(c.objects) == 1
    c.clear()
    assert len(c.objects) == 0
    print("canvas: OK")


# --------------------------------------------------------------------------
# Movement analytics tests
# --------------------------------------------------------------------------
def test_analyzer():
    a = MovementAnalyzer(window=50)
    # Straight, constant-velocity motion -> efficiency ~1.0, path ~= 300 px.
    for i in range(11):
        a.update("Right", np.array([100 + 30 * i, 200.0]), Gesture.PINCH, timestamp=i * 0.1)
    s = a.summary("Right")
    assert abs(s["efficiency"] - 1.0) < 1e-6, s["efficiency"]
    assert abs(s["path_length"] - 300.0) < 1e-6, s["path_length"]
    assert s["speed_avg"] > 0
    assert s["dominant_gesture"] == "pinch"

    # There-and-back motion -> poor efficiency.
    b = MovementAnalyzer(window=50)
    for i in range(6):
        b.update("Left", np.array([0.0, float(i * 20)]), Gesture.NONE, timestamp=i * 0.1)
    for i in range(6):
        b.update("Left", np.array([0.0, float(100 - i * 20)]), Gesture.NONE, timestamp=(6 + i) * 0.1)
    assert b.summary("Left")["efficiency"] < 0.2, b.summary("Left")["efficiency"]
    print("analyzer: OK")


if __name__ == "__main__":
    test_gestures()
    test_canvas()
    test_analyzer()
    print("\nAll logic tests passed.")
