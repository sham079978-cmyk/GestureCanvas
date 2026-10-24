# GestureCanvas

Draw in the air with **two hands** and then **pick the drawing up and move it aside**, using nothing but your webcam and [MediaPipe](https://developers.google.com/mediapipe) hand tracking.

GestureCanvas tracks both hands in real time, turns hand gestures into drawing strokes on a live camera overlay, analyses how each hand moves, and lets you grab any drawn object with a pinch and drag it wherever you like.

---

## Demo

| Mode | Gesture | What happens |
|------|---------|--------------|
| **DRAW** | Pinch (thumb + index) | A stroke is drawn following your index fingertip |
| **MOVE** | Pinch on an object | The object is grabbed and follows your hand; release to drop it |
| **Any** | Hold a peace sign ✌️ (or press `m`) | Switch between DRAW and MOVE |

Both hands work at the same time and draw in different colours, so you can sketch with both at once.

---

## Features

- **Dual-hand tracking** — up to two hands tracked simultaneously via MediaPipe Hands, each with its own colour and state.
- **Gesture-driven drawing** — a pinch starts a stroke, releasing it commits the stroke to the canvas. Jitter is filtered so lines stay clean.
- **Object grouping** — strokes drawn near each other are automatically grouped into a single *object*, so a whole drawing moves as one piece.
- **Grab & move** — in MOVE mode, pinch on any object to grab it and drag it aside; a highlight box shows what is held.
- **Hand movement analysis** — a live panel reports per-hand speed, average speed, path length, motion directness and smoothness, plus the dominant gesture.
- **Two ways to switch mode** — keyboard (`m`) or a held peace sign, so you can stay hands-free.
- **Undo / clear / export** — undo the last object, clear the canvas, or save both an annotated snapshot and a transparent PNG of just your artwork.
- **Configurable** — every threshold (pinch sensitivity, brush size, colours, grouping distance) lives in `config.py`.

---

## How it works

```
webcam frame ──► HandTracker (MediaPipe) ──► 21 landmarks per hand
                     │
                     ▼
             GestureRecognizer ──► PINCH / PEACE / POINT / OPEN_PALM / FIST
                     │
        ┌────────────┴─────────────┐
        ▼                          ▼
   DRAW mode                  MOVE mode
   pinch = draw stroke        pinch = grab object, move = translate it
        │                          │
        └──────────► Canvas (strokes + objects) ◄────────┘
                     │
                     ▼
          MovementAnalyzer ──► speed / path / directness / smoothness
                     │
                     ▼
              rendered frame + HUD
```

- **`hand_tracker.py`** wraps MediaPipe Hands and returns a plain `Hand` dataclass (21 landmarks in normalised and pixel coordinates). The frame is mirrored before detection so the hand labels and on-screen motion feel natural.
- **`gestures.py`** classifies each hand with pure geometry — no ML, no MediaPipe — which makes it unit-testable. Pinch detection uses hysteresis so a stroke doesn't flicker at the threshold, and an "index reaching forward" gate so a closed fist isn't mistaken for a pinch.
- **`canvas.py`** owns the committed objects and the in-progress stroke of each hand. New strokes merge into a nearby object or start a new one; each object carries an `offset` so it can be translated as a whole.
- **`movement_analyzer.py`** keeps a rolling window of positions and times per hand and derives the analytics shown in the HUD.
- **`main.py`** is the app: camera loop, the DRAW/MOVE state machine, keyboard controls and rendering.

---

## Requirements

- Python 3.9 – 3.12
- A working webcam
- `mediapipe`, `opencv-python`, `numpy` (see `requirements.txt`)

## Install

```bash
git clone https://github.com/<your-username>/GestureCanvas.git
cd GestureCanvas

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Run

```bash
python main.py
```

Optional flags:

```bash
python main.py --camera 1        # pick a different camera index
```

### Controls

| Key | Action |
|-----|--------|
| `m` | toggle DRAW / MOVE (or hold a peace sign for ~0.6 s) |
| `u` | undo the last object |
| `c` | clear the canvas |
| `a` | toggle the movement-analysis panel |
| `s` | save a snapshot (annotated frame + transparent artwork) to `snapshots/` |
| `q` / `Esc` | quit |

---

## Tuning

Everything is in `config.py`. The most useful knobs:

| Setting | Effect |
|---------|--------|
| `PINCH_RATIO` / `PINCH_RELEASE_RATIO` | How tightly you must pinch to draw / release |
| `BRUSH_THICKNESS` | Stroke thickness |
| `MIN_POINT_DISTANCE` | Higher = smoother lines, lower = more detail |
| `GROUP_DISTANCE` | How close strokes must be to become one object |
| `COLOR_RIGHT` / `COLOR_LEFT` | Per-hand stroke colours (BGR) |
| `PEACE_HOLD_SECONDS` | How long to hold the peace sign to switch modes |

---

## Tests

The gesture, canvas and analytics logic runs without a camera or MediaPipe:

```bash
python test_logic.py
```

It builds synthetic hand landmarks and checks gesture classification, object grouping, hit-testing, translation and the movement metrics.

---

## Project structure

```
GestureCanvas/
├── main.py               # app entry point, camera loop, state machine
├── hand_tracker.py       # MediaPipe wrapper + Hand dataclass
├── gestures.py           # geometric gesture classification
├── canvas.py             # strokes, objects, grouping, rendering
├── movement_analyzer.py  # per-hand movement analytics
├── hud.py                # on-screen display
├── config.py             # all tunable settings
├── test_logic.py         # camera-free unit tests
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Troubleshooting

- **"Could not open camera"** — another app may be using it; close it or pass `--camera 1`.
- **Drawing feels jittery** — increase `MIN_POINT_DISTANCE` in `config.py`.
- **Pinch triggers too easily / not enough** — adjust `PINCH_RATIO` (lower = must pinch tighter).
- **Poor tracking in low light** — add light; MediaPipe needs reasonably lit hands.

---

## Roadmap ideas

- Shape recognition (snap a rough circle into a clean circle)
- Colour picker gesture
- Multi-user / multi-object selection
- Saving an editable session file

---

## License

MIT — see [LICENSE](LICENSE).
