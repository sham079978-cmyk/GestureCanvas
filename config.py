"""Central configuration for GestureCanvas.

Every tunable knob lives here so the behaviour can be adjusted without
touching the application logic.
"""

# --------------------------------------------------------------------------
# Camera
# --------------------------------------------------------------------------
CAMERA_INDEX = 0          # which webcam to open (0 = default)
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FLIP_HORIZONTAL = True    # mirror the view so moving right moves right on screen

# --------------------------------------------------------------------------
# MediaPipe Hands
# --------------------------------------------------------------------------
MAX_NUM_HANDS = 2
MODEL_COMPLEXITY = 1              # 0 = fast, 1 = accurate
MIN_DETECTION_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.6

# --------------------------------------------------------------------------
# Gesture tuning
# --------------------------------------------------------------------------
# Pinch = thumb-tip and index-tip brought together. The distance between the
# two tips is normalised by hand size (wrist -> middle-finger knuckle) so it
# works at any distance from the camera.
PINCH_RATIO = 0.42          # enter pinch below this normalised distance
PINCH_RELEASE_RATIO = 0.58  # leave pinch above this (hysteresis avoids flicker)
FINGER_EXTEND_RATIO = 1.10  # tip must be this much further from wrist than the pip

# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------
BRUSH_THICKNESS = 8
MIN_POINT_DISTANCE = 3      # px: ignore jitter while the hand is almost still
ERASE_RADIUS = 45

# Colours are BGR (OpenCV order). One per hand so dual drawing is readable.
COLOR_RIGHT = (70, 70, 255)     # warm red
COLOR_LEFT = (255, 170, 40)     # cool blue
COLOR_GRAB_HIGHLIGHT = (0, 230, 255)   # outline on a grabbed object

# --------------------------------------------------------------------------
# Object grouping
# --------------------------------------------------------------------------
# When a stroke ends, it joins an existing drawn object if it is within this
# many pixels of that object's bounding box, otherwise it starts a new object.
GROUP_DISTANCE = 45

# --------------------------------------------------------------------------
# Movement analysis
# --------------------------------------------------------------------------
ANALYSIS_WINDOW = 30        # frames in the rolling statistics window

# --------------------------------------------------------------------------
# HUD / UX
# --------------------------------------------------------------------------
SHOW_ANALYSIS = True        # start with the analytics panel visible
PEACE_HOLD_SECONDS = 0.6    # hold a peace sign this long to toggle mode
