"""
Hand Gesture Drawing App  v3
============================
Draw on screen using webcam + MediaPipe hand tracking.

Fixed in v3:
  - Fist detection is now MUCH stricter (curl ratio + y-axis check)
  - Exit only triggers after holding fist for 1.5 seconds (countdown shown)
  - Drawing gesture (index+middle pinch) can't accidentally look like a fist

Compatible with:
    Python 3.11
    mediapipe >= 0.10
    opencv-python >= 4.8
    numpy >= 1.24

Installation:
    pip install opencv-python "mediapipe>=0.10" numpy
"""

import cv2
import numpy as np
import time
import os
import urllib.request

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarkerOptions, RunningMode

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
WEBCAM_ID         = 0
FRAME_W, FRAME_H  = 1280, 720
BRUSH_THICKNESS   = 6
DRAW_COLOR        = (0, 200, 255)   # BGR cyan-yellow
SMOOTHING         = 0.55            # EMA smoothing (0=max smooth, 1=raw)

# How long (seconds) the user must HOLD a fist before exit triggers
FIST_HOLD_SECONDS = 1.5

# Index+middle tips closer than this fraction of frame width → draw mode
DRAW_THRESHOLD    = 0.055

# MediaPipe landmark indices
INDEX_TIP  = 8;  INDEX_PIP  = 6;  INDEX_MCP  = 5
MIDDLE_TIP = 12; MIDDLE_PIP = 10; MIDDLE_MCP = 9
RING_TIP   = 16; RING_PIP   = 14; RING_MCP   = 13
PINKY_TIP  = 20; PINKY_PIP  = 18; PINKY_MCP  = 17
THUMB_TIP  = 4;  THUMB_IP   = 3;  THUMB_MCP  = 2
WRIST      = 0

# Model download
_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
_MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "hand_landmarker.task")


# ──────────────────────────────────────────────
# MODEL DOWNLOAD
# ──────────────────────────────────────────────

def ensure_model() -> str:
    if os.path.exists(_MODEL_FILE):
        return _MODEL_FILE
    pkg_dir  = os.path.dirname(mp.__file__)
    bundled  = os.path.join(pkg_dir, "modules", "hand_landmarker",
                            "hand_landmarker.task")
    if os.path.exists(bundled):
        return bundled
    print("Downloading hand_landmarker.task (~9 MB, one-time) ...")
    urllib.request.urlretrieve(_MODEL_URL, _MODEL_FILE)
    print("Download complete.\n")
    return _MODEL_FILE


# ──────────────────────────────────────────────
# GEOMETRY HELPERS
# ──────────────────────────────────────────────

def euclidean(p1, p2):
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def lm_px(landmark, w, h):
    """Normalised landmark → (pixel_x, pixel_y)."""
    return int(landmark.x * w), int(landmark.y * h)


def finger_curl_ratio(tip, pip, mcp, w, h):
    """
    Returns a 0-1 value describing how curled a finger is.
    1.0 = fully curled (tip close to MCP),  0.0 = fully extended.
    We compare tip-to-MCP distance vs PIP-to-MCP distance.
    Curled fingers have tip closer to MCP than their PIP joint.
    """
    tip_pt = lm_px(tip, w, h)
    pip_pt = lm_px(pip, w, h)
    mcp_pt = lm_px(mcp, w, h)
    d_tip  = euclidean(tip_pt, mcp_pt)
    d_pip  = euclidean(pip_pt, mcp_pt)
    if d_pip < 1e-6:
        return 0.0
    # ratio < 1 means tip is closer to MCP than PIP is → curled
    ratio = d_tip / (d_pip * 2.0)
    return max(0.0, min(1.0, 1.0 - ratio))


def is_fist(landmarks, w, h) -> bool:
    """
    Strict fist detection using TWO independent checks — both must pass:

    Check 1 — CURL RATIO:
        Each of the 4 fingers must have a curl ratio >= 0.45
        (tip is significantly closer to MCP than its fully extended position).

    Check 2 — Y-AXIS:
        Each fingertip must be BELOW its PIP joint in pixel space
        (positive y = downward on screen).

    Using both checks together avoids false positives from angled hands.
    """
    fingers = [
        (landmarks[INDEX_TIP],  landmarks[INDEX_PIP],  landmarks[INDEX_MCP]),
        (landmarks[MIDDLE_TIP], landmarks[MIDDLE_PIP], landmarks[MIDDLE_MCP]),
        (landmarks[RING_TIP],   landmarks[RING_PIP],   landmarks[RING_MCP]),
        (landmarks[PINKY_TIP],  landmarks[PINKY_PIP],  landmarks[PINKY_MCP]),
    ]

    CURL_THRESHOLD = 0.42   # how curled each finger must be (0–1)

    for tip_lm, pip_lm, mcp_lm in fingers:
        # Check 1: curl ratio
        curl = finger_curl_ratio(tip_lm, pip_lm, mcp_lm, w, h)
        if curl < CURL_THRESHOLD:
            return False

        # Check 2: tip y must be >= pip y  (tip below or at pip pixel row)
        tip_y = lm_px(tip_lm, w, h)[1]
        pip_y = lm_px(pip_lm, w, h)[1]
        if tip_y < pip_y - 15:   # allow 15 px slack for slightly angled hands
            return False

    return True


# ──────────────────────────────────────────────
# DRAWING / UI HELPERS
# ──────────────────────────────────────────────

def draw_skeleton(frame, landmarks, w, h):
    CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (5,9),(9,10),(10,11),(11,12),
        (9,13),(13,14),(14,15),(15,16),
        (13,17),(17,18),(18,19),(19,20),
        (0,17),
    ]
    pts = [lm_px(landmarks[i], w, h) for i in range(21)]
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (55, 55, 55), 1, cv2.LINE_AA)
    for p in pts:
        cv2.circle(frame, p, 2, (100, 100, 100), -1, cv2.LINE_AA)


def draw_hud(frame, drawing_mode, brush_size, color, fist_progress=0.0):
    """
    Top status bar.
    fist_progress: 0.0–1.0, shows exit countdown bar when > 0.
    """
    h_frame, w_frame = frame.shape[:2]

    # Dark top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w_frame, 46), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    label = "DRAWING" if drawing_mode else " HOVER "
    col   = (0, 220, 100) if drawing_mode else (160, 160, 160)
    cv2.putText(frame, label, (14, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, 2, cv2.LINE_AA)

    info = f"Brush:{brush_size}px  [+/-]size  [c]clear  [q]quit"
    cv2.putText(frame, info, (160, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)

    # Colour swatch
    cv2.circle(frame, (w_frame - 40, 23), 12, color, -1)
    cv2.circle(frame, (w_frame - 40, 23), 13, (255, 255, 255), 1)

    # Fist exit countdown bar at the bottom
    if fist_progress > 0.0:
        bar_w = int(w_frame * fist_progress)
        cv2.rectangle(frame, (0, h_frame - 8), (bar_w, h_frame),
                      (0, 60, 255), -1)
        secs_left = FIST_HOLD_SECONDS * (1.0 - fist_progress)
        msg = f"HOLD FIST to EXIT  ({secs_left:.1f}s)"
        cv2.putText(frame, msg,
                    (w_frame // 2 - 170, h_frame - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 80, 255), 2, cv2.LINE_AA)


# ──────────────────────────────────────────────
# DETECTOR  (mediapipe >= 0.10 Tasks API)
# ──────────────────────────────────────────────

class HandDetector:
    def __init__(self, model_path: str):
        base_opts = mp_python.BaseOptions(model_asset_path=model_path)
        opts = HandLandmarkerOptions(
            base_options                  = base_opts,
            running_mode                  = RunningMode.VIDEO,
            num_hands                     = 1,
            min_hand_detection_confidence = 0.65,
            min_hand_presence_confidence  = 0.65,
            min_tracking_confidence       = 0.55,
        )
        self._det   = mp_vision.HandLandmarker.create_from_options(opts)
        self._ts_ms = 0

    def detect(self, rgb_frame: np.ndarray):
        self._ts_ms += 33   # strictly increasing ms timestamp
        img    = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._det.detect_for_video(img, self._ts_ms)
        return result.hand_landmarks   # list[list[NormalizedLandmark]]

    def close(self):
        self._det.close()


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    model_path = ensure_model()
    print(f"Model: {model_path}\n")

    # Camera
    cap = cv2.VideoCapture(WEBCAM_ID)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam id={WEBCAM_ID}.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, 30)

    ok, sample = cap.read()
    if not ok:
        raise RuntimeError("Cannot read from webcam.")
    h, w = sample.shape[:2]
    print(f"Resolution: {w}x{h}")

    # Canvas
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # Detector
    detector = HandDetector(model_path)

    # Drawing state
    prev_x, prev_y     = None, None
    smooth_x, smooth_y = 0.0,  0.0
    drawing_mode       = False
    brush_size         = BRUSH_THICKNESS
    color              = DRAW_COLOR

    # Fist hold timer
    fist_start_time = None   # when the current fist gesture started

    # FPS
    fps, fps_frames, fps_time = 0.0, 0, time.time()

    print("\nRunning!  Gestures:")
    print("  ✏  Index + Middle CLOSE  → draw")
    print("  ✋  Fingers APART         → stop")
    print(f"  ✊  HOLD fist {FIST_HOLD_SECONDS}s         → exit")
    print("  Keys: [c] clear  [+/-] brush  [q] quit\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        all_hands = detector.detect(rgb)

        fist_progress = 0.0   # progress bar fill (0–1)

        if all_hands:
            landmarks = all_hands[0]

            draw_skeleton(frame, landmarks, w, h)

            # ── Fist check ───────────────────────
            fist_detected = is_fist(landmarks, w, h)

            if fist_detected:
                if fist_start_time is None:
                    fist_start_time = time.time()   # start the hold timer

                held = time.time() - fist_start_time
                fist_progress = min(held / FIST_HOLD_SECONDS, 1.0)

                if held >= FIST_HOLD_SECONDS:
                    # Confirmed exit — show final message
                    cv2.putText(frame, "FIST HELD  Exiting ...",
                                (w // 2 - 190, h // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                (0, 50, 255), 3, cv2.LINE_AA)
                    cv2.imshow("Hand Draw", frame)
                    cv2.waitKey(800)
                    break

                # Reset drawing while fist is held
                prev_x, prev_y = None, None
                drawing_mode   = False

            else:
                # Not a fist — reset the hold timer
                fist_start_time = None

                # ── Draw gesture ─────────────────
                ix, iy = lm_px(landmarks[INDEX_TIP],  w, h)
                mx, my = lm_px(landmarks[MIDDLE_TIP], w, h)

                tip_dist     = euclidean((ix, iy), (mx, my)) / w
                drawing_mode = tip_dist < DRAW_THRESHOLD

                # EMA smoothing
                if prev_x is None:
                    smooth_x, smooth_y = float(ix), float(iy)
                else:
                    smooth_x = SMOOTHING * ix + (1 - SMOOTHING) * smooth_x
                    smooth_y = SMOOTHING * iy + (1 - SMOOTHING) * smooth_y

                cur_x, cur_y = int(smooth_x), int(smooth_y)

                if drawing_mode:
                    if prev_x is not None:
                        cv2.line(canvas, (prev_x, prev_y), (cur_x, cur_y),
                                 color, brush_size, cv2.LINE_AA)
                        cv2.circle(canvas, (cur_x, cur_y),
                                   brush_size // 2, color, -1, cv2.LINE_AA)
                    prev_x, prev_y = cur_x, cur_y
                else:
                    prev_x, prev_y = None, None

                # Cursor ring
                ring_col = (0, 220, 100) if drawing_mode else (80, 80, 255)
                cv2.circle(frame, (cur_x, cur_y), brush_size + 4, ring_col, 2)
                cv2.circle(frame, (cur_x, cur_y), 3, (255, 255, 255), -1)

        else:
            # No hand visible
            prev_x, prev_y  = None, None
            drawing_mode    = False
            fist_start_time = None

        # ── Composite canvas ──────────────────
        gray     = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask  = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        frame    = cv2.add(cv2.bitwise_and(frame,  frame,  mask=mask_inv),
                           cv2.bitwise_and(canvas, canvas, mask=mask))

        # ── HUD ──────────────────────────────
        draw_hud(frame, drawing_mode, brush_size, color, fist_progress)

        fps_frames += 1
        if time.time() - fps_time >= 1.0:
            fps        = fps_frames / (time.time() - fps_time)
            fps_frames = 0
            fps_time   = time.time()
        cv2.putText(frame, f"{fps:.0f} fps", (w - 90, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (70, 70, 70), 1, cv2.LINE_AA)

        cv2.imshow("Hand Draw", frame)

        # ── Keys ─────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quit.")
            break
        elif key == ord('c'):
            canvas[:] = 0
            print("Canvas cleared.")
        elif key in (ord('+'), ord('=')):
            brush_size = min(brush_size + 2, 60)
        elif key == ord('-'):
            brush_size = max(brush_size - 2, 2)

    # Cleanup
    detector.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Goodbye!")


if __name__ == "__main__":
    main()