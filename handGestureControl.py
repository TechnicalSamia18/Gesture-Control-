import math
import cv2
import mediapipe as mp
import numpy as np
import pyautogui
from pycaw.pycaw import AudioUtilities

print("=" * 60)
print("Initializing Combined Gesture Control (Mouse + Volume)...")
print("=" * 60)

# ---------- MediaPipe Hands ----------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
print("✓ MediaPipe hand tracking initialized (2 hands)")
# ---------- Screen Info (for mouse) ----------
screen_width, screen_height = pyautogui.size()
print(f"✓ Screen size: {screen_width} x {screen_height}")

# ---------- Audio Control (pycaw NEW STYLE) ----------
try:
    device = AudioUtilities.GetSpeakers()
    volume = device.EndpointVolume  # direct handle, no Activate()
    vol_range = volume.GetVolumeRange()
    min_vol = vol_range[0]
    max_vol = vol_range[1]
    print("✓ Audio control initialized (EndpointVolume)")
except Exception as e:
    print(f"✗ Audio error: {e}")
    print("Check that pycaw + comtypes are installed for this Python version.")
    # If audio fails, you can still use mouse; we won't exit.
    volume = None
    min_vol = -65.25
    max_vol = 0.0

# ---------- Webcam ----------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("✗ Cannot open webcam")
    raise SystemExit

print("✓ Webcam ready")

# ---------- State ----------
is_muted = False


# ---------- Helper Functions ----------
def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)


def is_palm_open(landmarks, img_shape):
    h, w = img_shape[:2]

    fingertips = [8, 12, 16, 20]
    finger_bases = [5, 9, 13, 17]

    fingers_extended = 0

    for tip, base in zip(fingertips, finger_bases):
        tip_y = landmarks[tip].y * h
        base_y = landmarks[base].y * h
        if tip_y < base_y - 20:
            fingers_extended += 1

    thumb_tip = landmarks[4]
    thumb_base = landmarks[2]
    if abs(thumb_tip.x - thumb_base.x) * w > 30:
        fingers_extended += 1

    return fingers_extended >= 4


def is_fist(landmarks, img_shape):
    h, w = img_shape[:2]

    fingertips = [8, 12, 16, 20]
    palm_base = landmarks[0]

    fingers_closed = 0

    for tip_idx in fingertips:
        tip = landmarks[tip_idx]
        dist = calculate_distance(
            (tip.x * w, tip.y * h),
            (palm_base.x * w, palm_base.y * h)
        )
        if dist < 120:
            fingers_closed += 1

    return fingers_closed >= 3


def get_pinch_distance(landmarks, img_shape):
    h, w = img_shape[:2]

    thumb_tip = landmarks[4]
    index_tip = landmarks[8]

    x1, y1 = int(thumb_tip.x * w), int(thumb_tip.y * h)
    x2, y2 = int(index_tip.x * w), int(index_tip.y * h)

    distance = calculate_distance((x1, y1), (x2, y2))

    return distance, (x1, y1), (x2, y2)


print("\n" + "=" * 60)
print("  COMBINED GESTURE CONTROL ACTIVE")
print("=" * 60)
print("👈 Left hand:")
print("    • Move middle finger MCP = Move Mouse")
print("    • Bend index finger (tip down to PIP) = Left Click")
print("\n👉 Right hand:")
print("    • Open palm (5 fingers) = MUTE")
print("    • Fist = UNMUTE")
print("    • Pinch thumb + index (distance) = Volume")
print("\nPress 'Q' to quit")
print("=" * 60 + "\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    gesture_text = "Show your hands"
    status_color = (255, 255, 255)

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            handedness = results.multi_handedness[idx].classification[0].label

            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2)
            )

            landmarks = hand_landmarks.landmark

            # ----- LEFT HAND: mouse -----
            if handedness == "Left":
                gesture_text = "Left hand: Mouse control"
                status_color = (200, 200, 255)

                index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_mid = landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP]

                mcp = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                cursor_x = int(mcp.x * screen_width)
                cursor_y = int(mcp.y * screen_height)

                pyautogui.moveTo(cursor_x, cursor_y, duration=0.1)

                if index_tip.y >= index_mid.y:
                    pyautogui.click()

            # ----- RIGHT HAND: volume -----
            elif handedness == "Right" and volume is not None:
                if is_palm_open(landmarks, frame.shape):
                    if not is_muted:
                        volume.SetMute(1, None)
                        is_muted = True
                    gesture_text = "Right hand: PALM - MUTED"
                    status_color = (0, 0, 255)
                    cv2.putText(frame, "MUTED", (220, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

                elif is_fist(landmarks, frame.shape):
                    if is_muted:
                        volume.SetMute(0, None)
                        is_muted = False
                    gesture_text = "Right hand: FIST - UNMUTED"
                    status_color = (0, 255, 0)
                    cv2.putText(frame, "PLAYING", (180, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)

                else:
                    distance, (x1, y1), (x2, y2) = get_pinch_distance(landmarks, frame.shape)

                    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                    cv2.circle(frame, (x1, y1), 12, (255, 0, 255), cv2.FILLED)
                    cv2.circle(frame, (x2, y2), 12, (255, 0, 255), cv2.FILLED)
                    cv2.circle(frame, ((x1 + x2) // 2, (y1 + y2) // 2),
                               12, (0, 255, 255), cv2.FILLED)

                    vol_db = np.interp(distance, [20, 200], [min_vol, max_vol])
                    volume.SetMasterVolumeLevel(vol_db, None)

                    vol_percentage = np.interp(distance, [20, 200], [0, 100])
                    gesture_text = f"Right hand: Volume {int(vol_percentage)}%"
                    status_color = (0, 255, 255)

                    bar_x, bar_y = 50, 150
                    bar_w, bar_h = 35, 250

                    cv2.rectangle(frame,
                                  (bar_x, bar_y),
                                  (bar_x + bar_w, bar_y + bar_h),
                                  (255, 255, 255), 3)

                    fill_height = int(np.interp(vol_percentage, [0, 100], [0, bar_h]))
                    cv2.rectangle(frame,
                                  (bar_x, bar_y + bar_h - fill_height),
                                  (bar_x + bar_w, bar_y + bar_h),
                                  (0, 255, 0), cv2.FILLED)

                    cv2.putText(frame, f"{int(vol_percentage)}%",
                                (bar_x - 10, bar_y + bar_h + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 255, 0), 2)

    cv2.rectangle(frame, (0, 0), (640, 60), (50, 50, 50), cv2.FILLED)
    cv2.putText(frame, gesture_text, (15, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)

    cv2.putText(frame, "Press 'Q' to Quit", (450, 470),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Combined Gesture Control", frame)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), ord('Q')):
        break

cap.release()
cv2.destroyAllWindows()
print("\n✓ Combined Gesture Control Stopped. Goodbye!")