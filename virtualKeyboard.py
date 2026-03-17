"""
Virtual Keyboard ULTIMATE EDITION
Features:
  - Gesture Typing via Hand Landmarks
  - Voice-to-Text (say 'start' / 'stop')
  - Word Prediction Bar
  - 4 Themes (press T)
  - Emoji Panel (press E)
  - Undo/Redo (press Z / R)
  - Live WPM Counter
  - Stats Overlay (press S)
  - Hotword Expansion
  - Number Row
  - Voice Commands: 'clear all', 'backspace', 'new line',
                    'switch theme', 'emoji panel', 'show stats',
                    'undo', 'redo'
"""

import traceback
import sys
import cv2
import mediapipe as mp
from time import sleep, time
from pynput.keyboard import Controller, Key
import threading
import speech_recognition as sr
import queue
import collections
import os
import math

print("=" * 60)
print("  Virtual Keyboard ULTIMATE - Starting up...")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────
WINDOW_W        = 1280
WINDOW_H        = 720
KB_TOP          = 10
BTN_W, BTN_H    = 85, 85
PINCH_THRESHOLD = 40
CLICK_COOLDOWN  = 0.18
MODEL_PATH = r"C:\Users\AMIN TRADERS\Desktop\Gesture-Control-\hand_landmarker.task"

# ─────────────────────────────────────────────────────────────────
#  THEMES
# ─────────────────────────────────────────────────────────────────
THEMES = {
    "Neon Dark": {
        "bg":          (15,  15,  25),
        "key_normal":  (40,  40,  60),
        "key_special": (60,  20,  80),
        "key_hover":   (0,   200, 180),
        "key_press":   (0,   255, 120),
        "text":        (220, 220, 255),
        "text_press":  (0,   0,   0),
        "panel":       (10,  10,  20),
        "accent":      (0,   255, 180),
        "wpm_color":   (0,   255, 100),
    },
    "Cherry Blossom": {
        "bg":          (255, 230, 235),
        "key_normal":  (255, 200, 210),
        "key_special": (255, 160, 180),
        "key_hover":   (200, 80,  120),
        "key_press":   (255, 100, 140),
        "text":        (80,  20,  40),
        "text_press":  (255, 255, 255),
        "panel":       (255, 215, 220),
        "accent":      (220, 60,  100),
        "wpm_color":   (180, 0,   80),
    },
    "Ocean Blue": {
        "bg":          (5,   20,  50),
        "key_normal":  (10,  50,  100),
        "key_special": (20,  80,  140),
        "key_hover":   (0,   160, 220),
        "key_press":   (0,   220, 255),
        "text":        (180, 220, 255),
        "text_press":  (0,   0,   30),
        "panel":       (5,   15,  40),
        "accent":      (0,   200, 255),
        "wpm_color":   (0,   220, 255),
    },
    "Mint Fresh": {
        "bg":          (230, 255, 245),
        "key_normal":  (200, 240, 220),
        "key_special": (150, 210, 190),
        "key_hover":   (0,   160, 120),
        "key_press":   (0,   200, 150),
        "text":        (20,  80,  60),
        "text_press":  (255, 255, 255),
        "panel":       (210, 245, 230),
        "accent":      (0,   180, 130),
        "wpm_color":   (0,   150, 100),
    },
}
theme_names       = list(THEMES.keys())
current_theme_idx = 0

def theme():
    return THEMES[theme_names[current_theme_idx]]

# ─────────────────────────────────────────────────────────────────
#  WORD PREDICTOR
# ─────────────────────────────────────────────────────────────────
COMMON_WORDS = [
    "the","be","to","of","and","a","in","that","have","it","for","not",
    "on","with","he","as","you","do","at","this","but","his","by","from",
    "they","we","say","her","she","or","an","will","my","one","all","would",
    "there","their","what","so","up","out","if","about","who","get","which",
    "go","me","when","make","can","like","time","no","just","him","know",
    "take","people","into","year","your","good","some","could","them","see",
    "other","than","then","now","look","only","come","its","over","think",
    "also","back","after","use","two","how","our","first","well","way",
    "even","new","want","because","any","these","give","day","most","us",
    "hello","world","please","thank","sorry","help","yes","need","going",
    "today","tomorrow","yesterday","morning","evening","night","home",
    "work","school","food","water","phone","message","email","call","meet",
    "happy","sad","good","bad","okay","sure","maybe","really","very","much",
    "washroom","bathroom","office","meeting","lunch","dinner","break",
    "computer","keyboard","screen","camera","microphone","speaker",
]

class WordPredictor:
    def __init__(self):
        self.word_freq = collections.Counter(COMMON_WORDS)
        self.personal  = collections.Counter()

    def predict(self, prefix, n=4):
        if not prefix:
            return []
        prefix = prefix.lower()
        scored = {}
        for word, freq in self.word_freq.items():
            if word.startswith(prefix) and word != prefix:
                scored[word] = freq + self.personal.get(word, 0) * 5
        for word, freq in self.personal.items():
            if word.startswith(prefix) and word != prefix:
                if word not in scored:
                    scored[word] = freq * 5
        return sorted(scored, key=scored.get, reverse=True)[:n]

    def learn(self, word):
        if len(word) > 1:
            self.personal[word.lower()] += 1

predictor = WordPredictor()

# ─────────────────────────────────────────────────────────────────
#  EMOJI PANEL  (ASCII-safe for OpenCV)
# ─────────────────────────────────────────────────────────────────
EMOJIS = [
    ":)", ":D", ";)", ":P", ":(", ":o", "B)", ":*", "<3", ">:(",
    "(y)", "(n)", "\\o/", "lol", "omg", "wtf", "ily", "brb", "ttyl", "xD",
    "***", "~~~", "---", ">>>", "<<<", "!!!", "???", "...", "^^^", "###",
    "(c)", "(r)", "++", "--", "==", "!=", ">>", "<<", "~_~", "^_^",
]
show_emoji_panel = False

# ─────────────────────────────────────────────────────────────────
#  HOTWORDS
# ─────────────────────────────────────────────────────────────────
HOTWORDS = {
    "thx": "Thank you very much!",
    "brb": "Be right back!",
    "omw": "On my way!",
    "gm":  "Good morning!",
    "gn":  "Good night!",
    "ty":  "Thank you!",
    "np":  "No problem!",
    "idk": "I don't know",
}

# ─────────────────────────────────────────────────────────────────
#  STATS
# ─────────────────────────────────────────────────────────────────
stats = {
    "total_chars":   0,
    "total_words":   0,
    "session_start": time(),
    "wpm_history":   collections.deque(maxlen=60),
}
show_stats = False

def update_stats(char):
    stats["total_chars"] += 1
    if char == " ":
        stats["total_words"] += 1
        elapsed = time() - stats["session_start"]
        wpm = int((stats["total_words"] / elapsed) * 60) if elapsed > 0 else 0
        stats["wpm_history"].append(wpm)

# ─────────────────────────────────────────────────────────────────
#  KEYBOARD CONTROLLER
# ─────────────────────────────────────────────────────────────────
keyboard_ctrl = Controller()
typed_buffer  = []
undo_stack    = []

def undo_last():
    if typed_buffer:
        typed_buffer.pop()
        keyboard_ctrl.press('\b')
        keyboard_ctrl.release('\b')

def redo_last():
    if undo_stack:
        ch = undo_stack.pop()
        typed_buffer.append(ch)
        keyboard_ctrl.press(ch)
        keyboard_ctrl.release(ch)

def type_text_safe(text):
    for ch in text:
        keyboard_ctrl.press(ch)
        keyboard_ctrl.release(ch)
        typed_buffer.append(ch)
        update_stats(ch)
        buf_str = "".join(typed_buffer[-10:])
        for hw, expansion in HOTWORDS.items():
            if buf_str.endswith(hw):
                for _ in hw:
                    keyboard_ctrl.press('\b')
                    keyboard_ctrl.release('\b')
                    if typed_buffer:
                        typed_buffer.pop()
                for ech in (expansion + " "):
                    keyboard_ctrl.press(ech)
                    keyboard_ctrl.release(ech)
                    typed_buffer.append(ech)
                show_notification("Expanded: " + hw + " -> " + expansion[:25])
                return

# ─────────────────────────────────────────────────────────────────
#  NOTIFICATIONS
# ─────────────────────────────────────────────────────────────────
notification_msg  = ""
notification_time = 0.0

def show_notification(msg, duration=2.5):
    global notification_msg, notification_time
    notification_msg  = msg
    notification_time = time() + duration

# ─────────────────────────────────────────────────────────────────
#  VOICE ENGINE
# ─────────────────────────────────────────────────────────────────
voice_active     = False
voice_status_msg = "Say 'start' to activate voice"
voice_queue      = queue.Queue()
recognizer       = sr.Recognizer()
mic              = sr.Microphone()

recognizer.energy_threshold         = 300
recognizer.dynamic_energy_threshold = True
recognizer.pause_threshold          = 0.8

def handle_voice_command(text):
    global voice_active, current_theme_idx, show_emoji_panel, show_stats
    t = text.strip().lower()
    if "stop" in t:
        voice_active = False
        show_notification("Voice OFF")
        return True
    if "clear all" in t:
        for _ in typed_buffer:
            keyboard_ctrl.press('\b')
            keyboard_ctrl.release('\b')
        typed_buffer.clear()
        show_notification("Buffer cleared")
        return True
    if "backspace" in t:
        undo_last()
        return True
    if "new line" in t or "enter" in t:
        keyboard_ctrl.press(Key.enter)
        keyboard_ctrl.release(Key.enter)
        return True
    if "switch theme" in t or "change theme" in t:
        current_theme_idx = (current_theme_idx + 1) % len(theme_names)
        show_notification("Theme: " + theme_names[current_theme_idx])
        return True
    if "emoji panel" in t or "open emoji" in t:
        show_emoji_panel = not show_emoji_panel
        show_notification("Emoji panel " + ("opened" if show_emoji_panel else "closed"))
        return True
    if "show stats" in t or "statistics" in t:
        show_stats = not show_stats
        show_notification("Stats " + ("visible" if show_stats else "hidden"))
        return True
    if "undo" in t:
        undo_last()
        show_notification("Undo")
        return True
    if "redo" in t:
        redo_last()
        show_notification("Redo")
        return True
    return False

def voice_thread():
    """Voice always ON - no start/stop needed, just speak and it types."""
    global voice_active, voice_status_msg

    recognizer.energy_threshold         = 200
    recognizer.dynamic_energy_threshold = False

    try:
        with mic as source:
            print("Calibrating mic... stay quiet for 1 second")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            print("Mic ready! Just speak and it will type.")
    except Exception as e:
        print("Mic error: " + str(e))
        voice_status_msg = "Mic error - check microphone"
        return

    voice_active     = True   # Always active from the start
    voice_status_msg = "LISTENING... speak to type"

    while True:
        try:
            voice_status_msg = "LISTENING... speak to type"
            with mic as source:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=6)

            voice_status_msg = "Recognizing..."
            text = recognizer.recognize_google(audio)
            print("Heard: " + text)

            # Only handle stop command, everything else gets typed
            if text.strip().lower() == "stop typing":
                voice_active     = False
                voice_status_msg = "Voice PAUSED - say 'type' to resume"
            elif text.strip().lower() in ["type", "resume", "start typing"]:
                voice_active     = True
                voice_status_msg = "LISTENING... speak to type"
                show_notification("Voice resumed!")
            elif voice_active:
                if not handle_voice_command(text):
                    voice_queue.put(text + " ")
                    voice_status_msg = 'Typed: "' + text[:40] + '"'
                    show_notification('Typed: "' + text[:40] + '"')

        except sr.WaitTimeoutError:
            voice_status_msg = "LISTENING... speak to type"
        except sr.UnknownValueError:
            voice_status_msg = "LISTENING... (couldn't catch that)"
        except sr.RequestError as e:
            print("API error: " + str(e))
            voice_status_msg = "No internet for voice!"
            sleep(2)
        except Exception as e:
            print("Voice error: " + str(e))
            sleep(0.5)


# ─────────────────────────────────────────────────────────────────
#  CAMERA SELECTION
# ─────────────────────────────────────────────────────────────────
print("\nScanning for available cameras...")
available_cams = []
for i in range(6):
    test = cv2.VideoCapture(i)
    if test.isOpened():
        ret, _ = test.read()
        if ret:
            available_cams.append(i)
            print(f"  Found camera [{i}]")
    test.release()

if not available_cams:
    print("ERROR: No cameras found!")
    input("Press Enter to close...")
    sys.exit(1)

print("\nOpening preview of each camera for 3 seconds...")
print("Watch which window shows your PHONE camera!\n")

for i in available_cams:
    test = cv2.VideoCapture(i)
    start_t = time()
    while time() - start_t < 3:
        ret, frame = test.read()
        if ret:
            cv2.putText(frame, f"Camera Index: {i}", (40, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)
            cv2.putText(frame, "Phone or Laptop camera?", (40, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.imshow("Camera Preview - Check which is your phone", frame)
        cv2.waitKey(1)
    test.release()
cv2.destroyAllWindows()

# Ask user to pick
print("\n" + "=" * 50)
print("Available camera indices: " + str(available_cams))
print("=" * 50)
while True:
    try:
        choice = input("Enter the camera index to use for Virtual Keyboard: ").strip()
        cam_index = int(choice)
        if cam_index in available_cams:
            print(f"Using camera [{cam_index}]")
            break
        else:
            print(f"Invalid! Choose from: {available_cams}")
    except ValueError:
        print("Please enter a number!")

print("Opening camera " + str(cam_index) + "...")
cap = cv2.VideoCapture(cam_index)
cap.set(3, WINDOW_W)
cap.set(4, WINDOW_H)
if not cap.isOpened():
    print("ERROR: Could not open camera " + str(cam_index))
    input("Press Enter to close...")
    sys.exit(1)
print("Camera OK")


print("Loading hand detector...")
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

if not os.path.exists(MODEL_PATH):
    print("ERROR: Model not found at: " + MODEL_PATH)
    input("Press Enter to close...")
    sys.exit(1)

base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options      = mp_vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector     = mp_vision.HandLandmarker.create_from_options(options)
print("Hand detector OK")

print("Starting voice thread...")
try:
    threading.Thread(target=voice_thread, daemon=True).start()
    print("Voice thread started - say 'start' to activate mic")
except Exception as ve:
    print("WARNING: Voice failed to start: " + str(ve))


# ─────────────────────────────────────────────────────────────────
#  KEYBOARD LAYOUT
# ─────────────────────────────────────────────────────────────────
KEYS_ROWS = [
    ["1","2","3","4","5","6","7","8","9","0","<-"],
    ["q","w","e","r","t","y","u","i","o","p"],
    ["caps","a","s","d","f","g","h","j","k","l"],
    ["z","x","c","v"," ","b","n","m","->"],
]

class Button:
    def __init__(self, pos, text):
        self.first_pos  = pos
        self.text       = text
        self.press_anim = 0.0
        if text == " ":
            self.btn_size = [284, BTN_H]
        elif text in ["<-", "caps", "->"]:
            self.btn_size = [120, BTN_H]
        else:
            self.btn_size = [BTN_W, BTN_H]

def build_buttons():
    buttons = []
    for i, row in enumerate(KEYS_ROWS):
        for j, key in enumerate(row):
            if i == 3:
                if key == " ":
                    pos = [int(WINDOW_W / 2 - 142), BTN_H * i + KB_TOP + i * 8]
                elif key in ["b", "n", "m", "->"]:
                    sx = int(WINDOW_W / 2 + 142)
                    ki = ["b", "n", "m", "->"].index(key)
                    pos = [sx + BTN_W * ki, BTN_H * i + KB_TOP + i * 8]
                else:
                    pos = [BTN_W * j + 80, BTN_H * i + KB_TOP + i * 8]
            elif i == 2 and key != "caps":
                pos = [BTN_W * j + 130, BTN_H * i + KB_TOP + i * 8]
            else:
                pos = [BTN_W * j + 80, BTN_H * i + KB_TOP + i * 8]
            buttons.append(Button(pos, key))
    return buttons

buttonList = build_buttons()

# ─────────────────────────────────────────────────────────────────
#  PREDICTION BAR
# ─────────────────────────────────────────────────────────────────
prediction_buttons = []

def update_predictions():
    global prediction_buttons
    buf          = "".join(typed_buffer)
    words        = buf.split()
    current_word = words[-1] if words and not buf.endswith(" ") else ""
    suggestions  = predictor.predict(current_word, n=4) if current_word else []
    prediction_buttons = []
    px = 80
    for s in suggestions:
        w = max(120, len(s) * 18 + 20)
        prediction_buttons.append({"text": s, "rect": (px, 360, px + w, 410)})
        px += w + 10

# ─────────────────────────────────────────────────────────────────
#  DRAW HELPERS
# ─────────────────────────────────────────────────────────────────
def draw_rounded_rect(img, pt1, pt2, color, radius=12):
    x1, y1 = pt1
    x2, y2 = pt2
    r = radius
    cv2.rectangle(img, (x1 + r, y1),     (x2 - r, y2),     color, cv2.FILLED)
    cv2.rectangle(img, (x1,     y1 + r), (x2,     y2 - r), color, cv2.FILLED)
    cv2.circle(img, (x1 + r, y1 + r), r, color, cv2.FILLED)
    cv2.circle(img, (x2 - r, y1 + r), r, color, cv2.FILLED)
    cv2.circle(img, (x1 + r, y2 - r), r, color, cv2.FILLED)
    cv2.circle(img, (x2 - r, y2 - r), r, color, cv2.FILLED)

def blend_colors(c1, c2, t):
    return tuple(int(c1[i] * (1 - t) + c2[i] * t) for i in range(3))

def draw_keyboard(img, buttons, hover=None, caps_on=False):
    t = theme()
    overlay = img.copy()
    cv2.rectangle(overlay, (55, 0), (WINDOW_W - 55, 355), t["bg"], cv2.FILLED)
    img = cv2.addWeighted(overlay, 0.82, img, 0.18, 0)

    for btn in buttons:
        x, y   = btn.first_pos
        w, h   = btn.btn_size
        is_special = btn.text in ["<-", "caps", "->"]
        base_col   = t["key_special"] if is_special else t["key_normal"]

        if btn.press_anim > 0:
            col = blend_colors(base_col, t["key_press"], btn.press_anim)
            btn.press_anim = max(0.0, btn.press_anim - 0.08)
        elif hover == btn:
            col = t["key_hover"]
        else:
            col = base_col

        draw_rounded_rect(img, (x + 4, y + 4), (x + w + 4, y + h + 4), (0, 0, 0))
        draw_rounded_rect(img, (x, y), (x + w, y + h), col)

        if btn.text == " ":
            display = "SPACE"
        elif btn.text == "->":
            display = "ENTER"
        elif btn.text == "<-":
            display = "DEL"
        elif btn.text == "caps":
            display = "CAPS" if caps_on else "caps"
        else:
            display = btn.text

        txt_col = t["text_press"] if hover == btn else t["text"]
        font_sc = 0.5 if btn.text in ["caps", "->", "<-", " "] else 0.75
        (tw, th), _ = cv2.getTextSize(display, cv2.FONT_HERSHEY_SIMPLEX, font_sc, 2)
        tx = x + (w - tw) // 2
        ty = y + (h + th) // 2
        cv2.putText(img, display, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_sc, txt_col, 2)

    return img

def draw_prediction_bar(img, tip_x, tip_y):
    t = theme()
    if not prediction_buttons:
        return img, None
    overlay = img.copy()
    cv2.rectangle(overlay, (55, 352), (WINDOW_W - 55, 418), t["panel"], cv2.FILLED)
    img = cv2.addWeighted(overlay, 0.85, img, 0.15, 0)

    hovered_pred = None
    for pb in prediction_buttons:
        x1, y1, x2, y2 = pb["rect"]
        hovering = x1 < tip_x < x2 and y1 < tip_y < y2
        if hovering:
            hovered_pred = pb
        col = t["key_hover"] if hovering else t["key_special"]
        draw_rounded_rect(img, (x1, y1), (x2, y2), col, radius=8)
        (tw, th), _ = cv2.getTextSize(pb["text"], cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.putText(img, pb["text"],
                    (x1 + (x2 - x1 - tw) // 2, y1 + (y2 - y1 + th) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, t["text"], 2)

    cv2.putText(img, "SUGGESTIONS:", (WINDOW_W - 260, 390),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, t["accent"], 1)
    return img, hovered_pred

def draw_voice_panel(img, tip_x=-1, tip_y=-1):
    t  = theme()
    hw = WINDOW_H
    overlay = img.copy()
    cv2.rectangle(overlay, (0, hw - 70), (WINDOW_W, hw), t["panel"], cv2.FILLED)
    img = cv2.addWeighted(overlay, 0.88, img, 0.12, 0)

    # Mic dot indicator
    cx, cy  = 30, hw - 35
    dot_col = (0, 220, 0) if voice_active else (80, 80, 80)
    cv2.circle(img, (cx, cy), 14, dot_col, cv2.FILLED)
    cv2.circle(img, (cx, cy), 14, (255, 255, 255), 2)
    if voice_active:
        pulse_r = int(22 + 6 * math.sin(time() * 4))
        cv2.circle(img, (cx, cy), pulse_r, (0, 255, 0), 1)

    # Status text
    cv2.putText(img, voice_status_msg, (55, hw - 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, t["accent"], 2)

    # ── START button ──
    s_x1, s_y1, s_x2, s_y2 = 860, hw - 58, 980, hw - 12
    s_hover = s_x1 < tip_x < s_x2 and s_y1 < tip_y < s_y2
    s_col   = (0, 200, 80) if not s_hover else (0, 255, 100)
    if voice_active:
        s_col = (0, 80, 30)   # dimmed when already active
    draw_rounded_rect(img, (s_x1, s_y1), (s_x2, s_y2), s_col, radius=10)
    cv2.putText(img, "START", (s_x1 + 18, s_y2 - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # ── STOP button ──
    p_x1, p_y1, p_x2, p_y2 = 995, hw - 58, 1115, hw - 12
    p_hover = p_x1 < tip_x < p_x2 and p_y1 < tip_y < p_y2
    p_col   = (0, 0, 200) if not p_hover else (0, 60, 255)
    if not voice_active:
        p_col = (30, 30, 80)  # dimmed when already stopped
    draw_rounded_rect(img, (p_x1, p_y1), (p_x2, p_y2), p_col, radius=10)
    cv2.putText(img, "STOP", (p_x1 + 22, p_y2 - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Button labels
    cv2.putText(img, "Voice:", (800, hw - 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140, 140, 140), 1)

    # Return button rects so main loop can handle clicks
    start_btn = (s_x1, s_y1, s_x2, s_y2)
    stop_btn  = (p_x1, p_y1, p_x2, p_y2)
    return img, start_btn, stop_btn

def draw_wpm(img):
    t       = theme()
    wpm     = stats["wpm_history"][-1] if stats["wpm_history"] else 0
    elapsed = int(time() - stats["session_start"])
    mm, ss  = divmod(elapsed, 60)
    cv2.putText(img, "WPM: " + str(wpm),
                (WINDOW_W - 180, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, t["wpm_color"], 2)
    cv2.putText(img, "Words: " + str(stats["total_words"]),
                (WINDOW_W - 180, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, t["text"], 1)
    cv2.putText(img, "Time: {:02d}:{:02d}".format(mm, ss),
                (WINDOW_W - 180, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, t["text"], 1)
    return img

def draw_stats_overlay(img):
    t = theme()
    overlay = img.copy()
    cv2.rectangle(overlay, (150, 80), (1130, 600), t["panel"], cv2.FILLED)
    img = cv2.addWeighted(overlay, 0.93, img, 0.07, 0)
    cv2.rectangle(img, (150, 80), (1130, 600), t["accent"], 2)

    wpm     = stats["wpm_history"][-1] if stats["wpm_history"] else 0
    elapsed = int(time() - stats["session_start"])

    lines = [
        ("SESSION STATISTICS",                                0.9,  t["accent"]),
        ("",                                                  0.4,  t["text"]),
        ("Total Characters : " + str(stats["total_chars"]),   0.7,  t["text"]),
        ("Total Words      : " + str(stats["total_words"]),   0.7,  t["text"]),
        ("Current WPM      : " + str(wpm),                    0.7,  t["wpm_color"]),
        ("Session Time     : " + str(elapsed) + "s",          0.7,  t["text"]),
        ("",                                                  0.4,  t["text"]),
        ("Active Theme     : " + theme_names[current_theme_idx], 0.65, t["accent"]),
        ("",                                                  0.4,  t["text"]),
        ("HOTWORDS:",                                         0.65, t["accent"]),
    ]
    for k, v in HOTWORDS.items():
        lines.append(("  " + k + "  ->  " + v, 0.52, t["text"]))
    lines += [
        ("",                                                  0.4,  t["text"]),
        ("VOICE COMMANDS:",                                   0.65, t["accent"]),
        ("  start/stop  |  clear all  |  backspace  |  new line", 0.52, t["text"]),
        ("  switch theme  |  emoji panel  |  show stats  |  undo/redo", 0.52, t["text"]),
        ("",                                                  0.4,  t["text"]),
        ("Press [S] to close",                                0.6,  (100, 100, 100)),
    ]

    y = 120
    for line, sc, col in lines:
        cv2.putText(img, line, (180, y), cv2.FONT_HERSHEY_SIMPLEX, sc, col,
                    2 if sc >= 0.7 else 1)
        y += int(sc * 58)
    return img

def draw_emoji_panel(img, tip_x, tip_y):
    t = theme()
    overlay = img.copy()
    cv2.rectangle(overlay, (60, 420), (1220, 650), t["panel"], cv2.FILLED)
    img = cv2.addWeighted(overlay, 0.92, img, 0.08, 0)
    cv2.rectangle(img, (60, 420), (1220, 650), t["accent"], 2)

    hovered_emoji = None
    cols   = 20
    cell_w = (1220 - 60) // cols
    cell_h = (650 - 420) // 2

    for idx, emoji in enumerate(EMOJIS):
        row = idx // cols
        col = idx  % cols
        ex  = 60 + col * cell_w + 4
        ey  = 420 + row * cell_h + cell_h // 2 + 8
        x1  = 60 + col * cell_w
        y1  = 420 + row * cell_h
        x2  = x1 + cell_w
        y2  = y1 + cell_h
        hovering = x1 < tip_x < x2 and y1 < tip_y < y2
        if hovering:
            hovered_emoji = (emoji, x1, y1, x2, y2)
            draw_rounded_rect(img, (x1 + 2, y1 + 2), (x2 - 2, y2 - 2), t["key_hover"], radius=6)
        cv2.putText(img, emoji, (ex, ey), cv2.FONT_HERSHEY_SIMPLEX, 0.55, t["text"], 1)

    return img, hovered_emoji

def draw_notification(img):
    global notification_msg
    if not notification_msg or time() > notification_time:
        notification_msg = ""
        return img
    t = theme()
    text_size, _ = cv2.getTextSize(notification_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    tw = text_size[0]
    nx = (WINDOW_W - tw) // 2 - 10
    overlay = img.copy()
    cv2.rectangle(overlay, (nx - 10, 630), (nx + tw + 20, 672), (20, 20, 20), cv2.FILLED)
    img = cv2.addWeighted(overlay, 0.8, img, 0.2, 0)
    cv2.putText(img, notification_msg, (nx, 660),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, t["accent"], 2)
    return img

def draw_fingertip(img, x, y, pinching):
    col = (0, 255, 100) if pinching else (255, 0, 200)
    cv2.circle(img, (x, y), 14, col, cv2.FILLED)
    cv2.circle(img, (x, y), 14, (255, 255, 255), 2)
    if pinching:
        cv2.circle(img, (x, y), 22, (0, 255, 100), 2)

def draw_help(img):
    cv2.putText(img, "[T]=Theme  [E]=Emoji  [S]=Stats  [Z]=Undo  [R]=Redo  [ESC]=Quit",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)

# ─────────────────────────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("  Virtual Keyboard ULTIMATE EDITION - Ready!")
print("  Say 'start' -> voice ON  |  'stop' -> voice OFF")
print("  Keyboard shortcuts: T=Theme  E=Emoji  S=Stats  Z=Undo")
print("  ESC -> quit")
print("=" * 60)

clicked      = False
caps_on      = False
last_click_t = 0.0
debug_frame  = 0  # remove after fix

# Create window before loop so it stays open
cv2.namedWindow("Virtual Keyboard ULTIMATE", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Virtual Keyboard ULTIMATE", WINDOW_W, WINDOW_H)

try:
    while True:
        success, img = cap.read()
        if not success:
            print("Camera read failed - exiting")
            break

        img    = cv2.flip(img, 1)
        rgb    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_img)

        # Flush voice queue
        while not voice_queue.empty():
            txt = voice_queue.get_nowait()
            type_text_safe(txt)
            for word in txt.split():
                predictor.learn(word)
            update_predictions()

        # Hand tracking
        x_tip    = -1
        y_tip    = -1
        pinching = False

        if result.hand_landmarks:
            lm        = result.hand_landmarks[0]
            h0, w0, _ = img.shape
            x_tip = int(lm[8].x  * w0)
            y_tip = int(lm[8].y  * h0)
            x_mid = int(lm[12].x * w0)
            y_mid = int(lm[12].y * h0)
            dist  = math.hypot(x_tip - x_mid, y_tip - y_mid)
            pinching = dist < PINCH_THRESHOLD

        # Theme background tint
        bg_frame = img.copy()
        cv2.rectangle(bg_frame, (0, 0), (WINDOW_W, WINDOW_H), theme()["bg"], cv2.FILLED)
        img = cv2.addWeighted(bg_frame, 0.35, img, 0.65, 0)

        now          = time()
        hover_button = None

        # Key hover + click
        if x_tip > 0:
            for btn in buttonList:
                bx, by = btn.first_pos
                bw, bh = btn.btn_size
                if bx < x_tip < bx + bw and by < y_tip < by + bh:
                    hover_button = btn
                    if pinching and not clicked and (now - last_click_t) > CLICK_COOLDOWN:
                        btn.press_anim = 1.0
                        last_click_t   = now
                        clicked        = True
                        if btn.text == "<-":
                            undo_last()
                        elif btn.text == "caps":
                            caps_on = not caps_on
                            show_notification("CAPS " + ("ON" if caps_on else "OFF"))
                        elif btn.text == " ":
                            keyboard_ctrl.press(' ')
                            keyboard_ctrl.release(' ')
                            typed_buffer.append(' ')
                            update_stats(' ')
                            buf = "".join(typed_buffer)
                            words = buf.rstrip().split()
                            if words:
                                predictor.learn(words[-1])
                        elif btn.text == "->":
                            keyboard_ctrl.press(Key.enter)
                            keyboard_ctrl.release(Key.enter)
                        else:
                            ch = btn.text.upper() if caps_on else btn.text
                            type_text_safe(ch)
                        update_predictions()

        if not pinching:
            clicked = False

        # Prediction bar click
        img, hovered_pred = draw_prediction_bar(img, x_tip, y_tip)
        if hovered_pred and pinching and not clicked and (now - last_click_t) > CLICK_COOLDOWN:
            buf     = "".join(typed_buffer)
            words   = buf.split()
            partial = words[-1] if words and not buf.endswith(" ") else ""
            for _ in partial:
                keyboard_ctrl.press('\b')
                keyboard_ctrl.release('\b')
                if typed_buffer:
                    typed_buffer.pop()
            type_text_safe(hovered_pred["text"] + " ")
            predictor.learn(hovered_pred["text"])
            update_predictions()
            last_click_t = now
            clicked      = True
            show_notification("Predicted: " + hovered_pred["text"])

        # Draw keyboard
        img = draw_keyboard(img, buttonList, hover=hover_button, caps_on=caps_on)

        # Emoji panel
        if show_emoji_panel:
            img, hovered_emoji = draw_emoji_panel(img, x_tip, y_tip)
            if hovered_emoji and pinching and not clicked and (now - last_click_t) > CLICK_COOLDOWN:
                type_text_safe(hovered_emoji[0])
                last_click_t = now
                clicked      = True
                show_notification("Typed: " + hovered_emoji[0])

        # Fingertip dot
        if x_tip > 0:
            draw_fingertip(img, x_tip, y_tip, pinching)

        # Panels and overlays
        img, start_btn, stop_btn = draw_voice_panel(img, x_tip, y_tip)

        # START button click
        if x_tip > 0:
            sx1,sy1,sx2,sy2 = start_btn
            px1,py1,px2,py2 = stop_btn
            if sx1 < x_tip < sx2 and sy1 < y_tip < sy2:
                if pinching and not clicked and (now - last_click_t) > CLICK_COOLDOWN:
                    voice_active     = True
                    voice_status_msg = "LISTENING... speak to type"
                    show_notification("Voice ON!")
                    last_click_t = now
                    clicked      = True
            if px1 < x_tip < px2 and py1 < y_tip < py2:
                if pinching and not clicked and (now - last_click_t) > CLICK_COOLDOWN:
                    voice_active     = False
                    voice_status_msg = "Voice PAUSED"
                    show_notification("Voice OFF!")
                    last_click_t = now
                    clicked      = True

        img = draw_wpm(img)
        img = draw_notification(img)
        draw_help(img)

        if caps_on:
            cv2.putText(img, "CAPS ON", (WINDOW_W - 200, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)

        cv2.putText(img, "Theme: " + theme_names[current_theme_idx],
                    (10, WINDOW_H - 78), cv2.FONT_HERSHEY_SIMPLEX, 0.5, theme()["accent"], 1)

        if show_stats:
            img = draw_stats_overlay(img)

        cv2.imshow("Virtual Keyboard ULTIMATE", img)

        key = cv2.waitKey(10) & 0xFF
        if key == 27:
            break
        elif key == ord('t') or key == ord('T'):
            current_theme_idx = (current_theme_idx + 1) % len(theme_names)
            show_notification("Theme: " + theme_names[current_theme_idx])
        elif key == ord('e') or key == ord('E'):
            show_emoji_panel = not show_emoji_panel
            show_notification("Emoji " + ("opened" if show_emoji_panel else "closed"))
        elif key == ord('s') or key == ord('S'):
            show_stats = not show_stats
        elif key == ord('z') or key == ord('Z'):
            undo_last()
            show_notification("Undo")
        elif key == ord('r') or key == ord('R'):
            redo_last()
            show_notification("Redo")

except Exception as e:
    print("\n" + "=" * 60)
    print("  CRASH DETECTED - full error below:")
    print("=" * 60)
    traceback.print_exc()
    print("=" * 60)

except SystemExit:
    pass

finally:
    print("Cleaning up...")
    try:
        cap.release()
    except:
        pass
    try:
        cv2.destroyAllWindows()
    except:
        pass
    print("Session Stats:")
    print("  Words     : " + str(stats["total_words"]))
    print("  Characters: " + str(stats["total_chars"]))
    print("  Time      : " + str(int(time() - stats["session_start"])) + "s")
    print("Press Enter to close...")
    input()