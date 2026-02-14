import cv2
import mediapipe as mp
from time import sleep
from pynput.keyboard import Controller, Key

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

keyboard = Controller()

# MediaPipe Tasks API
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Path to your .task model
model_path = r"D:\Data\samia\Projects\ICAT2\hand_landmarker.task"

# Hand Landmarker
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

# Keyboard layout with Enter key added
keys = [
    ["q","w","e","r","t","y","u","i","o","p","<-"],  # Backspace
    ["caps","a","s","d","f","g","h","j","k","l"],    # Caps Lock
    ["z","x","c","v"," ","b","n","m","->"],          # Space and Enter
]

buttonList = []

class Button:
    def __init__(self, first_pos, text, btn_size=[85,85]):
        self.first_pos = first_pos
        self.text = text
        # Special keys width
        if text == " ":
            self.btn_size = [284, 85]  # spacebar
        elif text in ["<-", "caps", "->"]:
            self.btn_size = [120, 85]  # special keys
        else:
            self.btn_size = btn_size

def draw_all_buttons(img, buttonList, hover=None):
    overlay = img.copy()
    cv2.rectangle(overlay, (50,0), (1200,350), (50,50,50), cv2.FILLED)
    alpha = 0.6
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    for button in buttonList:
        x, y = button.first_pos
        w, h = button.btn_size

        # Shadow effect
        cv2.rectangle(img, (x+5, y+5), (x+w+5, y+h+5), (30,30,30), cv2.FILLED)

        # Button color
        color = (203, 192, 255) if button.text in ["<-", " ", "caps", "->"] else (200,200,200)

        # Hover effect
        if hover == button:
            color = (255, 210, 240) if button.text in ["<-", " ", "caps", "->"] else (180, 120, 200)

        # Font scale: smaller for caps
        font_scale = 2.5 if button.text == "caps" else 4

        cv2.rectangle(img, button.first_pos, (x+w, y+h), color, cv2.FILLED)
        cv2.putText(img, button.text, (x+18, y+62),
                    cv2.FONT_HERSHEY_PLAIN, font_scale, (0,0,0), 3)

    return img

# Row offsets
row_offset = [0, 50, 0]

# Generate all buttons
for i, row in enumerate(keys):
    for j, key in enumerate(row):
        # Middle row: keep Caps at original position, shift the rest
        if i == 1 and key != "caps":
            pos = [100*j + 80 + row_offset[i], 100*i + 10]
        # Third row: center spacebar, move B,N,M,→ slightly right for fine alignment
        elif i == 2:
            if key == " ":
                pos = [int(1280/2 - 167), 100*i + 10]  # tiny move more to right
            elif key in ["b","n","m","->"]:
                block_start_x = int(1280/2 + 133)  # tiny move more to right
                key_index = ["b","n","m","->"].index(key)
                pos = [block_start_x + 100*key_index, 100*i + 10]
            else:
                pos = [100*j + 80, 100*i + 10]
        else:
            pos = [100*j + 80, 100*i + 10]

        buttonList.append(Button(pos, key))

clicked = False
caps_on = False

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img,1)
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    detection_result = detector.detect(mp_image)

    hover_button = None

    if detection_result.hand_landmarks:
        hand_landmarks = detection_result.hand_landmarks[0]

        h, w, c = img.shape
        x_tip = int(hand_landmarks[8].x * w)
        y_tip = int(hand_landmarks[8].y * h)
        x_mid = int(hand_landmarks[12].x * w)
        y_mid = int(hand_landmarks[12].y * h)
        distance = ((x_tip - x_mid)**2 + (y_tip - y_mid)**2)**0.5

        for button in buttonList:
            bx, by = button.first_pos
            bw, bh = button.btn_size

            if bx < x_tip < bx + bw and by < y_tip < by + bh:
                hover_button = button
                if distance < 40 and not clicked:
                    if button.text == "<-":
                        keyboard.press('\b')
                        keyboard.release('\b')
                    elif button.text == "caps":
                        caps_on = not caps_on
                    elif button.text == " ":
                        keyboard.press(' ')
                        keyboard.release(' ')
                    elif button.text == "->":
                        keyboard.press(Key.enter)
                        keyboard.release(Key.enter)
                    else:
                        key_to_type = button.text.upper() if caps_on else button.text
                        keyboard.press(key_to_type)
                        keyboard.release(key_to_type)
                    clicked = True
                    sleep(0.2)
        if distance >= 40:
            clicked = False

    img = draw_all_buttons(img, buttonList, hover=hover_button)

    cv2.imshow("Virtual Keyboard", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
