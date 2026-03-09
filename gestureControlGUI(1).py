import cv2
import tkinter as tk
from PIL import Image, ImageTk
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
import screen_brightness_control as sbc
import numpy as np
import time

FPS_TARGET = 60

class HandGestureGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("HAND GESTURE CONTROL - STABILIZED")
        self.root.configure(bg="#020611")
        self.root.state("zoomed")

        # --- AI INITIALIZATION ---
        model_path = r"c:\Users\hp\Downloads\hand_landmarker.task"
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            running_mode=vision.RunningMode.VIDEO
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        self.last_ss_time = 0 
        self.cap = cv2.VideoCapture(0)

        root.rowconfigure(0, weight=0)
        root.rowconfigure(1, weight=1)
        root.rowconfigure(2, weight=0)
        root.columnconfigure(0, weight=1)

        self.module_buttons = {} 
        self.build_top_bar()
        self.build_main_content()
        self.build_bottom_bar()

        self.update_video()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def build_top_bar(self):
        top = tk.Frame(self.root, bg="#08192c", height=45)
        top.grid(row=0, column=0, sticky="ew")
        top.grid_propagate(False)
        self.status_bar_text = tk.Label(top, text="MODE: PHYSICAL RIGHT HAND ONLY", fg="#00f5ff", bg="#08192c", font=("Consolas", 10, "bold"))
        self.status_bar_text.pack(side="left", padx=15)
        tk.Label(top, text="HAND GESTURE SYSTEM", fg="#00f5ff", bg="#08192c", font=("Consolas", 14, "bold")).pack(side="left", expand=True)

    def build_main_content(self):
        main = tk.Frame(self.root, bg="#020611")
        main.grid(row=1, column=0, sticky="nsew", padx=15, pady=15)
        main.rowconfigure(0, weight=1)
        main.columnconfigure(0, weight=3)
        main.columnconfigure(1, weight=2)

        cam_shadow = tk.Frame(main, bg="#000000")
        cam_shadow.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        cam_inner = tk.Frame(cam_shadow, bg="#0b1c2e", bd=2, relief="raised")
        cam_inner.pack(fill="both", expand=True, padx=3, pady=3)
        self.video_label = tk.Label(cam_inner, bg="black")
        self.video_label.pack(fill="both", expand=True)

        right = tk.Frame(main, bg="#0b1c2e", bd=2, relief="raised")
        right.grid(row=0, column=1, sticky="nsew")

        gesture_frame = tk.LabelFrame(right, text="GESTURE DETECTED", fg="#00f5ff", bg="#0b1c2e", font=("Consolas", 10, "bold"), bd=2, relief="groove")
        gesture_frame.pack(fill="x", padx=15, pady=15)
        self.gesture_label = tk.Label(gesture_frame, text="Waiting...", fg="white", bg="#0b1c2e", font=("Consolas", 12))
        self.gesture_label.pack(pady=15)

        modules_frame = tk.LabelFrame(right, text="MODULES", fg="#00f5ff", bg="#0b1c2e", font=("Consolas", 10, "bold"), bd=2, relief="groove")
        modules_frame.pack(fill="x", padx=15, pady=5)
        for name in ["Mouse Control", "Virtual Keyboard", "Volume Control", "Media Control"]:
            btn = tk.Button(modules_frame, text=name, bg="#162a40", fg="white", relief="raised", bd=3, font=("Consolas", 10, "bold"), command=lambda n=name: self.toggle_module(n))
            btn.pack(fill="x", pady=6, padx=10)
            self.module_buttons[name] = {"button": btn, "active": False}

        self.ss_notify = tk.Label(right, text="System Ready", fg="#8fd7ff", bg="#0b1c2e", font=("Consolas", 10))
        self.ss_notify.pack(side="bottom", pady=20)

    def toggle_module(self, name):
        data = self.module_buttons[name]
        data["active"] = not data["active"]
        data["button"].config(bg="#00f5ff" if data["active"] else "#162a40", fg="black" if data["active"] else "white")

    def build_bottom_bar(self):
        bottom = tk.Frame(self.root, bg="#08192c", height=55)
        bottom.grid(row=2, column=0, sticky="ew")
        bottom.grid_propagate(False)
        tk.Label(bottom, text="☀ Brightness Control", bg="#08192c", fg="#ffd65c").pack(side="left", padx=15)
        self.bright_slider = tk.Scale(bottom, from_=0, to=100, orient="horizontal", bg="#08192c", fg="white", highlightthickness=0)
        self.bright_slider.pack(side="left", fill="x", expand=True, padx=20)

    def update_video(self):
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(10, self.update_video)
            return

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        timestamp = int(time.time() * 1000)
        result = self.detector.detect_for_video(mp_image, timestamp)
        
        if result.hand_landmarks and result.handedness:
            # --- THE MIRROR FIX ---
            # AI 'Left' = Your 'Physical Right' because the image is flipped.
            ai_label = result.handedness[0][0].category_name
            
            if ai_label == "Left": # This means it's your Physical Right hand
                lms = result.hand_landmarks[0]
                
                # Finger Detection Logic
                thumb_is_folded = lms[4].x > lms[3].x if lms[17].x > lms[5].x else lms[4].x < lms[3].x
                index_up  = lms[8].y < lms[6].y
                middle_up = lms[12].y < lms[10].y
                ring_up   = lms[16].y < lms[14].y
                pinky_up  = lms[20].y < lms[18].y

                dist = np.hypot(lms[8].x - lms[4].x, lms[8].y - lms[4].y) * 1000

                # --- BRIGHTNESS ---
                if index_up and not middle_up and not ring_up and not pinky_up:
                    val = int(np.interp(dist, [65, 240], [0, 100]))
                    sbc.set_brightness(val)
                    self.bright_slider.set(val)
                    self.gesture_label.config(text=f"BRIGHTNESS: {val}%")
                    cv2.line(frame, (int(lms[4].x*w), int(lms[4].y*h)), (int(lms[8].x*w), int(lms[8].y*h)), (0, 255, 0), 3)

                # --- SCREENSHOT ---
                elif index_up and middle_up and not ring_up and not pinky_up and thumb_is_folded:
                    self.gesture_label.config(text="RIGHT: PEACE SIGN")
                    if time.time() - self.last_ss_time > 3:
                        pyautogui.screenshot(f"screenshot_{int(time.time())}.png")
                        self.last_ss_time = time.time()
                        self.ss_notify.config(text="✓ SCREENSHOT SAVED!", fg="#00f5ff")
                        self.root.after(2000, lambda: self.ss_notify.config(text="System Ready", fg="#8fd7ff"))
                else:
                    self.gesture_label.config(text="Right Hand (No Command)")
            else:
                self.gesture_label.config(text="Left Hand (Ignored)")

        # Display Frame
        img_h, img_w = self.video_label.winfo_height(), self.video_label.winfo_width()
        if img_h > 10 and img_w > 10:
            resized = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (img_w, img_h))
            img_tk = ImageTk.PhotoImage(Image.fromarray(resized))
            self.video_label.configure(image=img_tk)
            self.video_label.image = img_tk

        self.root.after(1, self.update_video)

    def on_close(self):
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = HandGestureGUI(root)
    root.mainloop()