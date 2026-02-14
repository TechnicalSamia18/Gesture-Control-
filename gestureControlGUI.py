import cv2
import tkinter as tk
from PIL import Image, ImageTk

FPS_TARGET = 60


class HandGestureGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("HAND GESTURE CONTROL")
        self.root.configure(bg="#020611")
        self.root.state("zoomed")

        self.cap = cv2.VideoCapture(0)

        root.rowconfigure(0, weight=0)
        root.rowconfigure(1, weight=1)
        root.rowconfigure(2, weight=0)
        root.columnconfigure(0, weight=1)

        self.build_top_bar()
        self.build_main_content()
        self.build_bottom_bar()

        self.update_video()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ================= TOP BAR =================
    def build_top_bar(self):
        top = tk.Frame(self.root, bg="#08192c", height=45)
        top.grid(row=0, column=0, sticky="ew")
        top.grid_propagate(False)

        self.fps_label = tk.Label(
            top, text="FPS: 0",
            fg="#00f5ff", bg="#08192c",
            font=("Consolas", 10, "bold")
        )
        self.fps_label.pack(side="left", padx=15)

        tk.Label(
            top, text="HAND GESTURE CONTROL",
            fg="#00f5ff", bg="#08192c",
            font=("Consolas", 14, "bold")
        ).pack(side="left", expand=True)

        tk.Label(
            top, text="CAMERA ACTIVE | MODE: SCREEN CONTROL",
            fg="#8fd7ff", bg="#08192c",
            font=("Consolas", 10)
        ).pack(side="right", padx=15)

    # ================= MAIN CONTENT =================
    def build_main_content(self):

        main = tk.Frame(self.root, bg="#020611")
        main.grid(row=1, column=0, sticky="nsew", padx=15, pady=15)

        main.rowconfigure(0, weight=1)
        main.columnconfigure(0, weight=3)  # LEFT CAMERA (slightly bigger)
        main.columnconfigure(1, weight=2)  # RIGHT PANEL

        # ===== LEFT CAMERA (3D) =====
        cam_shadow = tk.Frame(main, bg="#000000")
        cam_shadow.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        cam_inner = tk.Frame(cam_shadow, bg="#0b1c2e", bd=2, relief="raised")
        cam_inner.pack(fill="both", expand=True, padx=3, pady=3)

        cam_inner.rowconfigure(0, weight=1)
        cam_inner.columnconfigure(0, weight=1)

        self.video_label = tk.Label(cam_inner, bg="black")
        self.video_label.grid(row=0, column=0, sticky="nsew")

        # ===== RIGHT PANEL (3D) =====
        right_shadow = tk.Frame(main, bg="#000000")
        right_shadow.grid(row=0, column=1, sticky="nsew")

        right = tk.Frame(right_shadow, bg="#0b1c2e", bd=2, relief="raised")
        right.pack(fill="both", expand=True, padx=3, pady=3)

        right.columnconfigure(0, weight=1)
        right.rowconfigure(2, weight=1)

        # Gesture Section
        gesture = tk.LabelFrame(
            right, text="GESTURE DETECTED",
            fg="#00f5ff", bg="#0b1c2e",
            font=("Consolas", 10, "bold"),
            bd=2, relief="groove"
        )
        gesture.grid(row=0, column=0, sticky="ew", padx=15, pady=15)

        tk.Label(
            gesture, text="[ ICON ]",
            bg="#020611", fg="#00f5ff",
            width=12, height=4
        ).pack(pady=8)

        self.gesture_label = tk.Label(
            gesture, text="Waiting...",
            fg="white", bg="#0b1c2e"
        )
        self.gesture_label.pack(pady=5)

        # Modules Section
        modules = tk.LabelFrame(
            right, text="MODULES",
            fg="#00f5ff", bg="#0b1c2e",
            font=("Consolas", 10, "bold"),
            bd=2, relief="groove"
        )
        modules.grid(row=1, column=0, sticky="ew", padx=15, pady=5)

        self.module_buttons = {}

        for name in ["Mouse Control", "Virtual Keyboard",
                     "Volume Control", "Media Control"]:

            btn = tk.Button(
                modules,
                text=name,
                bg="#162a40",
                fg="white",
                relief="raised",
                bd=3,
                font=("Consolas", 10, "bold"),
                cursor="hand2",
                command=lambda n=name: self.toggle_module(n)
            )
            btn.pack(fill="x", pady=6, padx=5)

            # Hover effect
            btn.bind("<Enter>", lambda e, b=btn: b.config(bg="#1f3d5a"))
            btn.bind("<Leave>", lambda e, b=btn: self.restore_button(b))

            self.module_buttons[name] = {"button": btn, "active": False}

        # Status
        status = tk.LabelFrame(
            right, text="STATUS",
            fg="#00f5ff", bg="#0b1c2e",
            font=("Consolas", 10, "bold"),
            bd=2, relief="groove"
        )
        status.grid(row=2, column=0, sticky="nsew", padx=15, pady=15)

        status.rowconfigure(1, weight=1)
        status.columnconfigure(0, weight=1)

        tk.Label(
            status, text="System Ready...",
            fg="white", bg="#0b1c2e"
        ).grid(row=0, column=0, sticky="w", padx=5, pady=5)

        self.thumb_label = tk.Label(status, bg="black")
        self.thumb_label.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

    # ================= BUTTON TOGGLE =================
    def toggle_module(self, name):
        data = self.module_buttons[name]
        data["active"] = not data["active"]

        if data["active"]:
            data["button"].config(bg="#00f5ff", fg="black")
        else:
            data["button"].config(bg="#162a40", fg="white")

    def restore_button(self, button):
        if button["bg"] != "#00f5ff":
            button.config(bg="#162a40")

    # ================= BOTTOM BAR =================
    def build_bottom_bar(self):
        bottom = tk.Frame(self.root, bg="#08192c", height=55)
        bottom.grid(row=2, column=0, sticky="ew")
        bottom.grid_propagate(False)

        tk.Label(bottom, text="☀", bg="#08192c",
                 fg="#ffd65c").pack(side="left", padx=15)

        tk.Scale(bottom, from_=0, to=100,
                 orient="horizontal",
                 bg="#08192c",
                 fg="white",
                 highlightthickness=0).pack(side="left", fill="x", expand=True)

        for text in ["MIC OFF", "APP SWITCH"]:
            btn = tk.Button(bottom, text=text,
                            bg="#162a40", fg="white",
                            relief="raised", bd=3, font=("Consolas", 10, "bold"),
                            cursor="hand2")
            btn.pack(side="left", padx=5)

            # Hover
            btn.bind("<Enter>", lambda e, b=btn: b.config(bg="#1f3d5a"))
            btn.bind("<Leave>", lambda e, b=btn: b.config(bg="#162a40"))

    # ================= VIDEO LOOP =================
    def update_video(self):
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(10, self.update_video)
            return

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        w = self.video_label.winfo_width()
        h = self.video_label.winfo_height()

        if w > 10 and h > 10:
            resized = cv2.resize(rgb, (w, h))
            img = ImageTk.PhotoImage(Image.fromarray(resized))
            self.video_label.configure(image=img)
            self.video_label.image = img

        self.root.after(int(1000 / FPS_TARGET), self.update_video)

    # ================= CLOSE =================
    def on_close(self):
        self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = HandGestureGUI(root)
    root.mainloop()
