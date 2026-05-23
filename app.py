"""
╔══════════════════════════════════════════════════════════════╗
║       AGE & GENDER DETECTOR  —  InsightFace Edition v4       ║
║       + Manual age offset slider + raw age debug display     ║
║                                                              ║
║  SETUP (one-time):                                           ║
║    pip install insightface onnxruntime opencv-python pillow  ║
║                                                              ║
║  buffalo_l auto-downloads (~300 MB) on first run.            ║
╚══════════════════════════════════════════════════════════════╝

NOTE ON AGE ACCURACY:
  buffalo_l is a general-purpose model not tuned for South Asian
  faces. It tends to overestimate by 5–15 years for people aged
  18–35. Use the "Age Offset" slider (negative = younger) to
  manually correct. The raw model output is always shown so you
  can judge the error yourself.
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import os

# ── dependency check ──────────────────────────────────────────
DEPS_OK = True
MISSING = ""
INS_OK  = True

try:
    import cv2
    import numpy as np
    from PIL import Image, ImageTk
except ImportError as e:
    DEPS_OK = False
    MISSING = str(e)

try:
    import insightface
    from insightface.app import FaceAnalysis
except ImportError:
    INS_OK = False

# ── colour palette ────────────────────────────────────────────
BG_DARK    = "#1A2B2B"
BG_MID     = "#1F3535"
BG_LIGHT   = "#264040"
TEAL       = "#2EC4B6"
TEAL_LIGHT = "#4FDBD0"
CORAL      = "#FF6B6B"
CORAL_DARK = "#E84C4C"
GOLD       = "#FFD166"
CREAM      = "#FAFAF2"
MUTED      = "#8AADA8"
GREEN      = "#06D6A0"


# ── helpers ───────────────────────────────────────────────────
def parse_gender(raw) -> str:
    """InsightFace returns 1/0 int OR 'M'/'F' string."""
    if isinstance(raw, str):
        r = raw.strip().upper()
        if r in ("M", "MALE"):   return "Male"
        if r in ("F", "FEMALE"): return "Female"
        return raw
    return "Male" if int(raw) == 1 else "Female"

def gender_symbol(g: str) -> str:
    return "M" if g == "Male" else "F"

def gender_color(g: str) -> str:
    return TEAL if g == "Male" else CORAL

def age_emoji(age: int) -> str:
    if age <= 4:  return "👶"
    if age <= 12: return "🧒"
    if age <= 19: return "🧑"
    if age <= 35: return "👨"
    if age <= 50: return "🧔"
    if age <= 65: return "👴"
    return "🧓"

def apply_age_correction(raw: float, offset: int) -> int:
    """
    Two-stage correction:
    1. Model bias correction  — buffalo_l overestimates for young adults
    2. User offset            — slider lets user fine-tune
    """
    # Stage 1: bias correction
    if raw <= 12:
        corrected = raw
    elif raw <= 18:
        corrected = raw - 1
    elif raw <= 28:
        corrected = raw * 0.78        # biggest overestimate zone
    elif raw <= 40:
        corrected = raw * 0.83
    elif raw <= 55:
        corrected = raw * 0.88
    else:
        corrected = raw * 0.93

    # Stage 2: user offset
    final = int(round(corrected)) + offset
    return max(1, min(final, 100))

def preprocess(img_bgr):
    """Upscale small images + CLAHE contrast boost."""
    h, w = img_bgr.shape[:2]
    if min(h, w) < 480:
        s = 480 / min(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w*s), int(h*s)),
                             interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.merge([clahe.apply(l), a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ── detection core ────────────────────────────────────────────
class AgeDetector:
    def __init__(self):
        self.app    = None
        self.loaded = False
        self.error  = None

    def load_models(self):
        try:
            self.app = FaceAnalysis(
                name="buffalo_l",
                providers=["CPUExecutionProvider"]
            )
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            self.loaded = True
            return True
        except Exception as e:
            self.error = str(e)
            return False

    def process_image(self, img_bgr, conf_threshold=0.45, age_offset=0):
        enhanced  = preprocess(img_bgr)
        annotated = img_bgr.copy()

        try:
            faces = self.app.get(enhanced)
        except Exception:
            return annotated, []

        oh, ow = img_bgr.shape[:2]
        eh, ew = enhanced.shape[:2]
        sx, sy = ow / ew, oh / eh

        results = []
        for face in faces:
            det_score = float(face.det_score)
            if det_score < conf_threshold:
                continue

            raw_age       = float(face.age)
            corrected_age = apply_age_correction(raw_age, age_offset)
            gender        = parse_gender(face.gender)

            bx = face.bbox
            x1 = max(0,  int(bx[0] * sx))
            y1 = max(0,  int(bx[1] * sy))
            x2 = min(ow, int(bx[2] * sx))
            y2 = min(oh, int(bx[3] * sy))

            results.append({
                "box":       (x1, y1, x2, y2),
                "age":       corrected_age,
                "raw_age":   int(round(raw_age)),
                "gender":    gender,
                "face_conf": det_score,
            })
            self._draw_result(annotated, x1, y1, x2, y2,
                              corrected_age, raw_age, gender, det_score)

        return annotated, results

    def _draw_result(self, frame, x1, y1, x2, y2,
                     age, raw_age, gender, conf):
        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (46, 196, 182), 2)

        # Corner brackets
        for (cx, cy, dx, dy) in [
            (x1, y1,  1,  1), (x2, y1, -1,  1),
            (x1, y2,  1, -1), (x2, y2, -1, -1)
        ]:
            cv2.line(frame, (cx, cy), (cx+dx*18, cy),       (255,107,107), 3)
            cv2.line(frame, (cx, cy), (cx, cy+dy*18),       (255,107,107), 3)

        # Main label: corrected age + gender
        label  = f"Age {age}  {gender}"
        font   = cv2.FONT_HERSHEY_DUPLEX
        scale  = 0.6
        (tw, th), _ = cv2.getTextSize(label, font, scale, 1)
        lx = x1
        ly = y1 - 12 if y1 - 12 - th - 8 >= 0 else y2 + th + 16
        cv2.rectangle(frame, (lx-4, ly-th-8), (lx+tw+8, ly+4), (20,40,40), -1)
        cv2.rectangle(frame, (lx-4, ly-th-8), (lx+tw+8, ly+4), (46,196,182), 1)
        cv2.putText(frame, label, (lx+2, ly-2), font, scale,
                    (250,250,242), 1, cv2.LINE_AA)

        # Raw model value (smaller, below main label)
        raw_lbl = f"raw:{int(round(raw_age))}"
        cv2.putText(frame, raw_lbl, (lx+2, ly + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (139, 173, 168), 1, cv2.LINE_AA)

        # Confidence badge
        badge = f"{conf*100:.0f}%"
        (bw, bh), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        cv2.rectangle(frame, (x2-bw-8, y2), (x2, y2+bh+6), (20,40,40), -1)
        cv2.putText(frame, badge, (x2-bw-4, y2+bh+1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (46,196,182), 1, cv2.LINE_AA)


# ── GUI ───────────────────────────────────────────────────────
class AgeDetectionApp:
    def __init__(self, root):
        self.root          = root
        self.detector      = AgeDetector()
        self.result_image  = None
        self.webcam_active = False
        self.cap           = None
        self._tk_img       = None
        self._last_raw     = []   # stores raw image for re-processing

        self._build_ui()

        if not DEPS_OK:
            self._dep_error("opencv-python / pillow / numpy", MISSING)
        elif not INS_OK:
            self._dep_error("insightface + onnxruntime",
                            "pip install insightface onnxruntime")
        else:
            self._load_models_async()

    # ── build UI ──────────────────────────────
    def _build_ui(self):
        self.root.title("Age & Gender Detector  ·  InsightFace v4")
        self.root.geometry("1100x740")
        self.root.minsize(900, 620)
        self.root.configure(bg=BG_DARK)

        self.f_title = ("Trebuchet MS", 22, "bold")
        self.f_sub   = ("Trebuchet MS", 10)
        self.f_btn   = ("Trebuchet MS", 10, "bold")
        self.f_card  = ("Trebuchet MS", 11, "bold")
        self.f_small = ("Trebuchet MS", 9)

        self._header()
        self._body()
        self._statusbar()

    def _header(self):
        hdr = tk.Frame(self.root, bg=BG_MID, height=70)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        L = tk.Frame(hdr, bg=BG_MID)
        L.pack(side="left", padx=24, pady=10)
        tk.Label(L, text="◈  AGE & GENDER DETECTOR",
                 font=self.f_title, bg=BG_MID, fg=TEAL).pack(anchor="w")
        tk.Label(L, text="InsightFace buffalo_l  ·  Accurate · Real-time · Auto-download",
                 font=self.f_sub, bg=BG_MID, fg=MUTED).pack(anchor="w")

        R = tk.Frame(hdr, bg=BG_MID)
        R.pack(side="right", padx=24)
        self.dot     = tk.Label(R, text="●", font=("Arial",16), bg=BG_MID, fg=GOLD)
        self.dot_lbl = tk.Label(R, text="Loading…", font=self.f_small, bg=BG_MID, fg=MUTED)
        self.dot.pack(side="left")
        self.dot_lbl.pack(side="left", padx=6)

    def _body(self):
        body = tk.Frame(self.root, bg=BG_DARK)
        body.pack(fill="both", expand=True, padx=16, pady=10)

        lp = tk.Frame(body, bg=BG_DARK, width=275)
        lp.pack(side="left", fill="y")
        lp.pack_propagate(False)
        self._controls(lp)
        self._results_panel(lp)

        rp = tk.Frame(body, bg=BG_MID)
        rp.pack(side="left", fill="both", expand=True, padx=(12,0))
        self._image_panel(rp)

    def _controls(self, parent):
        card = tk.Frame(parent, bg=BG_MID, padx=14, pady=14)
        card.pack(fill="x", pady=(0,10))

        tk.Label(card, text="INPUT SOURCE", font=self.f_card,
                 bg=BG_MID, fg=TEAL).pack(anchor="w", pady=(0,10))

        self.btn_upload = self._btn(card, "📂  Upload Image",
                                    TEAL, BG_DARK, self._upload_image)
        self.btn_upload.pack(fill="x", pady=(0,8))

        self.btn_cam = self._btn(card, "📷  Start Webcam",
                                 CORAL, BG_DARK, self._toggle_webcam)
        self.btn_cam.pack(fill="x", pady=(0,12))

        # ── Detection confidence ──────────────
        sep = tk.Frame(card, bg=BG_LIGHT, height=1)
        sep.pack(fill="x", pady=(0,10))

        tk.Label(card, text="Min Detection Score",
                 font=self.f_small, bg=BG_MID, fg=MUTED).pack(anchor="w", pady=(0,2))
        self.conf_var = tk.DoubleVar(value=0.45)
        tk.Scale(card, from_=0.2, to=0.95, resolution=0.05,
                 orient="horizontal", variable=self.conf_var,
                 bg=BG_MID, fg=CREAM, troughcolor=BG_LIGHT,
                 activebackground=TEAL, highlightthickness=0,
                 sliderrelief="flat", bd=0,
                 font=self.f_small).pack(fill="x")

        # ── Age offset ───────────────────────
        sep2 = tk.Frame(card, bg=BG_LIGHT, height=1)
        sep2.pack(fill="x", pady=(10,10))

        offset_hdr = tk.Frame(card, bg=BG_MID)
        offset_hdr.pack(fill="x", pady=(0,2))
        tk.Label(offset_hdr, text="Age Offset (fix model error)",
                 font=self.f_small, bg=BG_MID, fg=GOLD).pack(side="left")
        self.offset_lbl = tk.Label(offset_hdr, text="0",
                                   font=self.f_card, bg=BG_MID, fg=GOLD)
        self.offset_lbl.pack(side="right")

        self.offset_var = tk.IntVar(value=0)
        offset_scale = tk.Scale(
            card, from_=-20, to=20, resolution=1,
            orient="horizontal", variable=self.offset_var,
            bg=BG_MID, fg=CREAM, troughcolor=BG_LIGHT,
            activebackground=GOLD, highlightthickness=0,
            sliderrelief="flat", bd=0, font=self.f_small,
            command=self._on_offset_change)
        offset_scale.pack(fill="x")

        tk.Label(card,
                 text="← younger   |   older →\n"
                      "Drag left if age looks too high",
                 font=("Trebuchet MS", 8), bg=BG_MID, fg=MUTED,
                 justify="center").pack(pady=(2,0))

        # ── Info box ─────────────────────────
        info = tk.Frame(card, bg=BG_LIGHT, padx=10, pady=8)
        info.pack(fill="x", pady=(12,0))
        tk.Label(info,
                 text="ℹ  Image label shows:\n"
                      "  Age XX  (corrected)\n"
                      "  raw:YY  (model output)\n"
                      "Use offset to fine-tune.",
                 font=("Trebuchet MS", 8), bg=BG_LIGHT, fg=MUTED,
                 justify="left").pack(anchor="w")

    def _on_offset_change(self, val):
        v = int(float(val))
        sign = "+" if v >= 0 else ""
        self.offset_lbl.config(text=f"{sign}{v}")
        # Re-process last image live if available
        if self._last_raw_img is not None and not self.webcam_active:
            self._reprocess()

    def _results_panel(self, parent):
        card = tk.Frame(parent, bg=BG_MID, padx=14, pady=14)
        card.pack(fill="both", expand=True)

        tk.Label(card, text="RESULTS", font=self.f_card,
                 bg=BG_MID, fg=TEAL).pack(anchor="w", pady=(0,8))

        wrap = tk.Frame(card, bg=BG_MID)
        wrap.pack(fill="both", expand=True)

        self.res_canvas = tk.Canvas(wrap, bg=BG_MID,
                                    highlightthickness=0, bd=0)
        vsb = tk.Scrollbar(wrap, orient="vertical",
                           command=self.res_canvas.yview)
        self.res_canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self.res_canvas.pack(side="left", fill="both", expand=True)

        self.res_frame = tk.Frame(self.res_canvas, bg=BG_MID)
        self._res_win  = self.res_canvas.create_window(
            (0,0), window=self.res_frame, anchor="nw")
        self.res_frame.bind("<Configure>",
            lambda e: self.res_canvas.configure(
                scrollregion=self.res_canvas.bbox("all")))
        self.res_canvas.bind("<Configure>",
            lambda e: self.res_canvas.itemconfig(
                self._res_win, width=e.width))

        self._placeholder_result()

    def _placeholder_result(self):
        for w in self.res_frame.winfo_children():
            w.destroy()
        tk.Label(self.res_frame,
                 text="No faces detected yet.\nUpload an image or\nstart your webcam.",
                 font=self.f_small, bg=BG_MID, fg=MUTED,
                 justify="center", pady=20).pack()

    def _image_panel(self, parent):
        tb = tk.Frame(parent, bg=BG_MID, padx=12, pady=8)
        tb.pack(fill="x")
        tk.Label(tb, text="PREVIEW", font=self.f_card,
                 bg=BG_MID, fg=TEAL).pack(side="left")
        self.btn_save = self._btn(tb, "💾 Save", GOLD, BG_DARK,
                                  self._save, small=True)
        self.btn_save.pack(side="right")
        self.btn_save.config(state="disabled")
        self._btn(tb, "✕ Clear", BG_LIGHT, MUTED,
                  self._clear, small=True).pack(side="right", padx=(0,6))

        self.canvas = tk.Canvas(parent, bg=BG_DARK,
                                highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True, padx=12, pady=(0,12))
        self.canvas.bind("<Configure>", self._canvas_resize)
        self._last_raw_img = None
        self._placeholder_canvas()

    def _statusbar(self):
        bar = tk.Frame(self.root, bg=BG_LIGHT, height=28)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)
        self.status = tk.Label(bar, text="  Initialising…",
                               font=self.f_small, bg=BG_LIGHT, fg=MUTED,
                               anchor="w", padx=16)
        self.status.pack(side="left", fill="y")
        tk.Label(bar, text="InsightFace buffalo_l  ·  ONNX Runtime",
                 font=self.f_small, bg=BG_LIGHT, fg=MUTED, padx=16
                 ).pack(side="right", fill="y")

    # ── widget factory ────────────────────────
    def _btn(self, parent, text, bg, fg, cmd, small=False):
        f   = self.f_small if small else self.f_btn
        pad = (8,4) if small else (12,8)
        b   = tk.Button(parent, text=text, font=f, bg=bg, fg=fg,
                        relief="flat", cursor="hand2", bd=0,
                        padx=pad[0], pady=pad[1], command=cmd,
                        activebackground=TEAL_LIGHT, activeforeground=BG_DARK)
        hover = {TEAL:TEAL_LIGHT, CORAL:CORAL_DARK,
                 GOLD:"#FFE08A",  BG_LIGHT:"#2E4D4D"}.get(bg, bg)
        b.bind("<Enter>", lambda e: b.config(bg=hover))
        b.bind("<Leave>", lambda e: b.config(bg=bg))
        return b

    # ── canvas ────────────────────────────────
    def _placeholder_canvas(self):
        self.canvas.delete("all")
        w  = self.canvas.winfo_width()  or 600
        h  = self.canvas.winfo_height() or 460
        cx, cy = w//2, h//2
        self.canvas.create_rectangle(
            cx-80, cy-60, cx+80, cy+60,
            outline=BG_LIGHT, width=2, dash=(6,4))
        self.canvas.create_text(cx, cy-15, text="🖼",
                                font=("Arial",32), fill=BG_LIGHT)
        self.canvas.create_text(cx, cy+30,
                                text="Upload an image to begin",
                                font=self.f_sub, fill=MUTED)

    def _canvas_resize(self, e=None):
        if self.result_image is not None:
            self._show_cv(self.result_image)
        elif not self.webcam_active:
            self._placeholder_canvas()

    def _show_cv(self, cv_img):
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        cw  = self.canvas.winfo_width()  or 600
        ch  = self.canvas.winfo_height() or 460
        iw, ih = pil.size
        s   = min(cw/iw, ch/ih, 1.0)
        pil = pil.resize((int(iw*s), int(ih*s)), Image.LANCZOS)
        self._tk_img = ImageTk.PhotoImage(pil)
        self.canvas.delete("all")
        self.canvas.create_image(cw//2, ch//2, image=self._tk_img)

    # ── model loading ─────────────────────────
    def _load_models_async(self):
        self._set_status("Downloading / loading InsightFace buffalo_l…")
        threading.Thread(target=lambda:
            self.root.after(0, self._on_loaded,
                            self.detector.load_models()),
            daemon=True).start()

    def _on_loaded(self, ok):
        if ok:
            self.dot.config(fg=TEAL)
            self.dot_lbl.config(text="InsightFace ready ✓", fg=TEAL)
            self._set_status("✓  Model ready — upload an image or start the webcam.")
        else:
            self.dot.config(fg=CORAL)
            self.dot_lbl.config(text="Load failed ✗", fg=CORAL)
            self._set_status(f"⚠  {self.detector.error}")

    # ── upload ────────────────────────────────
    def _upload_image(self):
        if not self.detector.loaded:
            messagebox.showwarning("Not Ready","Model is still loading.")
            return
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Images","*.jpg *.jpeg *.png *.bmp *.webp"),
                       ("All","*.*")])
        if not path:
            return
        self._stop_webcam()
        self._set_status("Analysing…")
        self.root.update()

        def _run():
            img = cv2.imread(path)
            if img is None:
                self.root.after(0, lambda:
                    self._set_status("❌ Cannot read image."))
                return
            self._last_raw_img = img
            ann, res = self.detector.process_image(
                img, self.conf_var.get(), self.offset_var.get())
            self.root.after(0, self._on_result, ann, res)

        threading.Thread(target=_run, daemon=True).start()

    def _reprocess(self):
        """Re-run detection on cached image (called when offset changes)."""
        if self._last_raw_img is None:
            return
        def _run():
            ann, res = self.detector.process_image(
                self._last_raw_img,
                self.conf_var.get(),
                self.offset_var.get())
            self.root.after(0, self._on_result, ann, res)
        threading.Thread(target=_run, daemon=True).start()

    def _on_result(self, ann, results):
        self.result_image = ann
        self._show_cv(ann)
        self._update_results(results)
        self.btn_save.config(state="normal")
        n = len(results)
        if n == 0:
            self._set_status(
                "No faces detected — try lowering Min Detection Score.")
        else:
            parts = [f"Age {r['age']} (raw {r['raw_age']}) {r['gender']}"
                     for r in results]
            self._set_status(f"✓  {n} face{'s' if n>1 else ''}: {', '.join(parts)}")

    # ── results panel ─────────────────────────
    def _update_results(self, results):
        for w in self.res_frame.winfo_children():
            w.destroy()
        if not results:
            tk.Label(self.res_frame, text="No faces found.",
                     font=self.f_small, bg=BG_MID, fg=MUTED).pack(pady=10)
            return
        for i, r in enumerate(results, 1):
            self._face_card(r, i)

    def _face_card(self, r, i):
        card = tk.Frame(self.res_frame, bg=BG_LIGHT, padx=10, pady=8)
        card.pack(fill="x", pady=(0,6))

        # header
        hdr = tk.Frame(card, bg=BG_LIGHT)
        hdr.pack(fill="x", pady=(0,6))
        tk.Label(hdr, text=f"Face #{i}", font=self.f_card,
                 bg=BG_LIGHT, fg=CREAM).pack(side="left")
        tk.Label(hdr, text=f"{r['face_conf']*100:.0f}% conf",
                 font=self.f_small, bg=BG_LIGHT, fg=TEAL).pack(side="right")

        # age
        age = r["age"]
        raw = r["raw_age"]
        self._row(card, f"{age_emoji(age)}  Age", str(age), GOLD)

        # raw age note
        raw_f = tk.Frame(card, bg=BG_LIGHT)
        raw_f.pack(fill="x", pady=(0,2))
        tk.Label(raw_f, text="   model raw:",
                 font=("Trebuchet MS",8), bg=BG_LIGHT, fg=MUTED).pack(side="left")
        tk.Label(raw_f, text=str(raw),
                 font=("Trebuchet MS",8,"bold"), bg=BG_LIGHT, fg=MUTED).pack(side="left")
        diff = age - raw
        diff_txt = f"({'+' if diff>=0 else ''}{diff} offset)"
        tk.Label(raw_f, text=diff_txt,
                 font=("Trebuchet MS",8), bg=BG_LIGHT,
                 fg=GREEN if diff < 0 else CORAL).pack(side="left", padx=4)

        # age bar
        bf = tk.Frame(card, bg=BG_LIGHT)
        bf.pack(fill="x", pady=(2,8))
        bar_w = int(min(age / 90 * 220, 220))
        tk.Frame(bf, bg=GOLD,    height=5, width=bar_w).pack(side="left")
        tk.Frame(bf, bg=BG_DARK, height=5,
                 width=max(0,220-bar_w)).pack(side="left")

        # gender
        g    = r["gender"]
        gcol = gender_color(g)
        self._row(card, "Gender", g, gcol)

        badge = tk.Label(card, text=f" {gender_symbol(g)}  {g} ",
                         font=self.f_small, bg=gcol, fg=BG_DARK,
                         padx=6, pady=2)
        badge.pack(anchor="w", pady=(4,0))

    def _row(self, parent, label, value, col):
        row = tk.Frame(parent, bg=BG_LIGHT)
        row.pack(fill="x", pady=2)
        tk.Label(row, text=label, font=self.f_small,
                 bg=BG_LIGHT, fg=MUTED, width=13, anchor="w").pack(side="left")
        tk.Label(row, text=value, font=self.f_card,
                 bg=BG_LIGHT, fg=col).pack(side="left", padx=6)

    # ── webcam ────────────────────────────────
    def _toggle_webcam(self):
        if self.webcam_active: self._stop_webcam()
        else:                  self._start_webcam()

    def _start_webcam(self):
        if not self.detector.loaded:
            messagebox.showwarning("Not Ready","Model is still loading.")
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Webcam","Could not open webcam.")
            return
        self.webcam_active = True
        self.btn_cam.config(text="⏹  Stop Webcam", bg=CORAL_DARK)
        self._set_status("Webcam active…")
        self._cam_loop()

    def _stop_webcam(self):
        self.webcam_active = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.btn_cam.config(text="📷  Start Webcam", bg=CORAL)
        if self.result_image is None:
            self._placeholder_canvas()

    def _cam_loop(self):
        if not self.webcam_active:
            return
        ret, frame = self.cap.read()
        if ret:
            ann, res = self.detector.process_image(
                frame, self.conf_var.get(), self.offset_var.get())
            self.result_image = ann
            self._show_cv(ann)
            self._update_results(res)
        self.root.after(33, self._cam_loop)

    # ── save / clear ──────────────────────────
    def _save(self):
        if self.result_image is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG","*.png"),("JPEG","*.jpg")],
            initialfile="age_gender_result")
        if path:
            cv2.imwrite(path, self.result_image)
            self._set_status(f"✓  Saved → {os.path.basename(path)}")

    def _clear(self):
        self._stop_webcam()
        self.result_image  = None
        self._last_raw_img = None
        self._placeholder_canvas()
        self.btn_save.config(state="disabled")
        self._placeholder_result()
        self._set_status("Cleared.")

    def _set_status(self, msg):
        self.status.config(text=f"  {msg}")

    def _dep_error(self, pkg, detail):
        messagebox.showerror("Missing Package",
            f"Required:\n{pkg}\n\n{detail}\n\nInstall then restart.")
        self.dot.config(fg=CORAL)
        self.dot_lbl.config(text="Install deps first", fg=CORAL)

    def on_close(self):
        self._stop_webcam()
        self.root.destroy()


# ── entry point ───────────────────────────────────────────────
def main():
    root = tk.Tk()
    app  = AgeDetectionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.update_idletasks()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    w, h   = 1100, 740
    root.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")
    root.mainloop()

if __name__ == "__main__":
    main()
