# 🎯 Age & Gender Detector

A desktop application that detects **age and gender** from images or live webcam feed using deep learning.

Built with **InsightFace buffalo_l** + **OpenCV** + **Tkinter GUI**.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green)
![InsightFace](https://img.shields.io/badge/InsightFace-buffalo__l-orange)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)

---

## ✨ Features

- 📂 **Upload any image** — JPG, PNG, BMP, WebP
- 📷 **Live webcam** — real-time face detection at ~30 fps
- 🎯 **Accurate age & gender** — InsightFace buffalo_l model
- 🎚️ **Age Offset Slider** — manually correct model bias (−20 to +20)
- 📊 **Raw model output** shown alongside corrected value
- 💾 **Save results** — export annotated image
- 🎨 **Teal/Coral UI** — clean, attractive desktop interface

---

## 🖥️ Screenshots

> Upload an image → faces are detected automatically with age and gender labels.

---

## ⚙️ Requirements

- Python 3.8 or higher
- Internet connection (first run only — downloads ~300 MB model)

---

## 🚀 Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/age_gender_detector.git
cd age_gender_detector
```

### 2. Create a virtual environment (recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
python app.py
```

> **First launch:** InsightFace will automatically download the `buffalo_l` model (~300 MB) into `~/.insightface/`. This happens once and works offline after that.

---

## 📖 How to Use

| Step | Action |
|------|--------|
| 1 | Wait for **"InsightFace ready ✓"** in the top-right |
| 2 | Click **Upload Image** or **Start Webcam** |
| 3 | Faces are detected and labelled automatically |
| 4 | Use the **Age Offset** slider if the age looks wrong |
| 5 | Click **Save** to export the annotated image |

### Age Offset Slider
The buffalo_l model can overestimate age by 5–15 years for South Asian faces aged 18–35.  
Drag the slider **left (negative)** to correct downward, **right (positive)** to go higher.  
The image updates **live** as you drag.

---

## 🧠 How It Works

```
Input Image / Webcam Frame
        │
        ▼
  Preprocessing
  (CLAHE contrast + upscale small images)
        │
        ▼
  InsightFace buffalo_l
  (face detection + age + gender in one pass)
        │
        ▼
  Age Correction
  (bias curve + user offset slider)
        │
        ▼
  Annotated Output + Results Panel
```

### Models Used
| Model | Purpose | Size |
|-------|---------|------|
| `buffalo_l` (InsightFace) | Face detection + Age + Gender | ~300 MB (auto-downloaded) |

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `insightface` | ≥0.7.3 | Face analysis model |
| `onnxruntime` | ≥1.16.0 | Run ONNX models on CPU |
| `opencv-python` | ≥4.8.0 | Image processing |
| `Pillow` | ≥10.0.0 | Image display in Tkinter |
| `numpy` | ≥1.24.0 | Array operations |

---

## ⚠️ Known Limitations

- Age estimation is **approximate** (±3–8 years typical error)
- `buffalo_l` may **overestimate** for South Asian faces aged 18–35 — use the offset slider
- Requires **frontal, well-lit faces** for best results
- First run needs **internet** to download the model

---

## 🗂️ Project Structure

```
age_gender_detector/
├── app.py              # Main application
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── .gitignore          # Git ignore rules
```

---

## 🤝 Contributing

Pull requests are welcome! For major changes, open an issue first.

---

## 📄 License

MIT License — feel free to use, modify, and distribute.

---

## 🙏 Credits

- [InsightFace](https://github.com/deepinsight/insightface) — face analysis framework
- [OpenCV](https://opencv.org/) — computer vision library
- Age detection concept from [GeeksforGeeks](https://www.geeksforgeeks.org/age-detection-using-deep-learning-in-opencv/)
