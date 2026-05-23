# 🎯 Age & Gender Detector

A web app that detects **age and gender** from uploaded photos using deep learning.

Built with **InsightFace buffalo_l** + **OpenCV** + **Streamlit**.

 **[Live Demo](https://agegenderdetector-tcczawbgwkzbgnb5wfhyad.streamlit.app)**
 
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red)
![InsightFace](https://img.shields.io/badge/InsightFace-buffalo__l-orange)

---

## ✨ Features

- 📂 Upload any image (JPG, PNG, BMP, WebP)
- 🎯 Accurate age & gender — InsightFace buffalo_l model
- 🎚️ Age Offset Slider — manually correct model bias
- 📊 Raw model output shown alongside corrected value
- 💾 Download annotated result image
- 🎨 Teal/Coral dark theme UI

---

## 🚀 Run Locally

```bash
git clone https://github.com/varshini3406/age_gender_detector.git
cd age_gender_detector
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

## ☁️ Deploy on Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to https://streamlit.io/cloud
3. Click **"New app"**
4. Select your repo → set **Main file: `streamlit_app.py`**
5. Click **Deploy** — live in ~2 minutes!

---

## 🧠 How It Works

```
Upload Image
     │
     ▼
Preprocessing (CLAHE contrast + upscale)
     │
     ▼
InsightFace buffalo_l
(face detection + age + gender)
     │
     ▼
Age Correction (bias curve + user offset)
     │
     ▼
Annotated Result + Download
```

---

## ⚠️ Notes

- Age accuracy: ±5–10 years typical
- buffalo_l may overestimate for South Asian faces aged 18–35 — use the Age Offset slider
- First run downloads ~300 MB model automatically

---

## 📄 License

MIT License
