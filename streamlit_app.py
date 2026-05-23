"""
Age & Gender Detector — Streamlit Web App
Deploy free on Streamlit Cloud: https://streamlit.io/cloud
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# ── page config (MUST be first Streamlit call) ────────────────
st.set_page_config(
    page_title="Age & Gender Detector",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
  /* Main background */
  .stApp { background-color: #1A2B2B; color: #FAFAF2; }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background-color: #1F3535;
    border-right: 1px solid #264040;
  }
  [data-testid="stSidebar"] * { color: #FAFAF2 !important; }

  /* Headings */
  h1 { color: #2EC4B6 !important; font-family: 'Trebuchet MS', sans-serif; }
  h2, h3 { color: #2EC4B6 !important; }

  /* Metric cards */
  [data-testid="stMetric"] {
    background-color: #264040;
    border: 1px solid #2EC4B6;
    border-radius: 10px;
    padding: 12px 16px;
  }
  [data-testid="stMetricLabel"] { color: #8AADA8 !important; font-size:13px; }
  [data-testid="stMetricValue"] { color: #FFD166 !important; font-size:28px; font-weight:bold; }

  /* File uploader */
  [data-testid="stFileUploader"] {
    background-color: #1F3535;
    border: 2px dashed #2EC4B6;
    border-radius: 12px;
    padding: 20px;
  }

  /* Buttons */
  .stButton > button {
    background-color: #2EC4B6;
    color: #1A2B2B;
    font-weight: bold;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    font-size: 15px;
    width: 100%;
  }
  .stButton > button:hover { background-color: #4FDBD0; }

  /* Slider */
  .stSlider > div > div > div { background-color: #2EC4B6 !important; }

  /* Success / info boxes */
  .stSuccess { background-color: #1F3535 !important; border-left: 4px solid #06D6A0; }
  .stInfo    { background-color: #1F3535 !important; border-left: 4px solid #2EC4B6; }
  .stWarning { background-color: #1F3535 !important; border-left: 4px solid #FFD166; }

  /* Face result card */
  .face-card {
    background-color: #264040;
    border: 1px solid #2EC4B6;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
  }
  .face-card h4 { color: #FAFAF2; margin:0 0 8px 0; }
  .gender-badge-male   {
    display:inline-block; background:#2EC4B6; color:#1A2B2B;
    border-radius:20px; padding:3px 14px; font-weight:bold; font-size:14px;
  }
  .gender-badge-female {
    display:inline-block; background:#FF6B6B; color:#1A2B2B;
    border-radius:20px; padding:3px 14px; font-weight:bold; font-size:14px;
  }
  .age-value  { color:#FFD166; font-size:32px; font-weight:bold; }
  .raw-value  { color:#8AADA8; font-size:12px; }
  .conf-value { color:#2EC4B6; font-size:12px; }

  /* Divider */
  hr { border-color: #264040; }

  /* Hide Streamlit default elements */
  #MainMenu { visibility: hidden; }
  footer    { visibility: hidden; }
  header    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── helpers (same logic as desktop app) ───────────────────────
def parse_gender(raw) -> str:
    if isinstance(raw, str):
        r = raw.strip().upper()
        if r in ("M", "MALE"):   return "Male"
        if r in ("F", "FEMALE"): return "Female"
        return raw
    return "Male" if int(raw) == 1 else "Female"

def age_emoji(age: int) -> str:
    if age <= 4:  return "👶"
    if age <= 12: return "🧒"
    if age <= 19: return "🧑"
    if age <= 35: return "👨"
    if age <= 50: return "🧔"
    if age <= 65: return "👴"
    return "🧓"

def apply_age_correction(raw: float, offset: int) -> int:
    if raw <= 12:   corrected = raw
    elif raw <= 18: corrected = raw - 1
    elif raw <= 28: corrected = raw * 0.78
    elif raw <= 40: corrected = raw * 0.83
    elif raw <= 55: corrected = raw * 0.88
    else:           corrected = raw * 0.93
    return max(1, min(int(round(corrected)) + offset, 100))

def preprocess(img_bgr):
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

def draw_on_image(frame, x1, y1, x2, y2, age, raw_age, gender, conf):
    cv2.rectangle(frame, (x1, y1), (x2, y2), (46, 196, 182), 2)
    for (cx, cy, dx, dy) in [
        (x1,y1, 1, 1),(x2,y1,-1, 1),
        (x1,y2, 1,-1),(x2,y2,-1,-1)
    ]:
        cv2.line(frame,(cx,cy),(cx+dx*18,cy),       (255,107,107),3)
        cv2.line(frame,(cx,cy),(cx,cy+dy*18),       (255,107,107),3)

    label = f"Age {age}  {gender}"
    font  = cv2.FONT_HERSHEY_DUPLEX
    (tw,th),_ = cv2.getTextSize(label, font, 0.6, 1)
    lx = x1
    ly = y1-12 if y1-12-th-8 >= 0 else y2+th+16
    cv2.rectangle(frame,(lx-4,ly-th-8),(lx+tw+8,ly+4),(20,40,40),-1)
    cv2.rectangle(frame,(lx-4,ly-th-8),(lx+tw+8,ly+4),(46,196,182),1)
    cv2.putText(frame,label,(lx+2,ly-2),font,0.6,(250,250,242),1,cv2.LINE_AA)
    cv2.putText(frame,f"raw:{int(round(raw_age))}",(lx+2,ly+14),
                cv2.FONT_HERSHEY_SIMPLEX,0.38,(139,173,168),1,cv2.LINE_AA)
    badge = f"{conf*100:.0f}%"
    (bw,bh),_ = cv2.getTextSize(badge,cv2.FONT_HERSHEY_SIMPLEX,0.38,1)
    cv2.rectangle(frame,(x2-bw-8,y2),(x2,y2+bh+6),(20,40,40),-1)
    cv2.putText(frame,badge,(x2-bw-4,y2+bh+1),
                cv2.FONT_HERSHEY_SIMPLEX,0.38,(46,196,182),1,cv2.LINE_AA)

# ── load model (cached so it loads only once) ─────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="buffalo_l",
                          providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640))
        return app, None
    except Exception as e:
        return None, str(e)

# ── sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    conf_threshold = st.slider(
        "Min Detection Score",
        min_value=0.2, max_value=0.95,
        value=0.45, step=0.05,
        help="Lower = detect more faces. Higher = only confident detections."
    )

    age_offset = st.slider(
        "Age Offset",
        min_value=-20, max_value=20,
        value=0, step=1,
        help="Drag left if age looks too high. Drag right if too low."
    )
    if age_offset < 0:
        st.caption(f"⬅ Making ages {abs(age_offset)} years younger")
    elif age_offset > 0:
        st.caption(f"➡ Making ages {age_offset} years older")
    else:
        st.caption("No offset applied")

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
- Model: **InsightFace buffalo_l**
- Age accuracy: **±5–10 years**
- Works best with **frontal, well-lit** faces
- buffalo_l may overestimate for **South Asian faces aged 18–35** — use the Age Offset slider to correct
    """)
    st.markdown("---")
    st.markdown("Built with ❤️ using InsightFace + OpenCV + Streamlit")

# ── main page ─────────────────────────────────────────────────
st.markdown("# 🎯 Age & Gender Detector")
st.markdown("Upload a photo to detect **age and gender** using deep learning.")
st.markdown("---")

# Load model
with st.spinner("⏳ Loading InsightFace model (downloads ~300 MB on first run)…"):
    face_app, load_error = load_model()

if load_error:
    st.error(f"❌ Model failed to load: {load_error}")
    st.info("Run: `pip install insightface onnxruntime`")
    st.stop()
else:
    st.success("✅ Model ready!")

# File uploader
uploaded = st.file_uploader(
    "📂 Upload an image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    help="Supports JPG, PNG, BMP, WebP"
)

if uploaded is not None:
    # Read image
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        st.error("❌ Could not read image. Try a different file.")
        st.stop()

    # Process
    with st.spinner("🔍 Detecting faces…"):
        enhanced  = preprocess(img_bgr)
        annotated = img_bgr.copy()

        oh, ow = img_bgr.shape[:2]
        eh, ew = enhanced.shape[:2]
        sx, sy = ow/ew, oh/eh

        try:
            faces = face_app.get(enhanced)
        except Exception as e:
            st.error(f"Detection error: {e}")
            st.stop()

        results = []
        for face in faces:
            det_score = float(face.det_score)
            if det_score < conf_threshold:
                continue
            raw_age       = float(face.age)
            corrected_age = apply_age_correction(raw_age, age_offset)
            gender        = parse_gender(face.gender)
            bx = face.bbox
            x1 = max(0,  int(bx[0]*sx))
            y1 = max(0,  int(bx[1]*sy))
            x2 = min(ow, int(bx[2]*sx))
            y2 = min(oh, int(bx[3]*sy))
            results.append({
                "age": corrected_age, "raw_age": int(round(raw_age)),
                "gender": gender, "face_conf": det_score,
                "box": (x1,y1,x2,y2)
            })
            draw_on_image(annotated, x1, y1, x2, y2,
                         corrected_age, raw_age, gender, det_score)

    # ── layout: image left, results right ──────────────────
    col_img, col_res = st.columns([3, 2], gap="large")

    with col_img:
        st.markdown("### 🖼️ Result")
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        st.image(annotated_rgb, use_column_width=True)

        # Download button
        result_pil = Image.fromarray(annotated_rgb)
        buf = io.BytesIO()
        result_pil.save(buf, format="PNG")
        st.download_button(
            label="💾 Download Result",
            data=buf.getvalue(),
            file_name="age_gender_result.png",
            mime="image/png"
        )

    with col_res:
        st.markdown("### 📊 Results")

        if not results:
            st.warning("No faces detected. Try lowering the **Min Detection Score** slider in the sidebar.")
        else:
            st.success(f"✅ {len(results)} face{'s' if len(results)>1 else ''} detected")

            for i, r in enumerate(results, 1):
                age    = r["age"]
                raw    = r["raw_age"]
                gender = r["gender"]
                conf   = r["face_conf"]
                diff   = age - raw

                badge_class = "gender-badge-male" if gender == "Male" else "gender-badge-female"
                diff_color  = "#06D6A0" if diff <= 0 else "#FF6B6B"
                diff_str    = f"{'+' if diff>=0 else ''}{diff}"

                st.markdown(f"""
<div class="face-card">
  <h4>Face #{i}</h4>
  <div style="display:flex; align-items:center; gap:20px; margin-bottom:12px;">
    <div>
      <div style="color:#8AADA8; font-size:12px;">AGE</div>
      <div class="age-value">{age_emoji(age)} {age}</div>
      <div class="raw-value">model raw: {raw}
        <span style="color:{diff_color}; font-size:11px;">
          ({diff_str} offset)
        </span>
      </div>
    </div>
    <div>
      <div style="color:#8AADA8; font-size:12px; margin-bottom:6px;">GENDER</div>
      <span class="{badge_class}">
        {'♂' if gender=='Male' else '♀'}  {gender}
      </span>
    </div>
  </div>
  <div class="conf-value">Detection confidence: {conf*100:.0f}%</div>
</div>
""", unsafe_allow_html=True)

            # Summary metrics
            st.markdown("---")
            st.markdown("### 📈 Summary")
            mcols = st.columns(3)
            mcols[0].metric("Faces Found",   len(results))
            mcols[1].metric("Avg Age",
                            f"{int(round(sum(r['age'] for r in results)/len(results)))} yrs")
            mcols[2].metric("Males / Females",
                            f"{sum(1 for r in results if r['gender']=='Male')} / "
                            f"{sum(1 for r in results if r['gender']=='Female')}")

else:
    # Empty state
    st.markdown("""
<div style="
  text-align:center; padding:60px 20px;
  border: 2px dashed #264040; border-radius:16px;
  color:#8AADA8;
">
  <div style="font-size:48px;">🖼️</div>
  <div style="font-size:18px; margin-top:12px;">Upload an image above to get started</div>
  <div style="font-size:13px; margin-top:8px;">Supports JPG · PNG · BMP · WebP</div>
</div>
""", unsafe_allow_html=True)
