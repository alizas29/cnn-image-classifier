import os
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# =====================================================================
# KONFIGURASI — perlu disesuaikan dengan hasil training model
# =====================================================================
MODEL_PATH = "model/model.keras"
IMG_SIZE = (224, 224)                                # ukuran input model
CLASS_NAMES = ["kelas_1", "kelas_2", "kelas_3"]      # urutan HARUS sama dgn training
NORMALIZATION = "rescale"     # "rescale" (dibagi 255) atau "mobilenet" (preprocess_input)
LOW_CONFIDENCE = 0.60         # di bawah ini dianggap model kurang yakin
MODEL_INFO = "CNN + Transfer Learning (MobileNetV2)"  # deskripsi singkat model

GROUP_NAME = "Kelompok 4"
COURSE = "Artificial Intelligence — Kelas JWCA (LEC)"
MEMBERS = [
    ("Giorgio Wilson Wong", "2902692705"),
    ("Raihan Zaky Noordiantoro", "2902693802"),
    ("Aryuka Falian Jornasena", "2902692402"),
    ("Aliza Dzakiyyah Fathimah Az Zahra", "2902701754"),
    ("Asysyifa Isharul Jannah", "2902692604"),
]
# =====================================================================

st.set_page_config(page_title="CNN Image Classifier — Kelompok 4",
                   page_icon="🧠", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1100px;}
.hero {padding: 1.4rem 1.6rem; border-radius: 14px;
       background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
       color: #fff; margin-bottom: 1.5rem;}
.hero h1 {margin: 0; font-size: 2rem; color: #fff;}
.hero p {margin: .35rem 0 0; opacity: .9;}
.result-card {padding: 1.2rem 1.4rem; border-radius: 12px;
              border: 1px solid rgba(128,128,128,.25);
              background: rgba(124,58,237,.08); margin-bottom: 1rem;}
.result-label {font-size: .85rem; opacity: .7; margin: 0;}
.result-class {font-size: 2rem; font-weight: 700; margin: .1rem 0 0;}
.member {font-size: .88rem; line-height: 1.35; margin-bottom: .45rem;}
.member small {opacity: .65;}
.footer {text-align: center; opacity: .55; font-size: .8rem; margin-top: 3rem;}
</style>
""", unsafe_allow_html=True)

model_ready = os.path.exists(MODEL_PATH)

# ---------------------------- SIDEBAR --------------------------------
with st.sidebar:
    st.markdown(f"## 👥 {GROUP_NAME}")
    st.caption(COURSE)
    st.markdown("**Anggota:**")
    for name, nim in MEMBERS:
        st.markdown(f"<div class='member'>{name}<br><small>{nim}</small></div>",
                    unsafe_allow_html=True)

    st.divider()
    st.markdown("### ⚙️ Info Model")
    st.markdown(f"- **Arsitektur:** {MODEL_INFO}")
    st.markdown(f"- **Input:** {IMG_SIZE[0]}×{IMG_SIZE[1]} px")
    st.markdown(f"- **Jumlah kelas:** {len(CLASS_NAMES)}")
    with st.expander("Daftar kelas"):
        for c in CLASS_NAMES:
            st.markdown(f"- {c}")
    if model_ready:
        st.success("Model aktif ✅")
    else:
        st.warning("Model belum dipasang ⏳")

# ---------------------------- HEADER ---------------------------------
st.markdown("""
<div class="hero">
  <h1>🧠 CNN Image Classifier</h1>
  <p>Upload gambar, dan model Convolutional Neural Network akan memprediksi kelasnya.</p>
</div>
""", unsafe_allow_html=True)


# ---------------------------- MODEL ----------------------------------
@st.cache_resource(show_spinner="Memuat model...")
def load_model():
    import tensorflow as tf
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess(img: Image.Image) -> np.ndarray:
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    if NORMALIZATION == "mobilenet":
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        arr = preprocess_input(arr)
    else:
        arr = arr / 255.0
    return np.expand_dims(arr, axis=0)


# ---------------------------- MAIN -----------------------------------
uploaded = st.file_uploader("📤 Pilih gambar (JPG / PNG)",
                            type=["jpg", "jpeg", "png"])

if not uploaded:
    st.info("Silakan upload gambar untuk memulai prediksi.")
    with st.expander("💡 Tips agar hasil prediksi akurat"):
        st.markdown(
            "- Gunakan gambar yang jelas dan tidak blur\n"
            "- Objek utama berada di tengah dan cukup besar\n"
            "- Hindari gambar dengan banyak objek sekaligus\n"
            "- Gunakan gambar yang termasuk salah satu kelas di sidebar"
        )
else:
    img = Image.open(uploaded).convert("RGB")
    col_img, col_res = st.columns([1, 1], gap="large")

    with col_img:
        st.subheader("🖼️ Gambar Input")
        st.image(img, caption=f"{uploaded.name} • {img.size[0]}×{img.size[1]} px")

    with col_res:
        st.subheader("📊 Hasil Prediksi")
        if not model_ready:
            st.warning("Model belum dipasang. Upload `model.keras` ke folder `model/` "
                       "di repository GitHub.")
        else:
            try:
                model = load_model()
                with st.spinner("Memprediksi..."):
                    probs = model.predict(preprocess(img), verbose=0)[0]

                if len(probs) != len(CLASS_NAMES):
                    st.error(f"Jumlah output model ({len(probs)}) tidak sama dengan "
                             f"jumlah CLASS_NAMES ({len(CLASS_NAMES)}). "
                             "Periksa bagian KONFIGURASI.")
                else:
                    idx = int(np.argmax(probs))
                    conf = float(probs[idx])

                    st.markdown(
                        f"<div class='result-card'><p class='result-label'>Kelas terprediksi</p>"
                        f"<p class='result-class'>{CLASS_NAMES[idx]}</p></div>",
                        unsafe_allow_html=True)

                    st.metric("Confidence", f"{conf * 100:.1f}%")
                    st.progress(conf)

                    if conf < LOW_CONFIDENCE:
                        st.warning("⚠️ Model kurang yakin dengan prediksi ini. "
                                   "Gambar mungkin tidak termasuk kelas yang dikenali.")
                    else:
                        st.success("✅ Model cukup yakin dengan prediksi ini.")

                    st.markdown("**Probabilitas per kelas**")
                    order = np.argsort(probs)[::-1]
                    chart_df = pd.DataFrame(
                        {"Probabilitas": [float(probs[i]) for i in order]},
                        index=[CLASS_NAMES[i] for i in order])
                    st.bar_chart(chart_df, horizontal=True)

                    with st.expander("Lihat detail angka"):
                        st.table({
                            "Kelas": [CLASS_NAMES[i] for i in order],
                            "Probabilitas": [f"{probs[i] * 100:.2f}%" for i in order],
                        })
            except Exception as e:
                st.error(f"Terjadi error saat prediksi: {e}")

st.markdown(f"<div class='footer'>{GROUP_NAME} • {COURSE} • "
            "Dibuat dengan Streamlit & TensorFlow</div>", unsafe_allow_html=True)
