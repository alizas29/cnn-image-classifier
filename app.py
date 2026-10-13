import os
import numpy as np
import streamlit as st
from PIL import Image

# ====== KONFIGURASI (sesuaikan dengan hasil training) ======
MODEL_PATH = "model/model.keras"
IMG_SIZE = (224, 224)                       # ukuran input model
CLASS_NAMES = ["kelas_1", "kelas_2", "kelas_3"]  # urutan HARUS sama dgn training
# ===========================================================

st.set_page_config(page_title="CNN Image Classifier")
st.title("🧠 CNN Image Classifier")
st.write("Upload gambar untuk melihat hasil prediksi model.")

@st.cache_resource
def load_model():
    import tensorflow as tf
    return tf.keras.models.load_model(MODEL_PATH)

def preprocess(img):
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0   # samakan dgn training
    return np.expand_dims(arr, axis=0)

uploaded = st.file_uploader("Pilih gambar", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Gambar yang diupload", use_container_width=True)

    if not os.path.exists(MODEL_PATH):
        st.info("⏳ Model belum dipasang. Upload model.keras ke folder model/.")
    else:
        with st.spinner("Memprediksi..."):
            model = load_model()
            probs = model.predict(preprocess(img))[0]
        idx = int(np.argmax(probs))
        st.success(f"Prediksi: **{CLASS_NAMES[idx]}** ({probs[idx]*100:.1f}%)")
        st.bar_chart({name: float(p) for name, p in zip(CLASS_NAMES, probs)})
