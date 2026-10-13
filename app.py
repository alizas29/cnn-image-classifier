import streamlit as st
from PIL import Image

st.set_page_config(page_title="CNN Image Classifier")
st.title("🧠 CNN Image Classifier")
st.write("Upload gambar untuk melihat hasil prediksi model.")

uploaded = st.file_uploader("Pilih gambar", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Gambar yang diupload", use_container_width=True)
    st.info("⏳ Model belum dipasang. Prediksi akan tampil di sini.")
