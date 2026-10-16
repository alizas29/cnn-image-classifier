# 🧠 CNN Image Classifier — Kelompok 4

Aplikasi web sederhana untuk mengklasifikasikan gambar menggunakan model **Convolutional Neural Network (CNN)** yang telah dioptimasi. Pengguna cukup mengupload gambar, lalu aplikasi menampilkan kelas hasil prediksi beserta tingkat keyakinan (*confidence*) model.

Proyek ini dibuat untuk **Tugas Kelompok 2 (Week 8): Optimasi dan Implementasi dalam Aplikasi Sederhana**, mata kuliah *Artificial Intelligence*, kelas **JWCA (LEC)**, BINUS University.

🔗 **Live demo:** https://cnn-image-classifier-jwca-group4.streamlit.app

---

## ✨ Fitur

- Upload gambar format **JPG / PNG**
- Prediksi kelas gambar menggunakan model CNN
- Menampilkan **confidence** dalam persen dan progress bar
- Peringatan otomatis jika model kurang yakin (confidence < 60%)
- Grafik dan tabel **probabilitas untuk setiap kelas**
- Sidebar berisi informasi kelompok dan model

## 🛠️ Teknologi

| Komponen | Teknologi |
|---|---|
| Model | TensorFlow / Keras (CNN 3 blok konvolusi) |
| Optimasi | Hyperparameter tuning (Keras Tuner RandomSearch), Dropout, Batch Normalization, eksperimen transfer learning MobileNetV2 |
| Antarmuka web | Streamlit |
| Dataset | CIFAR-10 (60.000 gambar 32×32 piksel, 10 kelas) |
| Hosting | Streamlit Community Cloud |
| Version control | GitHub |

## 📁 Struktur Folder

```
cnn-image-classifier/
├── app.py              # Aplikasi Streamlit (UI + prediksi)
├── requirements.txt    # Daftar library Python
├── README.md           # Dokumentasi proyek
├── model/
│   └── model.keras     # Model CNN hasil training & tuning
└── notebooks/
    └── tuning.ipynb    # Notebook training, tuning, dan evaluasi model
```

## 🚀 Menjalankan Secara Lokal

1. **Clone repository**
   ```bash
   git clone https://github.com/alizas29/cnn-image-classifier.git
   cd cnn-image-classifier
   ```

2. **(Opsional) Buat virtual environment**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS / Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Jalankan aplikasi**
   ```bash
   streamlit run app.py
   ```
   Aplikasi akan terbuka otomatis di browser pada `http://localhost:8501`.

> Disarankan menggunakan **Python 3.11**.

## 🗂️ Dataset dan Model

Model dilatih menggunakan dataset **CIFAR-10** dengan 10 kelas:
`airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`.

- Ukuran input: **32 × 32 piksel**, 3 channel (RGB)
- Normalisasi: nilai piksel dibagi 255
- Optimasi: hyperparameter tuning dengan **Keras Tuner (RandomSearch)** pada jumlah filter, Batch Normalization, jumlah neuron Dense, dropout, dan learning rate
- Proses training, tuning, transfer learning dan evaluasi lengkap ada di `notebooks/tuning.ipynb`

> Karena gambar CIFAR-10 berukuran kecil (32×32), gambar beresolusi tinggi yang diunggah akan diperkecil terlebih dahulu. Hasil terbaik didapat dari gambar dengan satu objek utama yang jelas.

## ⚙️ Konfigurasi Model

Pengaturan model berada di bagian **KONFIGURASI** paling atas file `app.py`:

| Variabel | Keterangan |
|---|---|
| `MODEL_PATH` | Lokasi file model (default: `model/model.keras`) |
| `IMG_SIZE` | Ukuran input gambar, harus sama dengan saat training |
| `CLASS_NAMES` | Daftar nama kelas, **urutannya harus sama** dengan saat training |
| `NORMALIZATION` | `"rescale"` (piksel dibagi 255) atau `"mobilenet"` (`preprocess_input`) |
| `LOW_CONFIDENCE` | Batas confidence untuk memunculkan peringatan |

### Mengganti model

1. Simpan model hasil training dengan format Keras 3:
   ```python
   model.save("model.keras")
   ```
2. Upload `model.keras` ke folder `model/` (timpa file lama).
3. Sesuaikan `CLASS_NAMES`, `IMG_SIZE`, dan `NORMALIZATION` di `app.py`.
4. Commit ke GitHub. Streamlit Cloud akan melakukan deploy ulang secara otomatis.

> Upload lewat browser GitHub dibatasi **25 MB** per file. Untuk model yang lebih besar, gunakan GitHub Desktop atau `git push` (maksimal 100 MB).

## 🔄 Alur Kerja Aplikasi

```
Upload gambar → Konversi RGB → Resize (IMG_SIZE) → Normalisasi
      → model.predict() → Ambil kelas dengan probabilitas tertinggi
      → Tampilkan kelas, confidence, dan grafik probabilitas
```

## 👥 Anggota Kelompok 4

| No | Nama | NIM |
|---|---|---|
| 1 | Giorgio Wilson Wong | 2902692705 |
| 2 | Raihan Zaky Noordiantoro | 2902693802 |
| 3 | Aryuka Falian Jornasena | 2902692402 |
| 4 | Aliza Dzakiyyah Fathimah Az Zahra | 2902701754 |
| 5 | Asysyifa Isharul Jannah | 2902692604 |

---

*Artificial Intelligence — Kelas JWCA (LEC) • BINUS University • 2026*
