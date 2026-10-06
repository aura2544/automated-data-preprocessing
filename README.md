# Data Preprocessing & Text Similarity Analyzer 🔬

Aplikasi web dashboard modern berbasis **Python** dan **Streamlit** yang dirancang khusus untuk memfasilitasi mahasiswa dan praktisi data dalam merancang:
1. **Automated Data Preprocessing & Feature Engineering** menggunakan `Scikit-Learn` (`Pipeline`, `ColumnTransformer`, `SimpleImputer`, `StandardScaler`, `OneHotEncoder`, `OrdinalEncoder`) yang **100% bebas dari kebocoran data (Data Leakage)**.
2. **Text Cosine Similarity** menggunakan **Binary Representation** (1/0 tanpa TF-IDF) lengkap dengan rincian perhitungan aljabar linier step-by-step.

---

## 🌟 Fitur Utama

### 1. Automated Data Preprocessing & Pipeline Bebas Data Leakage (Fitur Utama)
- **Upload Dataset & Sample Data:** Mendukung upload dataset file `.csv` serta tombol satu-klik *"Gunakan Dataset Contoh"* yang memuat 1.000 baris data dengan tipe campuran (numerik, nominal, ordinal), missing values, dan outliers.
- **Eksplorasi Data Interaktif:** Ringkasan statistik, ringkasan kartu metrik (Total Data, Total Atribut, Missing, Numerical, Nominal, Ordinal), struktur tabel tipe data, dan visualisasi distribusi frekuensi.
- **Deteksi & Imputasi Missing Values:** Deteksi otomatis nilai kosong per kolom dengan Scikit-Learn `SimpleImputer` (`strategy='median'` untuk numerik dan `strategy='most_frequent'` untuk kategorikal).
- **Deteksi & Penanganan Outlier (Metode IQR):** Menghitung Q1, Q3, IQR, Lower Bound, dan Upper Bound. Dilengkapi visualisasi boxplot dan Scikit-Learn custom transformer `IQROutlierCapper` untuk capping/Winsorization yang dipelajari eksklusif dari data training.
- **Identifikasi & Pengelompokan Tipe Kolom:** Klasifikasi otomatis kolom menjadi Numerik, Nominal, dan Ordinal, dengan antarmuka untuk menyesuaikan hierarki tingkatan urutan ordinal secara manual.
- **Feature Engineering Otomatis:** Transformasi logaritmik (`log1p`), rasio finansial (`Expense_to_Income_Ratio`), kombinasi aritmatika (`Net_Savings`), serta binning umur (`Age_Group`) disertai tabel perbandingan jumlah fitur sebelum dan sesudah.
- **Scikit-Learn Pipeline & ColumnTransformer:** Arsitektur pipeline modular dengan pembagian Train/Test Split terlebih dahulu. Pipeline hanya di-`fit()` pada data training, dan data testing hanya di-`transform()`, menjamin **Zero Data Leakage**.
- **Download Hasil Preprocessing:** Ekspor data training terproses, data testing terproses, dan dataset gabungan ke format `.csv`.

### 2. Text Cosine Similarity (Binary Representation)
- **Input Teks Dinamis:** Tambah, edit, dan hapus beberapa teks/status (S1, S2, S3, dst.) atau reset ke contoh soal.
- **Pipeline Preprocessing Teks:** *Case Folding* (lowercase) ➔ *Remove Punctuation* (penghapusan tanda baca) ➔ *Tokenization* (pemecahan kata).
- **Binary Representation Matrix:** Ekstraksi kosakata unik (*vocabulary*) dan pembentukan matriks biner ($1$ = kata muncul, $0$ = kata tidak muncul) tanpa menggunakan TF-IDF.
- **Cosine Similarity:** Menghitung skor kemiripan semantik antar-pasangan unik teks:
  $$\text{Cos}(A, B) = \frac{A \cdot B}{\|A\| \times \|B\|}$$
- **Visualisasi & Card Pasangan Terpopuler:** Grafik batang perbandingan skor, kartu sorotan pasangan paling mirip, serta klasifikasi kategori:
  - $\ge 0.70$ : Sangat Mirip
  - $0.40 - 0.69$ : Cukup Mirip
  - $0.20 - 0.39$ : Kurang Mirip
  - $< 0.20$ : Tidak Mirip
- **Rincian Perhitungan Langkah-demi-Langkah (Expandable Section):** Menampilkan representasi vektor, irisan kata, Dot Product, Magnitude, dan rumus pecahan untuk presentasi akademik.

---

## 📁 Struktur Direktori Project

```text
c:\SEM 7\COSINE\
│
├── app.py                     # Entry point aplikasi dashboard Streamlit
│
├── modules/
│   ├── __init__.py
│   ├── preprocessing.py       # Scikit-Learn Pipeline & ColumnTransformer engine
│   ├── feature_engineering.py # Transformasi fitur row-wise (log, rasio, binning)
│   ├── outlier.py             # Deteksi IQR & Scikit-Learn IQROutlierCapper transformer
│   └── similarity.py          # Preprocessing teks, matriks biner, & Cosine Similarity
│
├── utils/
│   ├── __init__.py
│   └── helpers.py             # Custom CSS, deteksi tipe kolom heuristik, session state
│
├── data/
│   └── sample_dataset.csv     # Dataset 1,000 baris dengan missing value & outliers
│
├── requirements.txt           # Daftar pustaka yang dibutuhkan
└── README.md                  # Dokumentasi teknis aplikasi
```

---

## 🚀 Cara Menjalankan Aplikasi

### 1. Prasyarat
Pastikan Python 3.10 atau versi lebih baru telah terpasang di sistem Anda.

### 2. Instalasi Dependensi
Jalankan perintah berikut di terminal:

```bash
pip install -r requirements.txt
```

### 3. Menjalankan Dashboard
Jalankan aplikasi dengan perintah:

```bash
streamlit run app.py
```

Setelah perintah dijalankan, antarmuka web interaktif akan otomatis terbuka di peramban Anda (biasanya di `http://localhost:8501`).

---

## 🛡 Prinsip Anti-Kebocoran Data (Data Leakage Protection)

Salah satu kesalahan paling mendasar dalam data science adalah menghitung parameter statistik (seperti mean, median, standar deviasi, min, max, batas IQR, dan vocabulary) menggunakan seluruh dataset sebelum pembagian *train-test split*.

Aplikasi ini mengimplementasikan prinsip perlindungan kebocoran data secara ketat dengan mengintegrasikan Feature Engineering ke dalam Scikit-Learn Pipeline:
```text
RAW DATA
   │
   ▼
TRAIN / TEST SPLIT
   │
   ▼
Scikit-Learn Pipeline (fit HANYA pada X_train)
   │
   ▼
Feature Engineering (FeatureEngineer: Log_Income, Expense_to_Income_Ratio, Net_Savings, Age_Group)
   │
   ├─► Numerical Imputation (SimpleImputer: Median dari Train)
   │   └─► IQR Outlier Capping (IQROutlierCapper: Bounds dari Train)
   │       └─► Scaling (StandardScaler: μ & σ dari Train)
   │
   ├─► Nominal Imputation (SimpleImputer: Modus dari Train)
   │   └─► One-Hot Encoding (OneHotEncoder: Kategori dari Train)
   │
   └─► Ordinal Imputation (SimpleImputer: Modus dari Train)
       └─► Ordinal Encoding (OrdinalEncoder: Urutan Eksplisit)
   │
   ▼
PROCESSED DATA (X_train_proc & X_test_proc)
```

Dengan alur ini:
1. `pipeline.fit(X_train)` adalah satu-satunya proses fitting.
2. `X_test` ditransformasikan secara murni via `pipeline.transform(X_test)` tanpa pernah di-fit ulang.
3. Seluruh missing values tertangani tuntas (0 missing values).

