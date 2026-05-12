# 🌾 Dashboard Prediksi Produktivitas Pertanian Berbasis Perubahan Iklim

Sistem Pendukung Keputusan (DSS) untuk menganalisis pengaruh faktor lingkungan terhadap tingkat produktivitas hasil panen menggunakan Machine Learning.

## 📋 Deskripsi Proyek

Dashboard ini dirancang untuk membantu pengguna dalam menganalisis pengaruh faktor lingkungan terhadap tingkat produktivitas hasil panen. Sistem mengintegrasikan visualisasi data interaktif dan model machine learning untuk memberikan gambaran kondisi pertanian berdasarkan variabel iklim dan praktik pengelolaan lahan.

### 🎯 Fitur Utama

- **🏠 Home**: Informasi umum proyek, ringkasan dataset, dan performa model
- **📊 Dataset Overview**: Karakteristik data, distribusi kelas, dan analisis atribut
- **🔍 EDA (Exploratory Data Analysis)**: Visualisasi distribusi dan hubungan antar variabel
- **🤖 ML Prediction**: Simulasi prediksi real-time tanpa perlu upload dataset
- **📈 Model Evaluation**: Perbandingan performa Naive Bayes vs Decision Tree SMOTE

### 🧠 Model Machine Learning

- **Naive Bayes**: Model probabilistik dengan asumsi independensi fitur
- **Decision Tree + SMOTE**: Menangani data tidak seimbang dengan akurasi lebih tinggi

## 🚀 Cara Menjalankan

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Jalankan Dashboard

```bash
streamlit run app.py
```

Dashboard akan membuka di browser pada `http://localhost:8501`

## 📁 Struktur File

```
Dashboard/
├── app.py                          # Aplikasi Streamlit utama
├── cleaned_agriculture.csv         # Dataset yang telah dibersihkan
├── requirements.txt                # Dependencies
├── naive_bayes_model.pkl          # Model Naive Bayes
├── dt_smote_model.pkl             # Model Decision Tree SMOTE
├── encoder.pkl                     # Label Encoder
├── feature_columns.pkl             # Kolom fitur
└── README.md                       # Dokumentasi (file ini)
```

## 📊 Data Features

### Variabel Iklim
- **Average_Temperature_C**: Suhu rata-rata (°C)
- **Total_Precipitation_mm**: Curah hujan (mm)
- **CO2_Emissions_MT**: Emisi CO₂ (Metrik Ton)
- **Extreme_Weather_Events**: Kejadian cuaca ekstrem

### Variabel Pengelolaan Lahan
- **Irrigation_Access_%**: Akses irigasi (%)
- **Soil_Health_Index**: Indeks kesehatan tanah
- **Fertilizer_Use_KG_per_HA**: Penggunaan pupuk (KG/HA)
- **Pesticide_Use_KG_per_HA**: Penggunaan pestisida (KG/HA)

### Variabel Kategorikal
- **Crop_Type**: Jenis tanaman (9 kategori)
  - Coffee, Corn, Cotton, Fruits, Rice, Soybeans, Sugarcane, Vegetables, Wheat
- **Adaptation_Strategies**: Strategi adaptasi (4 kategori)
  - Drought-resistant Crops
  - No Adaptation
  - Organic Farming
  - Water Management

### Target Variable
- **Yield_Category**: Kategori produktivitas
  - 🔴 Rendah (Low)
  - 🟡 Sedang (Medium)
  - 🟢 Tinggi (High)

## 🎨 Tema Visual

Dashboard menggunakan tema **Navy Blue** dengan aksen **Gold** untuk tampilan profesional dan mudah dibaca:
- **Warna Utama**: Navy Blue (#001f3f)
- **Warna Sekunder**: Light Navy (#003d5c)
- **Aksen**: Gold (#FFD700)

## 💡 Cara Menggunakan Dashboard

### 1. Home Page
- Lihat ringkasan proyek dan tujuannya
- Periksa statistik dataset dan performa model terbaik
- Pelajari fitur-fitur dashboard

### 2. Dataset Overview
- Analisis distribusi kategori produktivitas
- Lihat statistik deskriptif semua fitur
- Pahami komposisi jenis tanaman dan strategi adaptasi

### 3. EDA Page
- Visualisasi distribusi setiap variabel
- Lihat hubungan antar variabel dengan produktivitas
- Identifikasi pola dan tren dalam data

### 4. ML Prediction
- Masukkan parameter lingkungan dan pengelolaan lahan
- Pilih jenis tanaman dan strategi adaptasi
- Dapatkan prediksi produktivitas dari kedua model
- Lihat probabilitas dan analisis risiko
- Terima rekomendasi berdasarkan hasil prediksi

### 5. Model Evaluation
- Bandingkan performa Naive Bayes vs Decision Tree SMOTE
- Analisis confusion matrix untuk setiap model
- Lihat detailed classification reports

## 📈 Interpretasi Hasil Prediksi

### Tingkat Risiko
- **🟢 RENDAH**: Produktivitas tinggi dari kedua model
- **🟡 SEDANG**: Prediksi berbeda antar model atau produktivitas sedang
- **🔴 TINGGI**: Produktivitas rendah dari salah satu atau kedua model

### Rekomendasi
Sistem memberikan rekomendasi spesifik berdasarkan tingkat risiko:
- **Rendah**: Lanjutkan praktik pengelolaan yang ada
- **Sedang**: Monitor kondisi dan siapkan alternatif
- **Tinggi**: Tingkatkan irigasi, perbaiki tanah, dan adaptasi lebih baik

## 🔧 Customization

### Mengubah Tema Warna
Edit bagian CSS di bawah `# ==================== STYLING ====================` dalam file `app.py`:

```python
navy_blue = "#001f3f"      # Ubah warna navy
light_navy = "#003d5c"     # Ubah warna secondary
accent_gold = "#FFD700"    # Ubah warna aksen
```

## 📝 Teknologi

- **Streamlit**: Framework web untuk dashboard interaktif
- **Pandas & NumPy**: Data manipulation
- **Scikit-learn**: Machine Learning
- **Plotly**: Interactive visualizations
- **Imbalanced-learn**: SMOTE untuk handle imbalanced data

## 📊 Dataset Statistics

- **Total Sampel**: Sesuai dengan cleaned_agriculture.csv
- **Total Fitur**: 21 variabel
- **Kelas Target**: 3 kategori (rendah, sedang, tinggi)
- **Missing Values**: 0 (data sudah dibersihkan)

## 🤝 Support

Untuk pertanyaan atau masalah:
1. Pastikan semua dependencies terinstall dengan benar
2. Cek bahwa semua file model (.pkl) ada di folder yang sama
3. Pastikan cleaned_agriculture.csv di folder yang sama

## 📄 License

Proyek ini adalah bagian dari tugas Data Mining - Semester 6

## 🎓 Saran Pengembangan Lebih Lanjut

- Tambahan model (Random Forest, SVM)
- Fitur upload dataset custom
- Export hasil prediksi ke CSV
- Database integration
- API endpoint
- Mobile application
