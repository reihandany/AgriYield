# Changelog - Dashboard Prediksi Produktivitas Pertanian

## Version 2.1 (Latest) - Updated to Latest Model

### 📋 Summary
Dashboard telah diperbarui untuk menggunakan **Decision Tree Latest Version** (`initial_dt_model.pkl`). Model sebelumnya (`final_dt_model.pkl`) telah digantikan dengan versi yang lebih terbaru.

### ✨ Perubahan Utama

#### 1. **Model Update**
- ❌ Removed: `final_dt_model.pkl`
- ✅ Added: `initial_dt_model.pkl` (Latest Decision Tree Model)

#### 2. **Code Changes** (app.py)
- Updated `load_data_and_models()`: Sekarang load `initial_dt_model.pkl`
- Updated page titles: "Final Version" → "Latest Version"
- Updated model references: "Final Model" → "Latest Model"

#### 3. **Documentation Updates**
- README.md: Updated file structure & model reference
- Home page: Updated to show "Latest Version"
- ML Prediction page: Updated model label
- Model Evaluation page: Updated title

### 📁 File Changes
```
❌ final_dt_model.pkl (removed from reference)
✅ initial_dt_model.pkl (new model file)
```

### 🚀 Features
- Same feature structure (21 variables)
- Same target classes (2: rendah, tinggi)
- Improved model performance (latest version)
- Backward compatible with existing dataset

### ✅ Testing Status
- [x] No syntax errors
- [x] Model loads successfully
- [x] All pages display correctly
- [x] Predictions work as expected
- [x] Documentation updated

---

## Version 2.0 (Previous) - Model Update to Final Version

### 📋 Summary
Dashboard was updated to use **Decision Tree Final Version** (`final_dt_model.pkl`) exclusively.

...


### ✨ Perubahan Utama

#### 1. **Model Migration**
- ❌ Dihapus: `naive_bayes_model.pkl`
- ❌ Dihapus: `dt_smote_model.pkl` 
- ✅ Ditambahkan: `final_dt_model.pkl` (Decision Tree Final Version)

#### 2. **Backend Changes** (app.py)
- `load_data_and_models()`: Simplified untuk load hanya Decision Tree Final Version
- `get_model_performance()`: Updated untuk return metrics hanya untuk Decision Tree
- Removed: Semua referensi ke `nb_model` (Naive Bayes)

#### 3. **UI/UX Improvements**
- **Home Page**: 
  - Menampilkan hanya informasi Decision Tree Final Version
  - Simplified model description
  
- **ML Prediction Page**:
  - Hasil prediksi ditampilkan dalam satu card (bukan dua)
  - Risk assessment menggunakan single model prediction
  - Cleaner interface dengan layout yang lebih focus
  
- **Model Evaluation Page**:
  - Metrics ditampilkan dalam format metric cards
  - Single confusion matrix untuk Decision Tree
  - Single classification report
  
- **Sidebar**:
  - Updated information: "Models: Decision Tree (Final Version)"
  - Target class updated to 2 kelas (rendah, tinggi)

#### 4. **Documentation**
- README.md: Updated dengan referensi model terbaru
- File structure: Diperbarui untuk menampilkan final_dt_model.pkl
- Features: Dihapus "Model Comparison" dari deskripsi fitur

### 🚀 Upgrade Instructions

1. Backup model files lama (jika ada):
   ```bash
   # Opsional jika file lama masih ada
   rm naive_bayes_model.pkl
   rm dt_smote_model.pkl
   ```

2. Pastikan file `final_dt_model.pkl` ada di directory:
   ```bash
   ls final_dt_model.pkl
   ```

3. Run dashboard:
   ```bash
   streamlit run app.py
   ```

### 📊 Performance Metrics

Model Decision Tree Final Version:
- Lebih akurat dibanding Naive Bayes
- Menangani data tidak seimbang dengan baik (SMOTE technique)
- Interpretasi tree yang mudah dipahami

### 🔄 Backward Compatibility

⚠️ **Breaking Change**: Dashboard ini sekarang hanya support **final_dt_model.pkl**. Jika Anda memiliki aplikasi yang menggunakan versi lama dengan dua model, Anda perlu melakukan upgrade.

### ✅ Testing Checklist

- [x] app.py tidak ada syntax errors
- [x] Load model berjalan tanpa error
- [x] Prediction page menampilkan single model result
- [x] Model evaluation page berjalan sempurna
- [x] README documentation updated
- [x] Sidebar information correct

### 📝 Catatan

- Semua feature columns tetap sama (21 variables)
- Dataset tetap menggunakan cleaned_agriculture.csv
- Encoder dan feature columns artifacts tetap kompatibel
- Dashboard UI/UX lebih clean dan fokus

---
**Updated**: May 2026
**Status**: Ready for Production ✅
