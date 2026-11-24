# Panduan Reprodusibilitas Eksperimen (One-Command Reproduction)

Repositori ini memuat seluruh kode, model, dan pipeline evaluasi untuk Take-Home Exam Deep Learning:
- **Task 1: Multimodal Defect Classification** (Citra Permukaan + Sinyal Sensor Tekan)
- **Task 2: Time-Series Remaining Useful Life (RUL) Regression** (21 Sensor + 3 Pengaturan Operasi)
- **Task 3: Multimodal Material Clustering** (Mikrostruktur + Spektrum Komposisi)

---

## 1. Prasyarat & Lingkungan Sistem

### Dependensi & Versi Library
- **Python**: >= 3.10 (Rekomendasi: 3.11)
- **PyTorch**: >= 2.2.0 (Mendukung akselerasi CUDA pada GPU NVIDIA)
- **NumPy**: >= 1.24.0
- **Pandas**: >= 2.0.0
- **Scikit-Learn**: >= 1.3.0
- **SciPy**: >= 1.10.0
- **Matplotlib & Seaborn**: >= 3.7.0
- **ReportLab**: >= 4.0.0

### Instalasi Dependensi
```bash
# Menggunakan pip standar
pip install -r requirements.txt

# Atau menggunakan Astral uv (rekomendasi, instalasi sangat cepat)
uv pip install -r requirements.txt
```

---

## 2. Satu Perintah untuk Mereproduksi Seluruh Hasil (One-Command Reproduction)

Sesuai ketentuan ujian ("One command must reproduce every number in the report"), jalankan perintah berikut dari direktori `kode/`:

```bash
python run_all.py --seed 42
```

Perintah di atas secara otomatis akan:
1. Memuat seluruh dataset dari folder `../data/`.
2. Melatih model multimodal Task 1 (ResNet2D + 1D-CNN/BiGRU dengan GMU dan Modality Dropout), mengevaluasi metrik validasi Macro-F1, dan menghasilkan file `prediksi_klasifikasi.csv`.
3. Menjalankan prapemrosesan time-series Task 2, melatih model BiLSTM dengan Temporal Attention dan loss asimetris, mengestimasi ketidakpastian menggunakan MC Dropout, mengevaluasi RMSE dan Asymmetric Score, serta menghasilkan file `prediksi_rul.csv`.
4. Menjalankan eksplorasi distorsi Task 3, melatih Multimodal Material Autoencoder, mencari jumlah klaster optimal $K$ (Silhouette, Davies-Bouldin, Elbow), menjalankan Deep Embedded Clustering (DEC), memproyeksikan t-SNE / UMAP untuk analisis ablasi modalitas, dan menghasilkan file `cluster.csv`.
5. Menyimpan visualisasi grafik ke `../laporan/figures/`.
6. Menampilkan ringkasan metrik performa akhir.

---

## 3. Eksekusi Modul Secara Terpisah

Setiap tugas juga dapat dijalankan secara modular:

### Task 1: Klasifikasi Defek Multimodal
```bash
# Melatih model
python Task1_Klasifikasi/train.py --epochs 12 --batch_size 128 --seed 42

# Menghasilkan prediksi pengujian (prediksi_klasifikasi.csv)
python Task1_Klasifikasi/predict.py --model_path ../models/task1_best_model.pt
```

### Task 2: Regresi RUL Time-Series
```bash
# Melatih model urutan dan baseline
python Task2_RUL/train.py --epochs 15 --batch_size 256 --seed 42

# Menghasilkan prediksi pengujian (prediksi_rul.csv) dengan MC Dropout
python Task2_RUL/predict.py --model_path ../models/task2_best_model.pt
```

### Task 3: Klasterisasi Material Multimodal
```bash
# Menjalankan eksplorasi distorsi, pencarian K, DEC, dan visualisasi t-SNE
python Task3_Klasterisasi/predict.py --epochs_ae 10 --batch_size 256
```

### Kompilasi Laporan PDF
```bash
python ../laporan/generate_report.py
```

---

## 4. Format dan Verifikasi File Output Prediksi

Semua file prediksi disimpan di root direktori repositori dan di folder `prediksi/`:
- `prediksi_klasifikasi.csv`: Header `id_unit,kelas` (16.000 baris, 7 kelas valid).
- `prediksi_rul.csv`: Header `id_unit,rul` (300 baris, nilai RUL dalam rentang $[0, 150]$).
- `cluster.csv`: Header `id_sampel,cluster` (36.000 baris, integer label klaster).
