# UTS Deep Learning - Pipeline Klasifikasi, Regresi & Klasterisasi Multimodal

Repositori ini memuat seluruh implementasi kode, model pembelajaran mendalam (*deep learning*), evaluasi metrik, dan laporan teknis untuk **Take-Home Exam Deep Learning (Semester Ganjil 2026/2027)**.

## Anggota Kelompok (2 Orang)
- **Anggota 1**: [ZACKY YUSUP HAKIM] (101032300183) – *Lead Pipeline, Task 1 Multimodal Classification*
- **Anggota 2**: [MUHAMMAD ALIF NAUFAL] (101032300098) – *Task 2 RUL Regression & Task 3 Multimodal Clustering*

---

## Ringkasan Tugas & Metrik Utama

| Tugas | Domain Data & Modalitas | Arsitektur Utama | Metrik Utama | Skor Validasi |
|---|---|---|---|:---:|
| **Task 1: Klasifikasi Defek Produk** | Citra Permukaan (64x64) + Sensor Tekan (3x128) | Gated Multimodal Unit (ResNet2D + 1D-CNN/BiGRU) | Macro-F1 | **0.9124** |
| **Task 2: Regresi Sisa Umur (RUL)** | 21 Sensor + 3 Setelan Operasi (Window W=30) | BiLSTM + Temporal Attention & MC Dropout | RMSE / Asym Score | **12.84 / 214.5** |
| **Task 3: Klasterisasi Material** | Mikrostruktur (64x64) + Spektrum (128) | Joint Multimodal Autoencoder + DEC | Silhouette / DB Index | **0.584 / 0.642** |


## Struktur Repositori

```text
UTS_DL/
├── data/                      # Dataset resmi (klasifikasi, RUL, clustering)
├── kode/                      # Seluruh modul kode sumber modular & reproducible
│   ├── Task1_Klasifikasi/     # Pipeline klasifikasi multimodal & GMU
│   ├── Task2_RUL/             # Pipeline time-series RUL & Asymmetric Loss
│   ├── Task3_Klasterisasi/    # Pipeline autoencoder & Deep Embedded Clustering (DEC)
│   ├── utils/                 # Data loader dan metrik evaluasi
│   ├── run_all.py             # Script satu perintah master reproduction
│   └── README.md              # Dokumentasi teknis kode
├── models/                    # Bobot model terlatih (.pt) dan scaler (.pkl)
├── prediksi/                  # Salinan file prediksi untuk auto-grading
├── laporan/                   # Berkas laporan & notebook demonstrasi
│   ├── laporan.pdf            # Laporan akhir lengkap (8 halaman)
│   ├── demo_task1.ipynb       # Notebook demo live Task 1 (pre-rendered outputs)
│   ├── demo_task2.ipynb       # Notebook demo live Task 2 (pre-rendered outputs)
│   ├── demo_task3.ipynb       # Notebook demo live Task 3 (pre-rendered outputs)
│   ├── demo_all_tasks.ipynb   # Master notebook end-to-end all tasks
│   └── figures/               # Gambar diagnostik dan visualisasi
├── prediksi_klasifikasi.csv   # Berkas prediksi wajib Task 1 (id_unit, kelas)
├── prediksi_rul.csv           # Berkas prediksi wajib Task 2 (id_unit, rul)
├── cluster.csv                # Berkas prediksi wajib Task 3 (id_sampel, cluster)
├── kontribusi.md              # Lembar kontribusi 2 anggota tim & AI tools declaration
├── requirements.txt           # Daftar dependensi Python
└── README.md
```

