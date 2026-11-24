# Lembar Kontribusi Anggota & Pernyataan Penggunaan AI

## 1. Pembagian Peran dan Kontribusi Anggota Tim (Kelompok 2 Orang)

Setiap anggota kelompok berkontribusi aktif secara merata dalam perancangan, implementasi, evaluasi eksperimen, dan penyusunan laporan, dengan riwayat pengerjaan yang terdokumentasi melalui *fork*, *commits*, dan *pull requests* (PR) di GitHub:

| Nama Anggota | NIM | Peran Utama & Modul yang Dikerjakan | Tautan Pull Request (PR) |
|---|---|---|---|
| **ZACKY YUSUP HAKIM** | 101032300183 | **Task 1 (Multimodal Classification) & Lead Pipeline**: Eksplorasi label noise inspektur pabrik, penanganan missing sensor modality, arsitektur Gated Multimodal Unit (GMU), evaluasi Macro-F1, serta integrasi pipeline dan pembuatan notebook demonstrasi. | [PR #1: Task 1 Multimodal Classification & Pipeline](https://github.com/sebastianvettel1212/UTS-Deep-Learning/pull/1) |
| **MUHAMMAD ALIF NAUFAL** | 101032300098 | **Task 2 (RUL Regression) & Task 3 (Multimodal Clustering)**: Prapemrosesan deret waktu 21 sensor, perumusan Asymmetric Loss, pemodelan BiLSTM-Attention, kalibrasi MC Dropout, Joint Multimodal Autoencoder, penentuan K optimal, dan Deep Embedded Clustering (DEC). | [PR #2: Task 2 RUL Regression & Task 3 DEC Clustering](https://github.com/sebastianvettel1212/UTS-Deep-Learning/pull/2) |

*Pernyataan Tanggung Jawab Bersama: Seluruh anggota kelompok telah membaca, meninjau, menguji, dan menyetujui seluruh kode serta isi laporan (`laporan.pdf`). Setiap anggota siap untuk memaparkan dan menjelaskan seluruh aspek teknis dan metodologis proyek pada ujian lisan (oral exam).*

---

## 2. Pernyataan Penggunaan Alat Bantu AI (AI Tools Declaration)

Sesuai dengan panduan integritas akademik dan ketentuan ujian take-home (*"AI tools: allowed. State the tool, the parts it helped with, and how the group verified the output"*), berikut adalah rincian transparan terkait pemanfaatan alat bantu AI:

1. **Alat Bantu yang Digunakan**: 
   - Antigravity AI Assistant (Google DeepMind)

2. **Bagian yang Dibantu oleh AI**:
   - Membantu penyusunan struktur kerangka (*boilerplate*) kode PyTorch untuk pipeline modular (dataset loader, training loops, dan checkpointing).
   - Membantu perumusan formulasi matematis untuk *smooth Huber clamping* pada fungsi kerugian asimetris Task 2 guna menghindari ledakan gradien eksponensial selama awal epoch.
   - Membantu perancangan kode visualisasi diagnostik multi-panel matplotlib untuk eksplorasi distorsi, kurva degradasi, dan proyeksi t-SNE.
   - Membantu otomatisasi perapian tata letak dokumen PDF laporan akhir menggunakan ReportLab.

3. **Metode dan Langkah Verifikasi Output oleh Tim**:
   - **Verifikasi Logika & Arsitektur:** Setiap blok arsitektur jaringan saraf diverifikasi dimensi tensornya pada setiap layer melalui pengujian *sanity check* satu batch (dummy tensor pass) sebelum pelatihan penuh dijalankan.
   - **Verifikasi Penanganan Nilai Hilang:** Logika forward-fill, masking sensor, dan modality dropout diuji secara khusus dengan unit test untuk memastikan tidak ada kebocoran informasi (*data leakage*) atau nilai NaN yang merambat ke fungsi kerugian.
   - **Verifikasi Metrik Independen:** Seluruh metrik evaluasi (Macro-F1, RMSE, Asymmetric Score, Silhouette Score, Davies-Bouldin) dihitung menggunakan modul metrik terpisah berbasis library baku (*scikit-learn* dan *NumPy*), diverifikasi terhadap nilai target acuan.
   - **Pemeriksaan Kode Kolektif (*Peer Review*):** Tidak ada kode yang diintegrasikan ke branch `main` tanpa melalui proses review dan persetujuan oleh kedua anggota kelompok pada setiap pull request GitHub.
