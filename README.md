# Studio 2 — Klasifikasi Sampah · Kelompok 16

Deep Learning (IF25-40401) · Teknik Informatika ITERA · Semester Ganjil 2026/2027

## Anggota Kelompok

| No | Nama Lengkap    | NIM          | Kontribusi utama |
|----|-----------------|--------------|------------------|
| 1  | SILVIA          | 123140133    | (isi)            |
| 2  | TRI PUTRI SORMIN | 123140013   | (isi)            |

## Ringkasan Pendekatan

- **Arsitektur / backbone**: EfficientNet-B0 pretrained ImageNet (timm), head 5 kelas. Baseline pembanding: ResNet-18 pretrained.
- **Strategi pelatihan**: fine-tuning seluruh layer, AdamW (lr maks 5e-4, weight decay 1e-2), scheduler OneCycleLR, 15 epoch, batch size 32, mixed precision (AMP), label smoothing 0,1, class weight (agar kelas dengan data sedikit seperti kertas_kardus lebih diperhatikan). Model terbaik dipilih berdasarkan Macro-F1 pada split validasi lokal.
- **Augmentasi**: RandomResizedCrop (scale 0,5–1,0), horizontal flip, TrivialAugmentWide, RandomErasing (p=0,25).
- **Data tambahan**: tidak ada.
- **Ukuran input & preprocessing**: Resize 292 → CenterCrop 256, normalisasi ImageNet (mean/std standar).

## Hasil

| Putaran | Commit | Macro-F1 (split validasi lokal) | Macro-F1 (diumumkan asisten) |
|---------|--------|----------------------------------|------------------------------|
| Preview 1 (14 Okt) | - | - | - |
| Preview 2 (21 Okt) | | 0,9401 | |
| Final (25 Okt)     | | | |

Catatan eksperimen singkat:

- Split lokal: 15% data train (410 gambar, seed 42) dipisahkan menjadi `my_val`.
- Baseline ResNet-18 (8 epoch, `train_baseline.py`): Macro-F1 0,9397, akurasi 0,9439. Kelas terlemah: kertas_kardus (F1 0,885).
- EfficientNet-B0 (15 epoch, `train_improved.py`): Macro-F1 0,9401, akurasi 0,9415. F1 kertas_kardus naik menjadi 0,922 berkat class weight, sementara gelas_plastik dan styrofoam turun sedikit. Selisih skor keseluruhan sangat kecil dan masih dalam batas kebetulan karena `my_val` kecil.
- Rencana: mencoba backbone lebih besar, melatih ulang pada seluruh data (`--full`), dan mengevaluasi ulang berdasarkan skor Preview 2.

## Cara Reproduksi

```bash
pip install timm
python train_baseline.py --data ./train --epochs 8     # membuat my_train/ dan my_val/
python train_improved.py --epochs 15                   # melatih EfficientNet-B0
python evaluate.py --submission ./submission --data ./my_val
```

Catatan: `train_baseline.py` dijalankan lebih dulu untuk membuat `my_train/` dan `my_val/`, dan juga menimpa `submission/model.pth`. Jalankan `train_improved.py` setelahnya agar `model.pth` yang tersimpan adalah EfficientNet-B0.

## Struktur Repositori

```
submission/
  model.py          # get_model() dan get_transform() (EfficientNet-B0)
  model.pth         # state_dict model terbaik
evaluate.py         # salinan skrip evaluasi resmi
train_baseline.py   # baseline ResNet-18
train_improved.py   # EfficientNet-B0 (augmentasi kuat, class weight, AMP)
```
