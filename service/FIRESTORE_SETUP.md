# Firestore setup (~5 menit, sekali saja per GCP project)

`/protect` di `app.py` menyimpan satu dokumen per job yang selesai ke
koleksi Firestore `protection_jobs` — ini penggunaan **Firebase/Firestore**
yang ditanyakan brief description resmi lomba. Yang tersimpan hanya
parameter, metrik, dan verdict — **tidak pernah** foto atau data
pribadi pengguna.

## 1. Enable API + buat database

```bash
export PROJECT_ID=<gcp-project-milikmu>   # project sendiri, bukan dari panitia
export REGION=asia-southeast1

gcloud services enable firestore.googleapis.com --project=$PROJECT_ID

# Buat database Firestore (Native mode). Kalau sudah ada, perintah ini
# gagal dengan pesan "already exists" — itu normal, lanjut saja.
gcloud firestore databases create --location=$REGION --project=$PROJECT_ID
```

## 2. Beri IAM ke service account Cloud Run

Service memakai Application Default Credentials (service account default
Cloud Run) — tanpa file kunci. Ia butuh role `datastore.user`:

```bash
SA_NUM=$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${SA_NUM}-compute@developer.gserviceaccount.com" \
  --role="roles/datastore.user"
```

## 3. Deploy (tidak ada env wajib)

Firestore aktif secara default. Opsi env di `gcloud run deploy`:

| Env | Default | Fungsi |
|---|---|---|
| `FIRESTORE_ENABLED` | `1` | Set `0` untuk menonaktifkan (mis. dev lokal tanpa kredensial GCP) |
| `FIRESTORE_COLLECTION` | `protection_jobs` | Nama koleksi |
| `GCP_PROJECT` | (auto dari ADC) | Override project bila perlu |

## 4. Verifikasi

1. Deploy service, lalu panggil satu `/protect` (atau mode mock tidak menulis apa-apa — harus request real).
2. Cek di console: `https://console.firebase.google.com/project/$PROJECT_ID/firestore`
   — koleksi `protection_jobs` harus berisi 1 dokumen dengan field
   `strength`, `metrics`, `verdict`, dan `created_at`.
3. Response `/protect` juga mengembalikan `meta.firestore_job_id` bila
   penulisan berhasil — bukti langsung untuk demo.

## Kalau gagal

- **Tidak ada dokumen tapi API tetap 200:** normal — modul ini fail-safe.
  Cek log Cloud Run: cari baris `firestore disabled:` atau
  `firestore write skipped` untuk penyebabnya (umumnya IAM belum
  terpasang atau API belum enable).
- **Local dev:** tanpa `gcloud auth application-default login`, modul
  menonaktifkan diri sendiri dan API tetap jalan penuh.

## Kalimat siap pakai untuk brief description (jujur, sesuai implementasi)

> Every protection job is persisted to Firebase Firestore
> (`protection_jobs` collection) with its parameters, metrics and
> verdict — photos are never stored. The Nuxt frontend and the FastAPI
> cloaking engine both run on Cloud Run (asia-southeast1); Gemini
> powers the plain-language /report evaluation.
