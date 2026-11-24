# AI CV Analyzer

AI CV Analyzer membantu kamu membandingkan isi CV dengan persyaratan suatu
posisi. Laporan menunjukkan skill yang sudah didukung oleh pengalaman di CV,
skill yang masih perlu dikembangkan, dan rencana belajar untuk menutup
kesenjangannya.

| Cara menjalankan | Kapan dipakai |
| --- | --- |
| [Streamlit](#local-setup) | Cara tercepat mencoba aplikasi di komputer dan versi yang bisa di-deploy ke [Streamlit Community Cloud](#deploy-on-streamlit-community-cloud). |
| [React dan Next.js](#nextjs-workspace) | Antarmuka web dengan formulir responsif, tahapan analisis, dan hasil dalam beberapa tab. |
| [CLI](#cli) | Menjalankan analisis dari terminal dan menyimpan laporan langsung ke file Markdown. |

Lihat [contoh laporan](samples/sample_output.md) untuk melihat format hasilnya.

## Cara kerja

1. **Baca CV.** Aplikasi mengambil teks dari PDF atau TXT UTF-8.
2. **Kenali skill.** LangGraph mengatur alur ekstraksi pengalaman dan skill yang tertulis atau tersirat di CV. SumoPod memproses teks melalui API chat completions.
3. **Bandingkan dengan persyaratan.** Aplikasi mengambil persyaratan dari deskripsi lowongan yang kamu tempel atau dari pencarian DDGS.
4. **Susun laporan.** Hasilnya berisi ikhtisar, kekuatan CV, kesenjangan skill, rencana belajar per minggu, dan catatan akhir. Laporan tersedia dalam bahasa Indonesia atau Inggris dan bisa diunduh sebagai `.md`.

Misalnya, jika pengalaman di CV berfokus pada Python dan posisi tujuan juga
membutuhkan Docker, laporan bisa menyarankan latihan mengemas aplikasi Python
dalam container.

<a name="local-setup"></a>

## Jalankan Streamlit di komputer

Gunakan **Python 3.11** dan jalankan perintah dari folder utama repo. Untuk
melakukan analisis sungguhan, siapkan API key SumoPod. Contoh laporan di
aplikasi bisa dibuka tanpa API key.

### 1. Instal dependensi

**Windows PowerShell**

~~~powershell
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) { py -3.11 -m venv .venv }
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
if (-not (Test-Path -LiteralPath .env)) { Copy-Item .env.example .env }
~~~

**macOS atau Linux**

~~~bash
test -x .venv/bin/python || python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
test -f .env || cp .env.example .env
~~~

Untuk virtual environment yang dibuat dengan uv tanpa pip, gunakan
`uv pip install --python .venv\Scripts\python.exe -r requirements.txt`
di Windows atau `uv pip install --python .venv/bin/python -r requirements.txt`
di macOS dan Linux.

### 2. Isi konfigurasi SumoPod

Buka `.env` di folder utama dan isi `API_KEY` dengan API key SumoPod milikmu.
Gunakan model yang tersedia untuk akunmu.
[Template .env](.env.example) · [Panduan API SumoPod](https://docs.sumopod.com/guides/ai/openai-compatible-api)

~~~env
API_KEY=your_sumopod_api_key
MODEL=glm-5.3-flash
BASE_URL=https://ai.sumopod.com/v1
~~~

`MODEL` dan `BASE_URL` memiliki nilai bawaan seperti di atas. DDGS bisa
digunakan tanpa API key pencarian. Git mengabaikan file `.env` dan
`.streamlit/secrets.toml` yang berisi kredensial lokal.

### 3. Mulai aplikasi

**Windows PowerShell**

~~~powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
~~~

**macOS atau Linux**

~~~bash
.venv/bin/python -m streamlit run app.py
~~~

Buka [Streamlit di localhost port 8501](http://127.0.0.1:8501/).

<a name="using-the-interface"></a>

## Cara menggunakan aplikasi

Untuk mencoba analisis pertama dengan data bawaan, ikuti langkah ini.

1. Centang **Gunakan CV contoh bila belum mengunggah file**. Untuk menganalisis CV sendiri, unggah PDF berisi teks atau TXT UTF-8 dengan ukuran maksimal **5 MB**.
2. Isi **Posisi tujuan** dengan `Senior AI Engineer` dan pilih **Indonesia** atau **English** di **Bahasa laporan**.
3. Pilih **Deskripsi lowongan**, lalu tempel isi [contoh deskripsi lowongan](samples/sample_job_description.txt) ke kolom deskripsi.
4. Klik **Analisis CV**. Tunggu sampai bagian **Hasil analisis** muncul.
5. Klik **Unduh laporan .md** untuk menyimpan hasil. Tab **Markdown** menyediakan teks sumber dan tombol salin.

**Lihat contoh laporan** membuka hasil yang sudah disediakan. Analisis dengan
CV contoh menjalankan pipeline sungguhan dan memakai kuota SumoPod.

### Pilih sumber persyaratan

| Mode | Cara pakai |
| --- | --- |
| **Deskripsi lowongan** | Tempel teks lowongan untuk membandingkan CV dengan persyaratan posisi tersebut secara langsung. Cocok untuk analisis pertama dan pengujian deploy. |
| **Pencarian web** | Isi posisi tujuan untuk mencari persyaratan umum lewat DDGS. Kamu bisa menempel deskripsi lowongan sebagai cadangan. |

Jika DDGS gagal dan deskripsi cadangan tersedia, aplikasi menggunakan
deskripsi tersebut dan menampilkan sumbernya di laporan. Jika pencarian gagal
dengan kolom deskripsi kosong, aplikasi meminta kamu mengisi deskripsi.

<a name="deploy-on-streamlit-community-cloud"></a>

## Deploy ke Streamlit Community Cloud

Community Cloud menjalankan `app.py` dan memasang dependensi Python dari
`requirements.txt`. Versi ini memakai antarmuka Streamlit dengan logo dan
favicon CV Analyzer.

1. Commit dan push perubahan ke branch pada repo GitHub atau fork yang kamu kelola. Sertakan `app.py`, `src/`, `samples/`, `requirements.txt`, `.streamlit/config.toml`, `web/app/icon.png`, dan `web/public/cv-analyzer-logo.png`.
2. Buka **Create app**, pilih repo dan branch tersebut, lalu isi **Main file path** dengan `app.py`. [Streamlit Community Cloud](https://share.streamlit.io/)
3. Di **Advanced settings**, pilih **Python 3.11** dan isi **Secrets** menggunakan contoh TOML di bawah. Ganti nilai API key dengan kredensial SumoPod milikmu. [Template Secrets](.streamlit/secrets.toml.example)
4. Simpan pengaturan dan klik **Deploy**. [Panduan resmi Streamlit](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)

~~~toml
API_KEY = "your_sumopod_api_key"
MODEL = "glm-5.3-flash"
BASE_URL = "https://ai.sumopod.com/v1"
~~~

<a name="verify-the-deployed-url"></a>

### Verifikasi setelah deploy

1. Buka URL `.streamlit.app` di desktop dan ponsel. Periksa logo, ikon tab browser, dan **Lihat contoh laporan**.
2. Jalankan [analisis dengan CV contoh dan deskripsi lowongan](#cara-menggunakan-aplikasi). Pastikan semua tahap selesai dan file `.md` bisa dibuka.
3. Ulangi dengan **Pencarian web** sambil membiarkan isi deskripsi sebagai cadangan. Jika DDGS berhasil, laporan menampilkan tautan sumber yang tersedia. Jika DDGS gagal, aplikasi menandai bahwa laporan memakai deskripsi cadangan.

Gunakan **Manage app** untuk membaca log jika deploy atau analisis gagal.
Pada aplikasi publik, analisis pengunjung memakai kuota SumoPod yang kamu
konfigurasi. Pembatasan akses tersedia di **Sharing → Only specific people
can view this app**. [Pengaturan akses Streamlit](https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app)

<a name="nextjs-workspace"></a>

## Jalankan React dan Next.js

Gunakan virtual environment dan `.env` dari [setup lokal](#local-setup).
Frontend memerlukan **Node.js 20.9 atau lebih baru**. Jalankan API Python dan
Next.js di dua terminal dari folder utama repo. [Persyaratan Next.js](https://nextjs.org/docs/app/getting-started/installation)

Di terminal pertama, instal dependensi lalu jalankan API.

**Windows PowerShell**

~~~powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-api.txt
.\.venv\Scripts\python.exe -m uvicorn src.http_api:app --host 127.0.0.1 --port 8000
~~~

**macOS atau Linux**

~~~bash
.venv/bin/python -m pip install -r requirements-api.txt
.venv/bin/python -m uvicorn src.http_api:app --host 127.0.0.1 --port 8000
~~~

Jika virtual environment memakai uv tanpa pip, jalankan perintah instalasi
uv pada [setup lokal](#local-setup) dengan `requirements-api.txt`.

Di terminal kedua, jalankan frontend.

~~~bash
npm --prefix web ci
npm --prefix web run dev -- --hostname 127.0.0.1
~~~

Buka [Next.js di localhost port 3000](http://127.0.0.1:3000/). Klik
**Contoh laporan** untuk melihat tampilan hasil atau **Gunakan CV contoh**
untuk menganalisis data bawaan.

Secara bawaan, Next.js meneruskan permintaan ke API Python di
`http://127.0.0.1:8000`. Jika alamat API berbeda, salin
[`web/.env.local.example`](web/.env.local.example) ke `web/.env.local` dan
atur `CV_ANALYZER_API_URL`. Simpan kredensial SumoPod di konfigurasi Python.

Untuk menjalankan build produksi lokal, hentikan server pengembangan Next.js
lalu jalankan perintah berikut dengan API Python tetap aktif.

~~~bash
npm --prefix web run build
npm --prefix web run start -- --hostname 127.0.0.1
~~~

Hosting versi Next.js memerlukan runtime Node.js, API Python yang bisa
dijangkau dari server Next.js, serta dukungan unggahan multipart dan respons
NDJSON yang dikirim bertahap.

## CLI

Jalankan contoh berikut dari folder utama repo dengan Python dari `.venv`.
Jika virtual environment belum aktif, ganti `python` dengan
`.\.venv\Scripts\python.exe` di Windows atau `.venv/bin/python` di macOS
dan Linux.

### Bandingkan dengan satu deskripsi lowongan

~~~bash
python main.py --cv samples/sample_cv.txt --role "Senior AI Engineer" --market-source job-description --job-description samples/sample_job_description.txt --language indonesia --out outputs/laporan.md
~~~

### Gunakan pencarian web dengan deskripsi cadangan

~~~bash
python main.py --cv samples/sample_cv.txt --role "Senior AI Engineer" --market-source ddgs --job-description samples/sample_job_description.txt --language english --out outputs/report-en.md
~~~

| Opsi | Keterangan |
| --- | --- |
| `--cv` | Wajib. Path ke PDF berbasis teks, TXT, atau Markdown. |
| `--role` | Wajib. Nama posisi yang ingin dianalisis. |
| `--market-source` | `ddgs` secara bawaan atau `job-description`. |
| `--job-description` | Path ke file TXT UTF-8. Wajib untuk mode `job-description` dan opsional sebagai cadangan dalam mode `ddgs`. |
| `--language` | `indonesia` secara bawaan atau `english`. |
| `--provider` | `sumopod` secara bawaan. Opsi `auto` juga menggunakan SumoPod. |
| `--out` | Path keluaran laporan Markdown. Nilai bawaan `report.md`. |

Jika berhasil, CLI mencetak lokasi file hasil. Jika gagal, CLI menampilkan
pesan error dan mengembalikan kode keluar selain nol.

## Data dan pemakaian sumber daya

- SumoPod menjalankan inferensi di layanan eksternal. Aplikasi mengirim teks CV dan persyaratan yang dipakai untuk analisis. Pencarian DDGS memakai posisi tujuan dan tahun untuk membentuk kueri.
- Streamlit dan API Python masing-masing menerima maksimal **2 analisis bersamaan** per proses. Timeout HTTP SumoPod adalah **45 detik** dengan maksimal **1 kali percobaan ulang** per permintaan.
- DDGS mengambil maksimal **5 hasil** dengan **600 karakter per cuplikan**. Pencarian memakai timeout HTTP **8 detik** dan maksimal **2 worker** per pencarian. Persyaratan dan tautan sumber publik disimpan di cache selama **6 jam** dengan kapasitas **128 entri**.
- Aplikasi membersihkan salinan file yang dipakai untuk analisis setelah proses berakhir. Streamlit menyimpan hasil dalam sesi pengguna. Unduh laporan jika ingin menyimpannya setelah sesi berakhir.
- PDF hasil scan perlu OCR agar teksnya bisa dibaca. Untuk pengujian atau demo, gunakan CV contoh atau data sintetis.

## Struktur repo

| Lokasi | Fungsi |
| --- | --- |
| [app.py](app.py) | Formulir, proses analisis, dan tampilan hasil Streamlit. |
| [main.py](main.py) | Entry point terminal. |
| [src/graph/workflow.py](src/graph/workflow.py) | Alur LangGraph yang digunakan oleh semua antarmuka. |
| [src/agents/](src/agents/) | Ekstraksi CV, analisis skill, persyaratan posisi, dan penyusunan laporan. |
| [src/llm_provider.py](src/llm_provider.py) dan [src/tools/market_search.py](src/tools/market_search.py) | Integrasi SumoPod, DDGS, dan deskripsi lowongan. |
| [src/http_api.py](src/http_api.py) dan [web/](web/) | API FastAPI dan antarmuka React/Next.js. |
| [samples/](samples/) | CV, deskripsi lowongan, dan laporan contoh. |
| [tests/](tests/) | Tes pipeline, Streamlit, dan API Python. |
| [Logo](web/public/cv-analyzer-logo.png) dan [favicon](web/app/icon.png) | Aset merek yang dipakai kedua antarmuka. |

Arsitektur, kontrak API, dan rincian pengujian tersedia di [catatan proyek](.agents/README.md).

<a name="testing"></a>

## Pengujian

Gunakan Python dari virtual environment proyek. Tes Python memakai respons
simulasi dan bisa dijalankan tanpa API key.

~~~bash
python -m unittest discover -s tests -v
~~~

Setelah memasang `requirements-api.txt` dan dependensi frontend, jalankan
pemeriksaan API dan Next.js.

~~~bash
python -m unittest discover -s tests/api -v
npm --prefix web run lint
npm --prefix web run build
npm --prefix web run typecheck
~~~

Di macOS, Linux, atau Git Bash, `bash scripts/smoke_test.sh` menjalankan tes
Python utama dengan virtual environment aktif. Perintah
`python scripts/generate_sample_output.py` menjalankan analisis live dan
memperbarui [laporan contoh](samples/sample_output.md) memakai kuota SumoPod.

### Hasil pengujian yang tercatat

- Pada **24 November 2026**, instalasi bersih Debian 11 dengan Python 3.11 lulus **20 tes otomatis**. **3 tes API** serta build, lint, dan type check Next.js juga lulus.
- Pengujian browser dengan SumoPod menghasilkan laporan Inggris dari PDF dan laporan Indonesia dari TXT. Unduhan, salin Markdown, tampilan ponsel, logo, favicon, dan validasi unggahan diperiksa.
- DDGS menghasilkan sumber pada pengujian Windows. Di Debian, DDGS gagal dan aplikasi berhasil menyelesaikan analisis dengan deskripsi cadangan.
- Pengujian pada URL produksi masih perlu dilakukan setelah deploy. [Langkah verifikasi](#verify-the-deployed-url)

## Pemecahan masalah

| Gejala | Yang perlu diperiksa |
| --- | --- |
| Layanan analisis belum dikonfigurasi | Pastikan `API_KEY` terisi di `.env` atau Streamlit Secrets, lalu mulai ulang aplikasi jika mengubah konfigurasi lokal. |
| SumoPod mengembalikan HTTP 401 atau 403 | Periksa API key dan izin aksesnya. |
| SumoPod mengembalikan HTTP 404 | Periksa `MODEL` dan `BASE_URL`. |
| SumoPod mengembalikan HTTP 429 | Periksa kuota dan coba lagi nanti. |
| Pencarian web gagal | Isi deskripsi cadangan atau gunakan mode **Deskripsi lowongan**. |
| Teks CV tidak terbaca | Gunakan PDF dengan teks yang bisa disalin atau file TXT UTF-8. |
| Next.js gagal menghubungi API | Buka [endpoint health API](http://127.0.0.1:8000/health) dan periksa `CV_ANALYZER_API_URL`. |

## Lisensi

MIT. [Baca lisensi](LICENSE).
