import type { AnalysisReport } from "./analysis";

export const EXAMPLE_REPORT: AnalysisReport = {
  role: "Data Platform Engineer",
  language: "indonesia",
  source: "job_description",
  example: true,
  markdown: `## Ikhtisar
CV contoh menunjukkan pengalaman membangun API dengan Python, menggunakan SQL, dan menerapkan aplikasi dengan Docker. Untuk posisi Data Platform Engineer, langkah berikutnya adalah mengembangkan pengalaman dengan Kubernetes dan pemantauan sistem.

## Kekuatan

| Skill | Catatan |
| --- | --- |
| Python | Digunakan untuk membangun API dan mengolah data. |
| SQL | Mendukung kueri dan validasi data pada basis data relasional. |
| Docker | Menunjukkan pengalaman mengemas aplikasi untuk deployment. |

## Kesenjangan Keahlian

| Skill | Catatan |
| --- | --- |
| Kubernetes | CV belum menyebutkan pengalaman mengelola aplikasi di klaster. |
| Pemantauan sistem | Belum ada contoh pengukuran kesehatan aplikasi di produksi. |

## Rencana Upskilling

### Minggu 1 — Dasar Kubernetes

- Jalankan API Python dalam klaster lokal.
- Tambahkan konfigurasi Deployment dan Service.
- Uji pemulihan ketika satu replika berhenti.

### Minggu 2 — Pemantauan aplikasi

- Tambahkan endpoint kesehatan dan metrik latensi.
- Buat dasbor untuk melihat kegagalan dan penggunaan sumber daya.
- Dokumentasikan hasil percobaan dalam portofolio.

## Catatan Akhir
Perkuat CV dengan bukti yang bisa diperiksa, seperti repositori proyek dan catatan hasil pengujian.

Basis kebutuhan skill: deskripsi lowongan contoh.
`,
};
