export type Language = "indonesia" | "english";
export type Source = "ddgs" | "job-description";
export type Stage = "reading" | "skills" | "requirements" | "report";

export interface AnalysisReport {
  markdown: string;
  role: string;
  language: Language;
  source: "ddgs" | "job_description";
  usedFallback?: boolean;
  example?: boolean;
}

type AnalysisEvent =
  | { type: "progress"; stage: Stage }
  | { type: "complete"; report: AnalysisReport }
  | { type: "error"; code: string };

export const STAGES: { key: Stage; title: string }[] = [
  { key: "reading", title: "Membaca CV" },
  { key: "skills", title: "Mengidentifikasi skill" },
  { key: "requirements", title: "Meninjau persyaratan posisi" },
  { key: "report", title: "Menyusun laporan" },
];

export const MAX_CV_BYTES = 5 * 1024 * 1024;

export const ERROR_MESSAGES: Record<string, string> = {
  invalid_form: "Periksa posisi tujuan dan deskripsi lowongan, lalu coba lagi.",
  missing_cv: "Tambahkan file CV atau gunakan CV contoh.",
  unsupported_file: "Pilih file PDF atau TXT.",
  file_too_large: "Ukuran file melebihi 5 MB. Pilih file yang lebih kecil.",
  invalid_cv: "Teks di file CV tidak bisa dibaca. Coba PDF yang teksnya bisa disalin atau file TXT UTF-8.",
  search_unavailable: "Pencarian web gagal mengambil persyaratan. Kamu bisa menempel deskripsi lowongan untuk melanjutkan.",
  backend_unavailable: "Layanan analisis belum bisa dihubungi. Coba lagi setelah layanan siap.",
  provider_not_configured: "Layanan AI belum dikonfigurasi. Hubungi pengelola aplikasi.",
  provider_unavailable: "Layanan AI sedang tidak merespons. Coba lagi beberapa saat lagi.",
  busy: "Layanan sedang memproses analisis lain. Coba lagi sebentar.",
  analysis_failed: "Laporan belum bisa dibuat. Periksa masukanmu dan coba lagi.",
};

export class AnalysisError extends Error {
  constructor(public readonly code: string) {
    super(ERROR_MESSAGES[code] || ERROR_MESSAGES.analysis_failed);
  }
}

export async function requestAnalysis(
  form: FormData,
  onProgress: (stage: Stage) => void,
  signal: AbortSignal,
): Promise<AnalysisReport> {
  const response = await fetch("/api/analyze", { method: "POST", body: form, signal });
  if (!response.ok) {
    const error = await response.json().catch(() => null);
    throw new AnalysisError(error?.code || "backend_unavailable");
  }
  if (!response.body) throw new AnalysisError("analysis_failed");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let pending = "";

  try {
    while (true) {
      const { value, done } = await reader.read();
      pending += decoder.decode(value, { stream: !done });
      const lines = pending.split("\n");
      pending = done ? "" : lines.pop() || "";
      for (const line of lines) {
        if (!line.trim()) continue;
        const event: AnalysisEvent = JSON.parse(line);
        if (event.type === "complete") return event.report;
        if (event.type === "error") throw new AnalysisError(event.code);
        if (event.type === "progress") onProgress(event.stage);
      }
      if (done) break;
    }
    throw new AnalysisError("analysis_failed");
  } finally {
    await reader.cancel().catch(() => undefined);
    reader.releaseLock();
  }
}

export function reportFilename(report: AnalysisReport) {
  const slug = report.role.normalize("NFKD").replace(/[\u0300-\u036f]/g, "")
    .toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
  return `laporan-${slug || "cv"}-${report.language}.md`;
}
