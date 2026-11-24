"use client";

import dynamic from "next/dynamic";
import { BookOpen, ChartNoAxesColumnIncreasing, Check, CircleAlert, Copy, Download, FileText, ListChecks, LoaderCircle, SearchCheck } from "lucide-react";
import { useState } from "react";
import { type AnalysisReport, type Stage, reportFilename, STAGES } from "@/lib/analysis";

const ReportBody = dynamic(() => import("./report-body"), {
  loading: () => <p className="report-rendering" role="status">Menampilkan laporan…</p>,
});

interface ReportPanelProps {
  report: AnalysisReport | null;
  phase: Stage | null;
  running: boolean;
  role: string;
  showExample: () => void;
}

const OUTCOMES = [
  { icon: Check, tone: "mint", title: "Kekuatan CV", text: "Skill yang sudah relevan dengan posisi tujuan." },
  { icon: ChartNoAxesColumnIncreasing, tone: "amber", title: "Kesenjangan skill", text: "Persyaratan yang belum terlihat di CV." },
  { icon: ListChecks, tone: "violet", title: "Rencana belajar", text: "Langkah praktis untuk mengembangkan skill." },
];

export default function ReportPanel({ report, phase, running, role, showExample }: ReportPanelProps) {
  const [copyFeedback, setCopyFeedback] = useState<{ report: AnalysisReport; state: "done" | "failed" }>();
  const copyState = copyFeedback?.report === report ? copyFeedback?.state : "idle";
  const stageIndex = STAGES.findIndex(({ key }) => key === phase);

  async function copyReport() {
    if (!report) return;
    try {
      await navigator.clipboard.writeText(report.markdown);
      setCopyFeedback({ report, state: "done" });
    } catch {
      setCopyFeedback({ report, state: "failed" });
    }
  }

  function downloadReport() {
    if (!report) return;
    const url = URL.createObjectURL(new Blob([report.markdown], { type: "text/markdown;charset=utf-8" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = reportFilename(report);
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  return (
    <section className="panel report-panel" id="hasil" aria-labelledby="report-title" tabIndex={-1}>
      <div className="panel-heading report-heading">
        <span className="panel-icon"><FileText size={20} aria-hidden="true" /></span>
        <div>
          <h2 id="report-title">Hasil analisis</h2>
          <p>{report ? report.role : "Ringkasan dan langkah pengembangan skill."}</p>
        </div>
        {report && <span className={`report-status ${report.example ? "example" : "ready"}`}>
          {report.example ? <BookOpen size={13} aria-hidden="true" /> : <Check size={13} aria-hidden="true" />}
          {report.example ? "Contoh" : "Siap"}
        </span>}
      </div>

      {running ? (
        <div className="report-progress">
          <div className="progress-intro" role="status" aria-live="polite">
            <LoaderCircle className="spin" size={23} aria-hidden="true" />
            <div>
              <h3>{stageIndex < 0 ? "Menyiapkan analisis" : STAGES[stageIndex].title}</h3>
              <p>Posisi tujuan {role}</p>
            </div>
          </div>
          <ol className="stage-list" aria-label="Tahapan analisis">
            {STAGES.map(({ key, title }, index) => (
              <li key={key} data-status={index < stageIndex ? "done" : index === stageIndex ? "active" : "waiting"}
                aria-current={index === stageIndex ? "step" : undefined}>
                <span className="stage-number" aria-hidden="true">{index < stageIndex ? <Check size={14} /> : index + 1}</span>
                <span>{title}</span>
              </li>
            ))}
          </ol>
          <p className="progress-note">Biasanya perlu beberapa puluh detik. Kamu bisa tetap di halaman ini sampai laporan siap.</p>
        </div>
      ) : report ? (
        <>
          <div className="report-toolbar">
            <p>{report.source === "ddgs" ? "Pencarian web" : "Deskripsi lowongan"}
              <span aria-hidden="true"> · </span>{report.language === "english" ? "English" : "Indonesia"}</p>
            <div className="report-actions">
              <button type="button" className="button subtle copy-button" onClick={copyReport} aria-label="Salin laporan">
                {copyState === "done" ? <Check size={16} aria-hidden="true" /> : <Copy size={16} aria-hidden="true" />}
                <span>{copyState === "done" ? "Disalin" : "Salin"}</span>
              </button>
              <button type="button" className="button secondary" onClick={downloadReport}>
                <Download size={16} aria-hidden="true" />Unduh .md
              </button>
            </div>
          </div>
          {report.usedFallback && <p className="fallback-notice" role="status">Pencarian web gagal. Laporan memakai deskripsi lowongan cadangan.</p>}
          <span className="sr-only" role="status">{copyState === "done" && "Laporan disalin."}</span>
          {copyState === "failed" && <p className="field-error" role="alert"><CircleAlert size={14} />Salin tidak tersedia. Gunakan Unduh .md.</p>}
          <ReportBody report={report} />
        </>
      ) : (
        <>
          <div className="report-empty">
            <div className="empty-icon"><SearchCheck size={30} strokeWidth={1.65} aria-hidden="true" /></div>
            <h3>Belum ada laporan</h3>
            <p>Tambahkan CV dan posisi tujuan untuk memulai analisis.</p>
            <button type="button" className="button text-button" onClick={showExample}>
              <BookOpen size={16} aria-hidden="true" />Lihat contoh laporan
            </button>
          </div>
          <ul className="outcome-list">
            {OUTCOMES.map(({ icon: Icon, tone, title, text }) => (
              <li key={title}>
                <span className={`outcome-icon ${tone}`}><Icon size={18} strokeWidth={2} aria-hidden="true" /></span>
                <div><h3>{title}</h3><p>{text}</p></div>
              </li>
            ))}
          </ul>
        </>
      )}
    </section>
  );
}
