"use client";

import { ArrowRight, BookOpen, BriefcaseBusiness, CircleAlert, CloudUpload, FileText, Globe2, Languages, LoaderCircle, ShieldCheck, X } from "lucide-react";
import Image from "next/image";
import Link from "next/link";
import { useEffect, useRef, useState, type DragEvent, type FormEvent, type ReactNode } from "react";
import {
  AnalysisError, ERROR_MESSAGES, MAX_CV_BYTES, requestAnalysis,
  type AnalysisReport, type Language, type Source, type Stage,
} from "@/lib/analysis";
import { EXAMPLE_REPORT } from "@/lib/example-report";
import brandMark from "@/public/cv-analyzer-logo.png";
import ReportPanel from "./report-panel";

type Field = "cv" | "role" | "description";
type FieldErrors = Partial<Record<Field, string>>;

function focusField(id: string) {
  requestAnimationFrame(() => document.getElementById(id)?.focus());
}

function FieldError({ id, message }: { id: string; message?: string }) {
  if (!message) return null;
  return <p className="field-error" id={id} role="alert"><CircleAlert size={14} aria-hidden="true" />{message}</p>;
}

export default function Workspace({ heroArt }: { heroArt: ReactNode }) {
  const [file, setFile] = useState<File | null>(null);
  const [demo, setDemo] = useState(false);
  const [role, setRole] = useState("");
  const [language, setLanguage] = useState<Language>("indonesia");
  const [source, setSource] = useState<Source>("ddgs");
  const [description, setDescription] = useState("");
  const [showFallback, setShowFallback] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [errors, setErrors] = useState<FieldErrors>({});
  const [requestError, setRequestError] = useState<string | null>(null);
  const [phase, setPhase] = useState<Stage | null>(null);
  const [running, setRunning] = useState(false);
  const [report, setReport] = useState<AnalysisReport | null>(null);
  const upload = useRef<HTMLInputElement>(null);
  const activeRequest = useRef<AbortController | null>(null);

  useEffect(() => () => activeRequest.current?.abort(), []);

  function clearFieldError(field: Field) {
    setErrors((current) => ({ ...current, [field]: undefined }));
  }

  function chooseFile(picked: File) {
    setRequestError(null);
    const errorCode = !/\.(pdf|txt)$/i.test(picked.name) ? "unsupported_file"
      : picked.size > MAX_CV_BYTES ? "file_too_large" : picked.size === 0 ? "invalid_cv" : null;
    setFile(errorCode ? null : picked);
    setDemo(false);
    setErrors((current) => ({ ...current, cv: errorCode ? ERROR_MESSAGES[errorCode] : undefined }));
  }

  function dropFile(event: DragEvent<HTMLButtonElement>) {
    event.preventDefault();
    setDragging(false);
    const dropped = event.dataTransfer.files[0];
    if (dropped && !running) chooseFile(dropped);
  }

  function showReport() {
    requestAnimationFrame(() => document.getElementById("hasil")?.focus());
  }

  function showExample() {
    if (activeRequest.current) return;
    setReport(EXAMPLE_REPORT);
    showReport();
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (activeRequest.current) return;
    const invalid: FieldErrors = {};
    if (!file && !demo) invalid.cv = errors.cv || ERROR_MESSAGES.missing_cv;
    if (!role.trim()) invalid.role = "Isi posisi tujuan sebelum menganalisis.";
    if (source === "job-description" && !description.trim()) invalid.description = "Tempel deskripsi lowongan yang ingin dibandingkan.";
    setErrors(invalid);
    setRequestError(null);
    if (invalid.cv || invalid.role || invalid.description) {
      focusField(invalid.cv ? "choose-cv" : invalid.role ? "target-role" : "job-description");
      return;
    }

    const form = new FormData();
    form.set("role", role.trim());
    form.set("language", language);
    form.set("source", source);
    form.set("description", description.trim());
    if (file) form.set("cv", file);
    else form.set("demo", "true");

    const controller = new AbortController();
    activeRequest.current = controller;
    setRunning(true);
    setPhase(null);
    setReport(null);
    if (window.matchMedia("(max-width: 840px)").matches) showReport();

    try {
      setReport(await requestAnalysis(form, setPhase, controller.signal));
    } catch (error) {
      if (!controller.signal.aborted) {
        setRequestError(error instanceof AnalysisError ? error.code : "backend_unavailable");
        focusField("analysis-error");
      }
    } finally {
      activeRequest.current = null;
      setRunning(false);
    }
  }

  const hasCv = !!file || demo;
  const showDescription = source === "job-description" || showFallback;

  return (
    <>
      <a href="#analisis" className="skip-link">Langsung ke analisis</a>
      <header className="site-header">
        <div className="header-inner">
          <Link className="wordmark" href="/" aria-label="CV Analyzer, halaman utama">
            <Image src={brandMark} alt="" width={40} height={40} loading="eager" className="brand-icon" />
            <span>CV Analyzer</span>
          </Link>
          <button type="button" className="button secondary header-example" onClick={showExample} disabled={running}>
            <BookOpen size={17} aria-hidden="true" />Contoh laporan
          </button>
        </div>
      </header>

      <main className="shell" id="analisis">
        <section className="hero" aria-labelledby="page-title">
          <div className="hero-content">
            <h1 id="page-title">Seberapa cocok CV-mu<br className="desktop-break" /> dengan posisi tujuan?</h1>
            <p>Kenali kekuatan, temukan skill yang masih dibutuhkan, dan susun langkah belajar berikutnya.</p>
          </div>
          <div className="hero-visual" aria-hidden="true">{heroArt}</div>
        </section>

        <div className="workspace-grid">
          <section className="panel form-panel" aria-labelledby="form-title">
            <div className="panel-heading">
              <span className="panel-icon"><FileText size={20} aria-hidden="true" /></span>
              <div>
                <h2 id="form-title">Siapkan analisis</h2>
                <p>Tambahkan CV dan pilih persyaratan yang ingin dibandingkan.</p>
              </div>
            </div>

            <form onSubmit={submit} noValidate>
              <fieldset className="analysis-fields" disabled={running}>
                <div className="field">
                  <div className="field-top">
                    <p className="field-label" id="cv-label">CV kandidat</p>
                    <button type="button" className="button link-button" onClick={() => {
                      setFile(null);
                      setDemo(true);
                      if (!role.trim()) setRole("Senior AI Engineer");
                      clearFieldError("cv");
                      clearFieldError("role");
                      setRequestError(null);
                    }}>Gunakan CV contoh</button>
                  </div>
                  <input ref={upload} type="file" accept=".pdf,.txt" className="sr-only" tabIndex={-1}
                    aria-label="Pilih file CV" onChange={(event) => {
                      const picked = event.target.files?.[0];
                      if (picked) chooseFile(picked);
                      event.target.value = "";
                    }} />
                  {hasCv ? (
                    <div className="selected-file" aria-label="CV terpilih">
                      <span className="upload-icon"><FileText size={20} aria-hidden="true" /></span>
                      <div className="file-meta">
                        <strong>{file?.name || "CV contoh"}</strong>
                        <small>{file ? `${Math.max(1, Math.ceil(file.size / 1024)).toLocaleString("id-ID")} KB` : "File TXT bawaan"}</small>
                      </div>
                      <button type="button" className="button icon-button" aria-label="Hapus CV" onClick={() => {
                        setFile(null);
                        setDemo(false);
                        focusField("choose-cv");
                      }}><X size={18} aria-hidden="true" /></button>
                    </div>
                  ) : (
                    <button id="choose-cv" type="button" className="dropzone" data-dragging={dragging}
                      aria-labelledby="choose-cv-label cv-label" data-invalid={!!errors.cv}
                      aria-describedby={errors.cv ? "cv-guidance cv-error" : "cv-guidance"}
                      onClick={() => upload.current?.click()}
                      onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
                      onDragLeave={(event) => {
                        if (!(event.relatedTarget instanceof Node) || !event.currentTarget.contains(event.relatedTarget)) setDragging(false);
                      }}
                      onDrop={dropFile}>
                      <span className="upload-icon"><CloudUpload size={22} aria-hidden="true" /></span>
                      <span className="dropzone-copy">
                        <strong id="choose-cv-label">Pilih file <span>atau seret ke sini</span></strong>
                        <small id="cv-guidance">PDF atau TXT · Maks. 5 MB</small>
                      </span>
                    </button>
                  )}
                  <FieldError id="cv-error" message={errors.cv} />
                </div>

                <div className="field">
                  <label className="field-label" htmlFor="target-role">Posisi tujuan</label>
                  <div className="input-shell">
                    <BriefcaseBusiness size={17} aria-hidden="true" />
                    <input id="target-role" name="role" autoComplete="off" placeholder="Mis. Senior AI Engineer" required maxLength={150}
                      value={role} onChange={(event) => { setRole(event.target.value); clearFieldError("role"); }}
                      aria-invalid={!!errors.role} aria-describedby={errors.role ? "role-error" : undefined} />
                  </div>
                  <FieldError id="role-error" message={errors.role} />
                </div>

                <fieldset className="field source-field">
                  <legend className="field-label">Sumber persyaratan</legend>
                  <div className="source-options">
                    {[
                      { value: "ddgs", label: "Pencarian web", detail: "Kebutuhan umum untuk posisi", icon: Globe2 },
                      { value: "job-description", label: "Deskripsi lowongan", detail: "Syarat dari satu lowongan", icon: FileText },
                    ].map(({ value, label, detail, icon: Icon }) => (
                      <label key={value} className="source-option" data-active={value === source}>
                        <input type="radio" name="source" value={value} checked={value === source}
                          onChange={() => { setSource(value as Source); clearFieldError("description"); setRequestError(null); }} />
                        <Icon size={18} aria-hidden="true" />
                        <span><strong>{label}</strong><small>{detail}</small></span>
                      </label>
                    ))}
                  </div>
                </fieldset>

                {source === "ddgs" && <button type="button" className="button link-button fallback-button"
                  onClick={() => setShowFallback((current) => !current)} aria-expanded={showFallback} aria-controls="description-field">
                  {showFallback ? "Sembunyikan deskripsi cadangan" : description.trim() ? "Lihat deskripsi cadangan" : "Tambahkan deskripsi cadangan"}
                </button>}
                <div id="description-field" className="field" hidden={!showDescription}>
                  <label className="field-label" htmlFor="job-description">{source === "job-description" ? "Deskripsi lowongan" : "Deskripsi cadangan"}</label>
                  {source === "ddgs" && <p id="description-hint" className="field-hint">Dipakai jika pencarian web tidak menghasilkan data.</p>}
                  <textarea id="job-description" name="description" rows={4} maxLength={15000} placeholder="Tempel persyaratan dari lowongan yang dipilih…"
                    value={description} onChange={(event) => { setDescription(event.target.value); clearFieldError("description"); }}
                    required={source === "job-description"} aria-invalid={!!errors.description}
                    aria-describedby={errors.description ? "description-error" : source === "ddgs" ? "description-hint" : undefined} />
                  <FieldError id="description-error" message={errors.description} />
                </div>

                <div className="form-actions">
                  <div className="language-field">
                    <label className="field-label" htmlFor="report-language">Bahasa laporan</label>
                    <div className="select-shell">
                      <Languages size={17} aria-hidden="true" />
                      <select id="report-language" name="language" value={language} onChange={(event) => setLanguage(event.target.value as Language)}>
                        <option value="indonesia">Indonesia</option>
                        <option value="english">English</option>
                      </select>
                    </div>
                  </div>
                  <button type="submit" className="button primary" disabled={running}>
                    {running ? <><LoaderCircle size={17} className="spin" aria-hidden="true" />Menganalisis…</>
                      : <>Analisis CV<ArrowRight size={18} aria-hidden="true" /></>}
                  </button>
                </div>
              </fieldset>

              {requestError && <div className="analysis-error" id="analysis-error" tabIndex={-1} role="alert">
                <CircleAlert size={18} aria-hidden="true" />
                <div><p>{ERROR_MESSAGES[requestError] || ERROR_MESSAGES.analysis_failed}</p>
                  {requestError === "search_unavailable" && <button type="button" className="button link-button" onClick={() => {
                    setSource("job-description");
                    setRequestError(null);
                    focusField("job-description");
                  }}>Pakai deskripsi lowongan</button>}
                </div>
              </div>}

              <p className="privacy-note"><ShieldCheck size={15} aria-hidden="true" />Teks CV dikirim ke SumoPod. File sementara dihapus setelah analisis.</p>
            </form>
          </section>

          <ReportPanel report={report} phase={phase} running={running} role={role} showExample={showExample} />
        </div>

        <footer className="site-footer">
          <span>CV Analyzer</span>
          <p>Dibuat oleh Naufal Firdaus</p>
        </footer>
      </main>
    </>
  );
}
