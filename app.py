from __future__ import annotations

from collections.abc import Callable
import logging
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from threading import BoundedSemaphore
import unicodedata

from dotenv import load_dotenv
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from src.agents.report_agent import validate_markdown
from src.config import get_setting
from src.graph.workflow import build_graph
from src.state import PipelineState


BASE_DIR = Path(__file__).resolve().parent
MAX_UPLOAD_MB = 5
LANGUAGES = {"indonesia": "Indonesia", "english": "English"}
SOURCES = {"ddgs": "Pencarian web", "job-description": "Deskripsi lowongan"}
STAGES = {
    "reading": "Membaca CV",
    "skills": "Mengidentifikasi skill",
    "requirements": "Meninjau persyaratan posisi",
    "report": "Menyusun laporan",
}
logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner=False)
def analysis_slots() -> BoundedSemaphore:
    return BoundedSemaphore(2)


def slugify(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^a-zA-Z0-9-]+", "-", value).strip("-")
    return re.sub(r"-+", "-", value).lower() or "report"


def run_analysis(
    state: PipelineState, uploaded: UploadedFile | None, on_progress: Callable[[str], None],
) -> PipelineState:
    run = build_graph(on_progress=on_progress)
    if uploaded is None:
        return PipelineState.model_validate(run(state))
    with TemporaryDirectory(prefix="cv-analyzer-") as directory:
        path = Path(directory) / ("cv" + Path(uploaded.name).suffix.lower())
        path.write_bytes(uploaded.getbuffer())
        state.cv_path = str(path)
        return PipelineState.model_validate(run(state))


def show_analysis_error(errors: list[str]) -> None:
    details = " ".join(errors).casefold()
    if "load cv error:" in details:
        message = "Teks CV tidak terbaca. Gunakan PDF yang teksnya bisa disalin atau TXT UTF-8."
    elif "market intel error:" in details and "web search" in details:
        message = "Pencarian web tidak tersedia. Tempel deskripsi lowongan untuk melanjutkan."
    elif "sumopod" in details:
        status = re.search(r"sumopod returned http (\d{3})", details)
        message = "SumoPod sedang gagal memproses analisis. Coba lagi atau hubungi pengelola aplikasi."
        if status:
            message = f"SumoPod mengembalikan HTTP {status[1]}. Hubungi pengelola aplikasi jika masih terjadi."
    else:
        message = "Laporan belum bisa dibuat. Periksa masukanmu dan coba lagi."
    st.error(message)


def submit_analysis(state: PipelineState, uploaded: UploadedFile | None, demo: bool) -> None:
    st.session_state.pop("analysis", None)
    if not state.target_role:
        st.warning("Isi posisi tujuan sebelum menganalisis.")
        return
    if state.market_source == "job-description" and not state.job_description:
        st.warning("Tempel deskripsi lowongan yang ingin dibandingkan.")
        return
    if uploaded is None and not demo:
        st.warning("Unggah file CV atau gunakan CV contoh.")
        return
    if uploaded is not None and (
        Path(uploaded.name).suffix.lower() not in {".pdf", ".txt"}
        or not 0 < uploaded.size <= MAX_UPLOAD_MB * 1024 * 1024
    ):
        st.warning("Pilih file PDF atau TXT yang berisi teks dan berukuran maksimal 5 MB.")
        return
    if not get_setting("API_KEY"):
        st.error("Layanan analisis belum dikonfigurasi. Pengelola aplikasi perlu mengisi API_KEY di Streamlit Secrets.")
        return

    status_area = st.empty()
    slots = analysis_slots()
    if not slots.acquire(blocking=False):
        st.warning("Layanan sedang memproses analisis lain. Coba lagi sebentar.")
        return
    try:
        status = status_area.status("Menyiapkan analisis…")
        final = run_analysis(state, uploaded, lambda stage: status.update(label=STAGES[stage]))
    except Exception as exc:
        logger.warning("Streamlit analysis failed (%s)", type(exc).__name__)
        show_analysis_error([])
        return
    finally:
        slots.release()
        status_area.empty()

    if final.errors or not final.report_markdown:
        show_analysis_error(final.errors)
        return
    source = (final.market_requirements or {}).get("source")
    st.session_state["analysis"] = {
        "markdown": final.report_markdown,
        "role": state.target_role,
        "language": state.language,
        "source": source,
        "used_fallback": state.market_source == "ddgs" and source == "job_description",
    }


def show_report(saved: dict) -> None:
    st.header("Hasil analisis", anchor=False)
    st.text(f"{saved['role']} · {LANGUAGES[saved['language']]}")
    source = "job-description" if saved["source"] == "job_description" else "ddgs"
    st.caption(f"Sumber persyaratan · {SOURCES[source]}")
    if saved["used_fallback"]:
        st.warning("Pencarian web gagal. Laporan memakai deskripsi lowongan cadangan.")
    st.download_button(
        "Unduh laporan .md",
        data=saved["markdown"],
        file_name=f"laporan-{slugify(saved['role'])}-{saved['language']}.md",
        mime="text/markdown",
        on_click="ignore",
    )
    report_tab, markdown_tab = st.tabs(["Laporan", "Markdown"])
    with report_tab:
        st.markdown(saved["markdown"])
        if validate_markdown(saved["markdown"], saved["language"]):
            st.info("Beberapa bagian laporan belum terisi. Periksa hasil sebelum menggunakannya.")
    with markdown_tab:
        st.code(saved["markdown"], language="markdown", wrap_lines=True)


def main() -> None:
    load_dotenv(BASE_DIR / ".env")
    st.set_page_config(page_title="CV Analyzer", page_icon=str(BASE_DIR / "web/app/icon.png"), layout="centered")
    with st.container(horizontal=True, vertical_alignment="center"):
        st.image(str(BASE_DIR / "web/public/cv-analyzer-logo.png"), width=40)
        st.markdown("**CV Analyzer**")
    st.title("Seberapa cocok CV-mu dengan posisi tujuan?", anchor=False)
    st.write("Kenali kekuatan CV, temukan skill yang dibutuhkan, dan susun langkah belajar berikutnya.")

    st.header("Siapkan analisis", anchor=False)
    with st.form("analysis_form"):
        uploaded = st.file_uploader("CV kandidat", type=["pdf", "txt"], max_upload_size=MAX_UPLOAD_MB)
        demo = st.checkbox("Gunakan CV contoh bila belum mengunggah file")
        role = st.text_input("Posisi tujuan", placeholder="Mis. Senior AI Engineer", max_chars=150).strip()
        language = st.selectbox("Bahasa laporan", LANGUAGES, format_func=LANGUAGES.__getitem__, key="report_language")
        source = st.selectbox("Sumber persyaratan", SOURCES, format_func=SOURCES.__getitem__, key="market_source")
        description = st.text_area(
            "Deskripsi lowongan",
            key="job_description_input",
            height=144,
            max_chars=15000,
            help="Wajib untuk mode Deskripsi lowongan. Bisa dipakai sebagai cadangan saat pencarian web gagal.",
        ).strip()
        st.caption("Teks CV dan deskripsi yang dipakai dikirim ke SumoPod. Salinan file untuk analisis dihapus setelah diproses.")
        run = st.form_submit_button("Analisis CV", type="primary")

    if run:
        submit_analysis(PipelineState(
            cv_path=str(BASE_DIR / "samples/sample_cv.txt"),
            target_role=role,
            language=language,
            market_source=source,
            job_description=description,
        ), uploaded, demo)
    saved = st.session_state.get("analysis")
    if saved:
        show_report(saved)
    else:
        with st.expander("Lihat contoh laporan"):
            st.markdown((BASE_DIR / "samples/sample_output.md").read_text(encoding="utf-8"))
    st.divider()
    st.caption("Dibuat oleh Naufal Firdaus untuk Krenovation Assessment Test")


if __name__ == "__main__":
    main()
