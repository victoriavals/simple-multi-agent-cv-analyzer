"""HTTP adapter for the Next.js workspace and the existing analysis pipeline."""

from __future__ import annotations

from collections.abc import Iterator
import json
from pathlib import Path
from queue import Queue
from tempfile import TemporaryDirectory
from threading import BoundedSemaphore, Thread
from typing import Annotated, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from .config import get_setting
from .graph.workflow import build_graph
from .state import PipelineState


ROOT = Path(__file__).resolve().parent.parent
MAX_CV_BYTES = 5 * 1024 * 1024
ANALYSIS_SLOTS = BoundedSemaphore(2)
load_dotenv(ROOT / ".env")

app = FastAPI(title="CV Analyzer API")


def api_error(code: str, status: int) -> JSONResponse:
    return JSONResponse({"type": "error", "code": code}, status_code=status, headers={"Cache-Control": "no-store"})


@app.exception_handler(RequestValidationError)
async def invalid_form(_request, _exception):
    return api_error("invalid_form", 422)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def analysis_events(state: PipelineState, content: bytes | None, suffix: str) -> Iterator[str]:
    if not ANALYSIS_SLOTS.acquire(blocking=False):
        yield '{"type":"error","code":"busy"}\n'
        return

    events: Queue[dict | None] = Queue()

    def emit_progress(stage: str) -> None:
        events.put({"type": "progress", "stage": stage})

    def run() -> None:
        try:
            with TemporaryDirectory(prefix="cv-analyzer-") as directory:
                if content is not None:
                    path = Path(directory) / f"cv{suffix}"
                    path.write_bytes(content)
                    state.cv_path = str(path)
                final = PipelineState.model_validate(build_graph(on_progress=emit_progress)(state))

            if final.errors or not final.report_markdown:
                details = " ".join(final.errors).casefold()
                code = "analysis_failed"
                if "load cv error" in details:
                    code = "invalid_cv"
                elif "market intel error" in details and "web search" in details:
                    code = "search_unavailable"
                elif "sumopod" in details:
                    code = "provider_unavailable"
                events.put({"type": "error", "code": code})
                return

            market = final.market_requirements or {}
            events.put({
                "type": "complete",
                "report": {
                    "markdown": final.report_markdown,
                    "role": final.target_role,
                    "language": final.language,
                    "source": market.get("source", final.market_source),
                    "usedFallback": final.market_source == "ddgs" and market.get("source") == "job_description",
                },
            })
        except Exception:
            events.put({"type": "error", "code": "analysis_failed"})
        finally:
            ANALYSIS_SLOTS.release()
            events.put(None)

    try:
        Thread(target=run, daemon=True, name="cv-analysis").start()
    except RuntimeError:
        ANALYSIS_SLOTS.release()
        yield '{"type":"error","code":"analysis_failed"}\n'
        return

    for event in iter(events.get, None):
        yield json.dumps(event, ensure_ascii=False) + "\n"


@app.post("/analyze")
async def analyze(
    role: Annotated[str, Form(max_length=150)],
    language: Annotated[Literal["indonesia", "english"], Form()] = "indonesia",
    source: Annotated[Literal["ddgs", "job-description"], Form()] = "ddgs",
    description: Annotated[str, Form(max_length=15000)] = "",
    demo: Annotated[bool, Form()] = False,
    cv: Annotated[UploadFile | None, File()] = None,
):
    content = None
    suffix = ""
    try:
        if not role.strip() or (source == "job-description" and not description.strip()):
            return api_error("invalid_form", 422)
        if cv is not None:
            suffix = Path(cv.filename or "").suffix.lower()
            if suffix not in {".pdf", ".txt"}:
                return api_error("unsupported_file", 422)
            content = await cv.read(MAX_CV_BYTES + 1)
            if len(content) > MAX_CV_BYTES:
                return api_error("file_too_large", 413)
            if not content:
                return api_error("invalid_cv", 422)
        elif not demo:
            return api_error("missing_cv", 422)
        if not get_setting("API_KEY"):
            return api_error("provider_not_configured", 503)
    finally:
        if cv is not None:
            await cv.close()

    state = PipelineState(
        cv_path=str(ROOT / "samples" / "sample_cv.txt"),
        target_role=role.strip(),
        language=language,
        market_source=source,
        job_description=description.strip() or None,
    )
    return StreamingResponse(
        analysis_events(state, content, suffix),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )
