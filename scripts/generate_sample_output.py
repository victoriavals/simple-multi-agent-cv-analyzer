#!/usr/bin/env python3
"""Generate the sample report using SumoPod and DDGS, with a sample job fallback."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from src.graph.workflow import build_graph
from src.state import PipelineState


def main() -> int:
    load_dotenv(ROOT / ".env")
    try:
        final = PipelineState.model_validate(build_graph()(PipelineState(
            cv_path=str(ROOT / "samples/sample_cv.txt"),
            target_role="Senior AI Engineer", language="indonesia",
            job_description=(ROOT / "samples/sample_job_description.txt").read_text(encoding="utf-8"),
        )))
        for warning in final.warnings:
            print(f"[WARN] {warning}")
        for error in final.errors:
            print(f"[ERR] {error}")
        if final.errors or not final.report_markdown:
            print("[ERR] No sample report generated.")
            return 1
        output = ROOT / "samples/sample_output.md"
        output.write_text(final.report_markdown, encoding="utf-8")
        print(f"[OK] Sample report written to {output}")
        return 0
    except Exception as exc:
        print(f"[ERR] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
