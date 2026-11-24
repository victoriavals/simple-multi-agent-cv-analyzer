from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from src.graph.workflow import build_graph
from src.state import PipelineState


def main() -> int:
    load_dotenv(Path(__file__).parent / ".env")
    parser = argparse.ArgumentParser(description="AI CV Analyzer (SumoPod + DDGS)")
    parser.add_argument("--cv", required=True, help="Path to CV file (.txt or .pdf)")
    parser.add_argument("--role", required=True, help="Target role, e.g. 'Senior AI Engineer'")
    parser.add_argument("--out", default="report.md", help="Output markdown path")
    parser.add_argument("--provider", default="sumopod", choices=["sumopod", "auto"], help="LLM provider (auto is an alias for sumopod)")
    parser.add_argument("--language", default="indonesia", choices=["english", "indonesia"], help="Report language")
    parser.add_argument("--market-source", default="ddgs", choices=["ddgs", "job-description"], help="Search the web or use a supplied job description")
    parser.add_argument("--job-description", type=Path, help="UTF-8 text file: primary source in job-description mode, fallback in ddgs mode")
    args = parser.parse_args()
    if args.market_source == "job-description" and args.job_description is None:
        parser.error("--job-description is required with --market-source job-description")
    try:
        job_description = args.job_description.read_text(encoding="utf-8") if args.job_description else None
        final = PipelineState.model_validate(build_graph()(PipelineState(
            cv_path=args.cv, target_role=args.role, language=args.language,
            provider=args.provider, market_source=args.market_source, job_description=job_description,
        )))
        for warning in final.warnings:
            print(f"[WARN] {warning}")
        if final.errors or not final.report_markdown:
            for error in final.errors:
                print(f"[ERR] {error}")
            print("[ERR] No report produced.")
            return 1
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(final.report_markdown, encoding="utf-8")
        print(f"[OK] Report written to: {out_path.resolve()}")
        return 0
    except Exception as exc:
        print(f"[ERR] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
