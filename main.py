from __future__ import annotations
import argparse
from pathlib import Path
from dotenv import load_dotenv
import sys
from pathlib import Path as _P
BASE_DIR = _P(__file__).parent.resolve()
sys.path.insert(0, str(BASE_DIR))
from src.state import PipelineState
from src.graph.workflow import build_graph
from src.llm_provider import normalize_provider

def main():
    load_dotenv()  # load .env if exists

    parser = argparse.ArgumentParser(description="AI Multi-Agent CV Analyzer (Gemini/Mistral)")
    parser.add_argument("--cv", required=True, help="Path to CV file (.txt or .pdf)")
    parser.add_argument("--role", required=True, help="Target role, e.g. 'Senior AI Engineer'")
    parser.add_argument("--out", default="report.md", help="Output markdown path")
    parser.add_argument("--provider", default="auto", choices=["auto","gemini","mistral"], help="LLM provider selection")
    parser.add_argument("--language", default="indonesia", choices=["english","indonesia"], help="Report language")
    args = parser.parse_args()

    state = PipelineState(
        cv_path=args.cv, 
        target_role=args.role, 
        language=args.language,
        provider=normalize_provider(args.provider)
    )
    run = build_graph()
    final = run(state)
    # LangGraph app.invoke may return a plain dict; coerce into PipelineState for uniform handling
    if isinstance(final, dict):
        final = PipelineState.model_validate(final)

    if final.errors:
        print("[WARN] Pipeline completed with errors:")
        for e in final.errors:
            print(" -", e)

    if final.report_markdown:
        out_path = Path(args.out)
        out_path.write_text(final.report_markdown, encoding="utf-8")
        print(f"[OK] Report written to: {out_path.resolve()}")
    else:
        print("[ERR] No report produced.")

if __name__ == "__main__":
    main()
