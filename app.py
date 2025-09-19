from __future__ import annotations
import io
import os
from pathlib import Path
import sys
import tempfile
import unicodedata
import re
import streamlit as st
BASE_DIR = Path(__file__).parent.resolve()
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.state import PipelineState
from src.graph.workflow import build_graph
from src.llm_provider import normalize_provider
from src.agents.report_agent import validate_markdown
from dotenv import load_dotenv


def slugify(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^a-zA-Z0-9-]+", "-", value).strip("-")
    value = re.sub(r"-+", "-", value)
    return value.lower() or "report"


def main():
    load_dotenv()
    st.set_page_config(page_title="AI CV Analyzer", page_icon="📄", layout="centered")
    st.title("AI CV Analyzer")
    st.caption("Analyze a candidate CV against a target role and generate a concise Markdown report.")

    with st.sidebar:
        role = st.text_input("Target role", placeholder="e.g., Senior AI Engineer")
        language = st.selectbox("Language", options=["english", "indonesia"], index=1)
        provider_label = st.selectbox("LLM Provider", options=["Auto", "Gemini", "Mistral"], index=0)
        uploaded = st.file_uploader("Upload CV (.pdf or .txt)", type=["pdf", "txt"], accept_multiple_files=False)
        demo = st.checkbox("Demo mode (use sample CV if no file uploaded)")
        run = st.button("Run analysis")

    if run:
        # Validate API keys early for crisp UX
        secrets = {}
        try:
            if hasattr(st, "secrets"):
                secrets = dict(st.secrets)
        except Exception:
            secrets = {}
        tavily_key = secrets.get("TAVILY_API_KEY") or os.getenv("TAVILY_API_KEY")
        # In provider selection, only require keys for chosen provider
        prov_code = normalize_provider(provider_label)
        if not tavily_key:
            st.error("Missing TAVILY_API_KEY. Add it to .env or Streamlit secrets.")
            return
        # Optional checks for other providers to improve UX
        if prov_code == "gemini":
            gkey = secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not gkey:
                st.error("Missing GEMINI_API_KEY (or GOOGLE_API_KEY). Add it to .env or Streamlit secrets.")
                return
        if prov_code == "mistral":
            mkey = secrets.get("MISTRAL_API_KEY") or os.getenv("MISTRAL_API_KEY")
            if not mkey:
                st.error("Missing MISTRAL_API_KEY. Add it to .env or Streamlit secrets.")
                return

        if not role:
            st.warning("Please enter a target role.")
            return
        if not uploaded and not demo:
            st.warning("Please upload a CV file (.pdf or .txt), or enable 'Demo mode'.")
            return

        # Persist to a temp file with the correct suffix
        tmp_path = None
        if uploaded:
            suffix = "." + uploaded.name.split(".")[-1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = tmp.name
        else:
            # Use sample
            sample_path = Path(__file__).parent / "samples" / "sample_cv.txt"
            if not sample_path.exists():
                st.error("Sample CV not found at samples/sample_cv.txt.")
                return
            tmp_path = str(sample_path)

        try:
            state = PipelineState(cv_path=tmp_path, target_role=role, language=language, provider=prov_code)
            run_graph = build_graph()
            with st.spinner("Running analysis..."):
                try:
                    final = run_graph(state)
                except Exception as e:
                    st.error(f"Pipeline error: {e}")
                    return
                if isinstance(final, dict):
                    final = PipelineState.model_validate(final)
        finally:
            # Clean up temp file
            try:
                # Don't delete when using sample
                if uploaded and tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except Exception:
                pass

        if final.errors:
            st.warning("\n".join(final.errors))

        if final.report_markdown:
            # Display the report
            st.markdown(final.report_markdown)
            
            # Validate markdown and show issues if any
            try:
                issues = validate_markdown(final.report_markdown, language)
                if issues:
                    st.info("📋 **Report validation notes:**\n" + "\n".join([f"• {issue}" for issue in issues]))
            except Exception:
                pass  # Skip validation if function not available
            
            # Download button with exact same content
            fname = f"report-{slugify(role)}-{slugify(language)}.md"
            st.download_button(
                label="Download as .md",
                data=final.report_markdown.encode("utf-8"),
                file_name=fname,
                mime="text/markdown"
            )
        else:
            st.error("No report produced.")


if __name__ == "__main__":
    main()
