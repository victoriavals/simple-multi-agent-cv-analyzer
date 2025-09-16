from __future__ import annotations
from typing import Dict, Any, List
import os
from langchain.schema import SystemMessage, HumanMessage


def _read_secrets() -> dict:
    try:
        import streamlit as st  # type: ignore
        if hasattr(st, "secrets"):
            return dict(st.secrets)
    except Exception:
        pass
    return {}


def fetch_market_blurbs(role: str) -> List[str]:
    secrets = _read_secrets()
    api_key = secrets.get("TAVILY_API_KEY") or os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is missing. Set it in .env or Streamlit secrets.")
    from tavily import TavilyClient
    client = TavilyClient(api_key=api_key)
    res = client.search(query=f"{role} required skills tech stack 2025", max_results=8)
    blurbs: List[str] = []
    for item in res.get("results", []):
        title = item.get("title", "")
        content = (item.get("content", "") or "")[:600]
        blob = (title + "\n" + content).strip().lower()
        if blob:
            blurbs.append(blob)
    if not blurbs:
        raise RuntimeError("No Tavily results. Try a different role or check your API key limits.")
    return blurbs


def synthesize_market_skills(blurbs: List[str], llm: Any) -> List[str]:
    system = SystemMessage(content=(
        "You distill current market skills for a target role from web snippets. Output only a comma-separated list of concrete tools/skills, lowercase, max 30, no soft skills."
    ))
    human = HumanMessage(content=(
        "Snippets:\n" + "\n---\n".join(blurbs) + "\n\nReturn only the skills list, comma-separated."
    ))
    resp = llm.invoke([system, human])
    content = getattr(resp, "content", "").strip()
    tokens = [t.strip().lower() for t in content.split(",") if t.strip()]
    # Dedupe and limit
    tokens = sorted(set(tokens))[:30]
    return tokens


def get_market_requirements(target_role: str, llm: Any) -> Dict[str, Any]:
    blurbs = fetch_market_blurbs(target_role)
    skills = synthesize_market_skills(blurbs, llm)
    if not skills:
        raise RuntimeError("LLM returned no skills from market snippets.")
    return {"role": target_role, "source": "tavily", "skills": skills}
