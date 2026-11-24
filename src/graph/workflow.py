from __future__ import annotations
from typing import Callable
from langgraph.graph import StateGraph, END
from ..state import PipelineState
from ..utils import load_cv
from ..llm_provider import get_llm, normalize_provider
from ..agents.cv_parser import parse_cv_to_structured
from ..agents.skill_analyst import analyze_skills
from ..agents.market_intel import market_intelligence_agent
from ..agents.report_agent import make_report

def build_graph(
    on_progress: Callable[[str], None] | None = None,
) -> Callable[[PipelineState], PipelineState]:
    # llm will be constructed using the state's selected provider at runtime
    llm_holder = {"llm": None}

    def load_cv_node(state: PipelineState) -> PipelineState:
        if on_progress is not None:
            on_progress("reading")
        if not state.target_role.strip():
            state.errors.append("Please enter a target role.")
            return state
        if state.market_source == "job-description" and not (state.job_description or "").strip():
            state.errors.append("Please provide a job description for this analysis mode.")
            return state
        try:
            state.cv_raw_text = load_cv(state.cv_path)
            if not state.cv_raw_text.strip():
                raise ValueError("The CV contains no readable text.")
        except Exception as e:
            state.errors.append(f"Load CV error: {e}")
            return state
        # Initialize LLM once using selected provider
        if llm_holder["llm"] is None:
            try:
                prov = normalize_provider(state.provider)
                llm_holder["llm"] = get_llm(provider=prov, temperature=0.2)
            except Exception as e:
                state.errors.append(f"LLM init error: {e}")
                return state
        return state

    def parse_node(state: PipelineState) -> PipelineState:
        if state.errors:
            return state
        if not state.cv_raw_text:
            state.errors.append("CV kosong. Gagal mem-parsing.")
            return state
        try:
            state.cv_structured = parse_cv_to_structured(state.cv_raw_text, llm_holder["llm"])
        except Exception as e:
            state.errors.append(f"CV parse error: {e}")
        return state

    def analyze_node(state: PipelineState) -> PipelineState:
        if state.errors:
            return state
        if not state.cv_structured:
            state.errors.append("CV belum terstruktur.")
            return state
        try:
            if on_progress is not None:
                on_progress("skills")
            state.analyzed_skills = analyze_skills(state.cv_structured, llm_holder["llm"])
        except Exception as e:
            state.errors.append(f"Skill analysis error: {e}")
        return state

    def market_node(state: PipelineState) -> PipelineState:
        if state.errors:
            return state
        try:
            if on_progress is not None:
                on_progress("requirements")
            state.market_requirements = market_intelligence_agent(
                state.target_role, llm_holder["llm"], state.job_description, state.market_source
            )
            state.warnings.extend(state.market_requirements.get("warnings", []))
        except Exception as e:
            state.errors.append(f"Market intel error: {e}")
        return state

    def report_node(state: PipelineState) -> PipelineState:
        if state.errors:
            return state
        if not state.cv_structured or not state.analyzed_skills or not state.market_requirements:
            state.errors.append("Data belum lengkap untuk membuat report.")
            return state
        try:
            if on_progress is not None:
                on_progress("report")
            state.report_markdown = make_report(
                state.cv_structured,
                state.analyzed_skills,
                state.market_requirements,
                llm_holder["llm"],
                getattr(state, "language", "english")
            )
        except Exception as e:
            state.errors.append(f"Report generation error: {e}")
        return state

    g = StateGraph(PipelineState)
    g.add_node("load_cv", load_cv_node)
    g.add_node("parse", parse_node)
    g.add_node("analyze", analyze_node)
    g.add_node("market", market_node)
    g.add_node("report", report_node)

    g.set_entry_point("load_cv")
    g.add_edge("load_cv", "parse")
    g.add_edge("parse", "analyze")
    g.add_edge("analyze", "market")
    g.add_edge("market", "report")
    g.add_edge("report", END)

    app = g.compile()

    def runner(state: PipelineState) -> PipelineState:
        try:
            return app.invoke(state)
        finally:
            llm = llm_holder["llm"]
            llm_holder["llm"] = None
            close = getattr(llm, "close", None)
            if callable(close):
                close()

    return runner
