from __future__ import annotations
import json
import re
from typing import Any, Dict, List
from pydantic import BaseModel, Field
from langchain.schema import SystemMessage, HumanMessage


# -------- Data models --------
class TableItem(BaseModel):
    skill: str
    notes: str


class WeekPlan(BaseModel):
    title: str
    tasks: List[str] = Field(default_factory=list)


class ReportData(BaseModel):
    overview: str = ""
    strengths: List[TableItem] = Field(default_factory=list)
    gaps: List[TableItem] = Field(default_factory=list)
    plan_weeks: List[WeekPlan] = Field(default_factory=list)
    final_notes: str = ""


# -------- Helpers --------
def _diff_lists(candidate: List[str], market: List[str]) -> Dict[str, List[str]]:
    set_c = set([x.lower() for x in candidate])
    set_m = set([x.lower() for x in market])
    return {
        "strengths": sorted(set_c.intersection(set_m)),
        "gaps": sorted(set_m - set_c),
        "extras": sorted(set_c - set_m),
    }


def build_report_prompt(language: str, context: Dict[str, Any]) -> List[Any]:
    lang = (language or "").lower()
    is_id = lang.startswith("indo") or lang == "id" or lang == "indonesia"
    sys = SystemMessage(content=(
        "You are a reporting assistant. Return ONLY JSON, no markdown, no code fences.\n"
        "Use the exact schema for ReportData. Ensure short, factual sentences."
    ))
    schema_example = (
        '{"overview":"","strengths":[{"skill":"","notes":""}],'
        '"gaps":[{"skill":"","notes":""}],'
        '"plan_weeks":[{"title":"","tasks":[""]}],'
        '"final_notes":""}'
    )
    parts = [
        "LANGUAGE: indonesian" if is_id else "LANGUAGE: english",
        "SCHEMA (JSON shape example):",
        schema_example,
        "CONTEXT:",
        json.dumps(context, ensure_ascii=False),
        "REQUIREMENTS:",
        "- strengths/gaps are concrete technical skills with short justification in notes.",
        "- plan_weeks length 2–4, each with 3–5 actionable tasks.",
        "- Output ONLY valid JSON for ReportData.",
    ]
    user = HumanMessage(content="\n".join(parts))
    return [sys, user]


def _extract_json_block(text: str) -> str:
    t = text.strip()
    # Strip code fences if any
    t = re.sub(r"^```[a-zA-Z]*\n|```$", "", t, flags=re.MULTILINE)
    # Find first {...}
    m = re.search(r"\{[\s\S]*\}", t)
    if m:
        return t[m.start():m.end()]
    return t


def _default_weeks_from_gaps(gaps: List[TableItem], language: str) -> List[WeekPlan]:
    is_id = (language or "").lower().startswith("indo")
    titles_id = ["Dasar & Instalasi", "Latihan Inti", "Proyek Mini"]
    titles_en = ["Setup & Basics", "Core Practice", "Mini Project"]
    titles = titles_id if is_id else titles_en
    tasks = []
    for g in gaps[:3]:
        if is_id:
            tasks.append([f"Pelajari dasar {g.skill}", f"Ikuti tutorial 1-2 jam {g.skill}", f"Praktikkan 2-3 latihan {g.skill}"])
        else:
            tasks.append([f"Learn {g.skill} basics", f"Follow a 1-2h tutorial on {g.skill}", f"Practice 2-3 exercises using {g.skill}"])
    while len(tasks) < 3:
        tasks.append(["Riset teknologi relevan", "Catat ringkasan 1 halaman", "Diskusikan dengan mentor/teman"] if is_id else ["Research relevant tech", "Write a one-page summary", "Discuss with a peer/mentor"])
    return [WeekPlan(title=titles[i], tasks=tasks[i]) for i in range(3)]


def validate_report_data(rd: ReportData, language: str, context: Dict[str, Any]) -> None:
    is_id = (language or "").lower().startswith("indo")
    # Ensure strengths or gaps not both empty
    if not rd.strengths and not rd.gaps:
        placeholder = "Belum teridentifikasi" if is_id else "Not identified yet"
        rd.gaps = [TableItem(skill="-", notes=placeholder)]
    # Length of plan weeks between 2 and 4
    if len(rd.plan_weeks) < 2 or len(rd.plan_weeks) > 4:
        rd.plan_weeks = _default_weeks_from_gaps(rd.gaps, language)


def generate_report_data(llm: Any, language: str, context: Dict[str, Any]) -> ReportData:
    messages = build_report_prompt(language, context)
    try:
        resp = llm.invoke(messages)
        content = getattr(resp, "content", "").strip()
    except Exception as e:
        # LLM failed, fallback
        strengths = [TableItem(skill=s, notes="terkait kebutuhan pasar") for s in context.get("diff", {}).get("strengths", [])][:5]
        gaps = [TableItem(skill=s, notes="prioritas belajar") for s in context.get("diff", {}).get("gaps", [])][:5]
        if not strengths and not gaps:
            if (language or "").lower().startswith("indo"):
                gaps = [TableItem(skill="-", notes="Belum teridentifikasi")]
            else:
                gaps = [TableItem(skill="-", notes="Not identified yet")]
        return ReportData(
            overview=context.get("summary", "")[:600],
            strengths=strengths,
            gaps=gaps,
            plan_weeks=_default_weeks_from_gaps(gaps, language),
            final_notes=("Gunakan rencana belajar untuk menutup kesenjangan utama." if (language or "").lower().startswith("indo") else "Follow the upskilling plan to close key gaps."),
        )

    raw = _extract_json_block(content)
    try:
        data = json.loads(raw)
        rd = ReportData.model_validate(data)
    except Exception:
        # Attempt salvage once more
        try:
            data = json.loads(_extract_json_block(raw))
            rd = ReportData.model_validate(data)
        except Exception:
            # fallback minimal
            return ReportData(
                overview=context.get("summary", "")[:600],
                strengths=[TableItem(skill=s, notes="sesuai pasar") for s in context.get("diff", {}).get("strengths", [])][:5],
                gaps=[TableItem(skill=s, notes="prioritas belajar") for s in context.get("diff", {}).get("gaps", [])][:5] or [TableItem(skill="-", notes=("Belum teridentifikasi" if (language or "").lower().startswith("indo") else "Not identified yet"))],
                plan_weeks=_default_weeks_from_gaps([TableItem(skill=s, notes="") for s in context.get("diff", {}).get("gaps", [])], language),
                final_notes=("Gunakan rencana belajar untuk menutup kesenjangan utama." if (language or "").lower().startswith("indo") else "Follow the upskilling plan to close key gaps."),
            )

    validate_report_data(rd, language, context)
    return rd


def postprocess_markdown(md: str, language: str) -> str:
    s = md.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse 3+ blank lines to 2
    s = re.sub(r"\n{3,}", "\n\n", s)
    is_id = (language or "").lower().startswith("indo")
    final_header = "## Catatan Akhir" if is_id else "## Final Notes"
    # Deduplicate final section keeping only last
    matches = list(re.finditer(re.escape(final_header), s))
    if len(matches) > 1:
        last = matches[-1].start()
        head = s[:last]
        # remove earlier final sections from head
        head = re.sub(re.escape(final_header) + r"[\s\S]*$", "", head, flags=re.MULTILINE)
        s = head + s[last:]
    # Trim trailing spaces
    s = re.sub(r"[ \t]+\n", "\n", s)
    return s.rstrip() + "\n"


def render_markdown_id(report: ReportData) -> str:
    lines: List[str] = []
    lines += ["## Ikhtisar", report.overview.strip() or "-"]
    lines += ["", "## Kekuatan", "| Skill | Catatan |", "|---|---|"]
    if report.strengths:
        for item in report.strengths:
            lines.append(f"| {item.skill} | {item.notes} |")
    else:
        lines.append("| - | Belum teridentifikasi |")
    lines += ["", "## Kesenjangan Keahlian", "| Skill | Catatan |", "|---|---|"]
    if report.gaps:
        for item in report.gaps:
            lines.append(f"| {item.skill} | {item.notes} |")
    else:
        lines.append("| - | Belum teridentifikasi |")
    lines += ["", "## Rencana Upskilling"]
    for i, wk in enumerate(report.plan_weeks, 1):
        title = wk.title.strip() or f"Minggu {i}"
        lines.append(f"**Minggu {i} — {title}**")
        for t in wk.tasks or ["-"]:
            lines.append(f"- {t}")
    lines += ["", "## Catatan Akhir", report.final_notes.strip() or "-"]
    md = "\n".join(lines)
    return postprocess_markdown(md, "indonesia")


def render_markdown_en(report: ReportData) -> str:
    lines: List[str] = []
    lines += ["## Overview", report.overview.strip() or "-"]
    lines += ["", "## Strengths", "| Skill | Evidence/Notes |", "|---|---|"]
    if report.strengths:
        for item in report.strengths:
            lines.append(f"| {item.skill} | {item.notes} |")
    else:
        lines.append("| - | Not identified yet |")
    lines += ["", "## Skill Gaps", "| Skill | Evidence/Notes |", "|---|---|"]
    if report.gaps:
        for item in report.gaps:
            lines.append(f"| {item.skill} | {item.notes} |")
    else:
        lines.append("| - | Not identified yet |")
    lines += ["", "## Actionable Upskilling Plan"]
    for i, wk in enumerate(report.plan_weeks, 1):
        title = wk.title.strip() or f"Week {i}"
        lines.append(f"**Week {i} — {title}**")
        for t in wk.tasks or ["-"]:
            lines.append(f"- {t}")
    lines += ["", "## Final Notes", report.final_notes.strip() or "-"]
    md = "\n".join(lines)
    return postprocess_markdown(md, "english")


def validate_markdown(md: str, language: str) -> List[str]:
    issues: List[str] = []
    is_id = (language or "").lower().startswith("indo")
    if is_id:
        headers = ["## Ikhtisar", "## Kekuatan", "## Kesenjangan Keahlian", "## Rencana Upskilling", "## Catatan Akhir"]
        final = "## Catatan Akhir"
    else:
        headers = ["## Overview", "## Strengths", "## Skill Gaps", "## Actionable Upskilling Plan", "## Final Notes"]
        final = "## Final Notes"
    for h in headers:
        if h not in md:
            issues.append(f"Missing section: {h}")
    if len(re.findall(re.escape(final), md)) != 1:
        issues.append("Final section appears not exactly once.")
    for i, h in enumerate(headers):
        start = md.find(h)
        if start == -1:
            continue
        end = min([md.find(h2) for h2 in headers[i+1:] if md.find(h2) != -1] + [len(md)])
        body = md[start+len(h):end].strip()
        if not body:
            issues.append(f"Empty section: {h}")
    return issues


def make_report(cv_structured: Dict[str, Any],
                analyzed_skills: Dict[str, Any],
                market: Dict[str, Any],
                llm: Any,
                language: str,
                style: Dict[str, Any] | None = None) -> str:
    explicit = analyzed_skills.get("explicit_skills", [])
    implicit = analyzed_skills.get("implicit_skills", [])
    market_sk = market.get("skills", [])
    diff = _diff_lists(explicit + implicit, market_sk)
    context = {
        "summary": cv_structured.get("summary", "")[:800],
        "explicit": explicit,
        "implicit": implicit,
        "market": market_sk,
        "diff": diff,
        "role": market.get("role", ""),
        "source": market.get("source", "")
    }
    rd = generate_report_data(llm, language, context)
    if (language or "").lower().startswith("indo"):
        return render_markdown_id(rd)
    return render_markdown_en(rd)
