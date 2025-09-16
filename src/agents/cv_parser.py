from __future__ import annotations
from typing import Any, Dict, List, Optional
import re
from pydantic import BaseModel, Field
from langchain.schema import SystemMessage, HumanMessage


class ExperienceItem(BaseModel):
    company: Optional[str] = None
    title: Optional[str] = None
    period: Optional[str] = None
    bullets: List[str] = Field(default_factory=list)


class ProjectItem(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    tech: List[str] = Field(default_factory=list)


class CVSchema(BaseModel):
    name: Optional[str] = None
    summary: Optional[str] = None
    skills_explicit: List[str] = Field(default_factory=list)
    experiences: List[ExperienceItem] = Field(default_factory=list)
    projects: List[ProjectItem] = Field(default_factory=list)
    education: Optional[str] = None


def naive_section_split(text: str) -> Dict[str, str]:
    sections = {"summary": "", "experience": "", "projects": "", "education": "", "skills": ""}
    t = text.replace("\r", "")
    patterns = {
        "experience": r"(experience|work experience|pengalaman kerja)\s*[:\n]",
        "projects": r"(projects|project|proyek)\s*[:\n]",
        "education": r"(education|pendidikan)\s*[:\n]",
        "skills": r"(skills|kemampuan|keahlian)\s*[:\n]",
    }
    idxs = []
    for name, pat in patterns.items():
        m = re.search(pat, t, flags=re.I)
        if m:
            idxs.append((m.start(), name))
    idxs.sort()
    if not idxs:
        sections["summary"] = t.strip()
        return sections
    first_start = idxs[0][0]
    sections["summary"] = t[:first_start].strip()
    for i, (start, name) in enumerate(idxs):
        end = idxs[i + 1][0] if i + 1 < len(idxs) else len(t)
        sections[name] = t[start:end].strip()
    return sections


def parse_skills_from_text(skills_blob: str) -> list[str]:
    raw = re.split(r"[,;\n]", skills_blob)
    norm = sorted(set([s.strip().lower() for s in raw if s.strip()]))
    return norm


def parse_cv_llm(text: str, llm: Any) -> Dict[str, Any]:
    """Parse resume text into CVSchema using LLM with strict JSON-only output.
    Falls back to naive parsing if LLM fails.
    """
    system = SystemMessage(content=(
        "You extract structured data from resumes into a strict JSON schema. Output only JSON. No commentary."
    ))
    schema_json = (
        '{"name": str|null, "summary": str|null, "skills_explicit": string[] (lowercase, concrete technologies only), '
        '"experiences": [{"company": str|null, "title": str|null, "period": str|null, "bullets": string[]}], '
        '"projects": [{"name": str|null, "description": str|null, "tech": string[]}], "education": str|null}'
    )
    human = HumanMessage(content=(
        "Resume text:\n" + text + "\n\nSchema (JSON) you must return exactly:\n" + schema_json
    ))
    try:
        resp = llm.invoke([system, human])
        content = getattr(resp, "content", "").strip()
        # Try to load JSON robustly
        import json
        data = json.loads(content)
        model = CVSchema.model_validate(data)
        # Normalize skills
        model.skills_explicit = sorted(set([s.strip().lower() for s in model.skills_explicit if s and s.strip()]))
        return model.model_dump()
    except Exception:
        # Fallback to naive parsing
        sections = naive_section_split(text)
        skills = parse_skills_from_text(sections.get("skills", "")) if sections.get("skills") else []
        return {
            "name": None,
            "summary": sections.get("summary", ""),
            "skills_explicit": skills,
            "experiences": [],
            "projects": [],
            "education": sections.get("education", ""),
        }
