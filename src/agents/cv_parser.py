from __future__ import annotations
from typing import Any, Dict, List
import re
import json
import logging
from pydantic import BaseModel, Field
from langchain.schema import SystemMessage, HumanMessage


class CVExperience(BaseModel):
    company: str | None = None
    title: str | None = None
    period: str | None = None
    bullets: list[str] = Field(default_factory=list)


class CVProject(BaseModel):
    name: str | None = None
    description: str | None = None
    tech: list[str] = Field(default_factory=list)


class CVSchema(BaseModel):
    name: str | None = None
    summary: str | None = None
    skills_explicit: list[str] = Field(default_factory=list)
    experiences: list[CVExperience] = Field(default_factory=list)
    projects: list[CVProject] = Field(default_factory=list)
    education: str | None = None


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


def _extract_json_block(text: str) -> str:
    """Extract JSON from LLM response, handling code fences and finding first {...} block."""
    t = text.strip()
    # Strip code fences if any
    t = re.sub(r"^```[a-zA-Z]*\n|```$", "", t, flags=re.MULTILINE)
    # Find first {...}
    m = re.search(r"\{[\s\S]*\}", t)
    if m:
        return t[m.start():m.end()]
    return t


def parse_cv_llm(text: str, llm: Any) -> Dict[str, Any]:
    """Parse resume text into CVSchema using LLM with strict JSON-only output.
    Falls back to naive parsing if LLM fails.
    """
    system = SystemMessage(content=(
        "You extract structured data from resumes into a strict JSON schema. Output ONLY JSON. No commentary, no markdown."
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
        
        # Try to extract and parse JSON robustly
        raw_json = _extract_json_block(content)
        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError:
            # Try once more with different extraction
            raw_json = _extract_json_block(content)
            data = json.loads(raw_json)
        
        model = CVSchema.model_validate(data)
        # Normalize skills
        model.skills_explicit = sorted(set([s.strip().lower() for s in model.skills_explicit if s and s.strip()]))
        return model.model_dump()
        
    except Exception as e:
        logging.warning(f"LLM parsing failed: {e}, falling back to naive parsing")
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


def parse_cv_to_structured(text: str, llm: Any) -> Dict[str, Any]:
    """Main entry point for CV parsing. Tries LLM-based parsing first, falls back to naive on failure."""
    return parse_cv_llm(text, llm)
