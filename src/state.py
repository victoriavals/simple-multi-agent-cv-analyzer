from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

class PipelineState(BaseModel):
    # Input
    cv_path: str
    target_role: str
    language: str | None = None
    provider: str = "sumopod"
    market_source: Literal["ddgs", "job-description"] = "ddgs"
    job_description: Optional[str] = None

    # Intermediate
    cv_raw_text: Optional[str] = None
    cv_structured: Optional[Dict[str, Any]] = None
    analyzed_skills: Optional[Dict[str, Any]] = None
    market_requirements: Optional[Dict[str, Any]] = None

    # Output
    report_markdown: Optional[str] = None
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
