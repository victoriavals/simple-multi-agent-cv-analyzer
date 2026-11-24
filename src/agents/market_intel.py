from __future__ import annotations
from typing import Any, Dict
from ..tools.market_search import get_market_requirements


def market_intelligence_agent(
    target_role: str, llm: Any, job_description: str | None = None, market_source: str = "ddgs"
) -> Dict[str, Any]:
    return get_market_requirements(target_role, llm, job_description, market_source)
