from __future__ import annotations
from typing import Any, Dict, List
from langchain.schema import HumanMessage, SystemMessage

# Beberapa keyword heuristik biar tetap bisa jalan tanpa LLM
IMPLICIT_MAP = {
    "pytorch": ["autograd", "tensor ops", "gpu training"],
    "tensorflow": ["graph execution", "model serving"],
    "sklearn": ["model selection", "pipeline", "cross-validation"],
    "langchain": ["prompt design", "tool calling", "retrieval"],
    "docker": ["containerization", "image build", "runtime isolation"],
    "kubernetes": ["orchestration", "scaling", "service mesh"],
    "mlflow": ["experiment tracking", "model registry"],
    "airflow": ["dag scheduling", "etl orchestration"],
    "redis": ["caching", "pubsub", "kv store"],
    "postgres": ["sql", "indexing", "query planning"],
    "rag": ["vector search", "chunking", "embeddings"],
    "faiss": ["ann search", "index types", "recall metrics"],
    "weaviate": ["vector db", "schema", "hybrid search"],
    "opensearch": ["fulltext", "bm25", "vector hybrid"],
    "onnx": ["model export", "runtime"],
    "huggingface": ["transformers", "tokenizers", "datasets"]
}

def infer_implicit_skills(explicit: List[str]) -> List[str]:
    result = set()
    for token in explicit:
        if token in IMPLICIT_MAP:
            for v in IMPLICIT_MAP[token]:
                result.add(v)
    return sorted(result)

def analyze_skills(cv_structured: Dict[str, Any], llm: Any) -> Dict[str, Any]:
    # Base explicit skills from LLM CV parser output (preferred key 'skills_explicit'; keep 'skills_list' for backward compat)
    base_skills = cv_structured.get("skills_explicit") or cv_structured.get("skills_list") or []
    explicit = sorted(set([s.lower() for s in base_skills]))

    # LLM pass to infer additional explicit skills from narrative text
    narrative = "\n\n".join([
        str(cv_structured.get("summary", "")),
        str(cv_structured.get("experience", "")),
        str(cv_structured.get("projects", "")),
        "\n".join([p.get("description", "") for p in cv_structured.get("projects", []) if isinstance(p, dict)])
    ])
    messages = [
        SystemMessage(content=(
            "Extract only concrete technical skills/tools/libraries from the text. Return a comma-separated list, lowercase, no commentary."
        )),
        HumanMessage(content=(
            "TEXT:\n" + narrative + "\n\nCURRENT EXPLICIT SKILLS:\n" + ", ".join(explicit)
        )),
    ]
    extra: List[str] = []
    try:
        resp = llm.invoke(messages)
        content = getattr(resp, "content", "")
        extra = [x.strip().lower() for x in content.split(",") if x.strip()]
        extra = [e for e in extra if e not in explicit]
    except Exception:
        extra = []

    combined = sorted(set(explicit + extra))
    implicit = infer_implicit_skills(combined)

    return {
        "explicit_skills": combined,
        "implicit_skills": implicit,
        "notes": "Implicit skills inferred via simple mapping; extra explicit skills mined by LLM from summary/experience/projects."
    }
