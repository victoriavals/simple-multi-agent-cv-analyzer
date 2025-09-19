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

def _is_tech_skill(token: str) -> bool:
    """Filter out obviously non-technical tokens."""
    token = token.strip().lower()
    if len(token) < 2:
        return False
    
    # Filter out common non-tech words
    non_tech = {
        "and", "or", "the", "a", "an", "in", "on", "at", "to", "for", "of", "with", "by",
        "experience", "skills", "knowledge", "ability", "capable", "proficient", "expert",
        "years", "months", "work", "working", "project", "projects", "development", "developing",
        "team", "teams", "management", "managing", "lead", "leading", "senior", "junior",
        "strong", "good", "excellent", "basic", "advanced", "intermediate", "beginner",
        "using", "used", "use", "including", "include", "such", "as", "like", "etc",
        "various", "multiple", "different", "several", "many", "some", "all", "most",
        "business", "company", "industry", "client", "customer", "user", "users"
    }
    
    if token in non_tech:
        return False
    
    # Keep tokens that look like tech (contain numbers, have specific patterns)
    if any(char.isdigit() for char in token):  # e.g., "python3", "java8"
        return True
    
    # Keep common tech patterns
    tech_patterns = ['.js', '.py', '.java', '.net', '.css', '.html', '.xml', '.json', '.sql']
    if any(pattern in token for pattern in tech_patterns):
        return True
    
    # Keep if it's a known tech term (basic heuristic)
    tech_keywords = {
        'api', 'sql', 'nosql', 'rest', 'graphql', 'json', 'xml', 'html', 'css', 'javascript',
        'python', 'java', 'golang', 'rust', 'scala', 'kotlin', 'swift', 'typescript',
        'react', 'vue', 'angular', 'node', 'express', 'django', 'flask', 'spring',
        'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'ansible',
        'git', 'github', 'gitlab', 'jenkins', 'cicd', 'devops', 'microservices',
        'mongodb', 'postgresql', 'mysql', 'redis', 'elasticsearch', 'kafka',
        'tensorflow', 'pytorch', 'sklearn', 'pandas', 'numpy', 'jupyter'
    }
    
    return token in tech_keywords or len(token) >= 3


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
    explicit = sorted(set([s.lower().strip() for s in base_skills if s and s.strip()]))

    # LLM pass to infer additional explicit skills from narrative text
    narrative = "\n\n".join([
        str(cv_structured.get("summary", "")),
        str(cv_structured.get("experience", "")),
        str(cv_structured.get("projects", "")),
        "\n".join([p.get("description", "") for p in cv_structured.get("projects", []) if isinstance(p, dict)])
    ]).strip()
    
    messages = [
        SystemMessage(content=(
            "Extract concrete technical skills, tools, programming languages, frameworks, and libraries from text. "
            "Return ONLY a comma-separated list (lowercase). No prose, no explanations, no commentary."
        )),
        HumanMessage(content=(
            f"TEXT:\n{narrative}\n\nCURRENT SKILLS:\n{', '.join(explicit)}\n\n"
            "Extract additional technical skills not already listed:"
        )),
    ]
    
    extra: List[str] = []
    try:
        resp = llm.invoke(messages)
        content = getattr(resp, "content", "").strip()
        
        # Parse comma-separated response
        candidates = [x.strip().lower() for x in content.split(",") if x.strip()]
        
        # Filter and deduplicate
        extra = [
            skill for skill in candidates 
            if skill not in explicit and _is_tech_skill(skill)
        ]
    except Exception:
        extra = []

    # Combine and deduplicate all explicit skills
    combined = sorted(set(explicit + extra))
    
    # Infer implicit skills from the combined explicit skills
    implicit = infer_implicit_skills(combined)

    return {
        "explicit_skills": combined,
        "implicit_skills": implicit,
        "notes": "Implicit inferred via mapping; extra explicit via LLM."
    }
