from __future__ import annotations
import os
from typing import Any, Callable, List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI


PROVIDERS = {"auto", "gemini", "mistral"}


def _read_secrets() -> dict[str, str]:
    try:
        import streamlit as st  # type: ignore
        if hasattr(st, "secrets"):
            return dict(st.secrets)
    except Exception:
        pass
    return {}


def _get_secret(name: str) -> Optional[str]:
    secrets = _read_secrets()
    return secrets.get(name) or os.getenv(name)


def normalize_provider(p: str | None) -> str:
    if not p:
        return "auto"
    p = p.strip().lower()
    if p in ("auto", "gemini", "mistral"):
        return p
    return "auto"


def build_gemini(temperature: float = 0.2) -> ChatGoogleGenerativeAI:
    # Prefer GEMINI_API_KEY, fallback to GOOGLE_API_KEY
    key = _get_secret("GEMINI_API_KEY") or _get_secret("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) is missing. Set it in .env or Streamlit secrets.")
    model = _get_secret("GEMINI_MODEL") or "gemini-2.0-flash"
    os.environ["GEMINI_API_KEY"] = key
    os.environ["GOOGLE_API_KEY"] = key
    return ChatGoogleGenerativeAI(model=model, temperature=temperature)


def build_mistral(temperature: float = 0.2) -> ChatMistralAI:
    key = _get_secret("MISTRAL_API_KEY")
    if not key:
        raise RuntimeError("MISTRAL_API_KEY is missing. Set it in .env or Streamlit secrets.")
    model = _get_secret("MISTRAL_MODEL") or "mistral-large-latest"
    os.environ["MISTRAL_API_KEY"] = key
    return ChatMistralAI(model=model, temperature=temperature)


class MultiProviderLLM:
    """Try multiple provider builders in order. Build lazily and failover on errors."""

    def __init__(self, builders: List[Callable[[], Any]]):
        self.builders = builders
        self._instances: List[Any | None] = [None] * len(builders)
        self._errors: List[str] = []

    def invoke(self, messages: list[Any]) -> Any:
        self._errors.clear()
        last_exc: Optional[Exception] = None
        for i, b in enumerate(self.builders):
            # Build if needed
            if self._instances[i] is None:
                try:
                    self._instances[i] = b()
                except Exception as e:
                    self._errors.append(f"build[{i}]: {e}")
                    last_exc = e
                    continue
            model = self._instances[i]
            try:
                return model.invoke(messages)
            except Exception as e:
                self._errors.append(f"invoke[{i}]: {e}")
                last_exc = e
                continue
        raise RuntimeError("All providers failed: " + "; ".join(self._errors)) from last_exc


def get_llm(provider: str = "auto", temperature: float = 0.2) -> Any:
    p = normalize_provider(provider)
    if p == "gemini":
        return build_gemini(temperature)
    if p == "mistral":
        return build_mistral(temperature)
    # auto
    return MultiProviderLLM([
        lambda: build_gemini(temperature),
        lambda: build_mistral(temperature),
    ])
