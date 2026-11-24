from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, convert_to_openai_messages
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from .config import get_setting


class LLMProviderError(RuntimeError):
    """A provider failure that must not be hidden by parsing fallbacks."""


class EmptyLLMResponseError(LLMProviderError):
    """The model completed the request without usable text."""


def normalize_provider(provider: str | None) -> str:
    value = (provider or "sumopod").strip().lower()
    if value in {"sumopod", "auto"}:
        return "sumopod"
    raise ValueError("Unsupported LLM provider. Use 'sumopod'.")


class SumoPodLLM:
    """Adapt SumoPod chat completions to the agents' invoke interface."""

    def __init__(self, temperature: float = 0.2):
        key = get_setting("API_KEY")
        if not key:
            raise LLMProviderError("Missing API_KEY. Set your SumoPod key in .env or Streamlit Secrets.")
        self.model = get_setting("MODEL", "glm-5.3-flash")
        self.base_url = get_setting("BASE_URL", "https://ai.sumopod.com/v1").rstrip("/")
        self.temperature = temperature
        self.client = OpenAI(api_key=key, base_url=self.base_url, timeout=45.0, max_retries=1)

    def invoke(self, messages: list[Any]) -> AIMessage:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=convert_to_openai_messages(messages),
                max_tokens=4096,
                temperature=self.temperature,
            )
        except APITimeoutError as exc:
            raise LLMProviderError("SumoPod request timed out. Please try again later.") from exc
        except APIConnectionError as exc:
            raise LLMProviderError("Could not connect to SumoPod. Check BASE_URL and your connection.") from exc
        except APIStatusError as exc:
            raise LLMProviderError(
                f"SumoPod returned HTTP {exc.status_code}. Check API_KEY, MODEL, and your API quota."
            ) from exc
        if not response.choices:
            raise LLMProviderError("SumoPod returned no completion choices.")
        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise LLMProviderError("SumoPod response reached its token limit. Try a shorter CV or another MODEL.")
        if choice.finish_reason == "content_filter":
            raise LLMProviderError("SumoPod filtered the response. Check the input or choose another MODEL.")
        content = choice.message.content
        if not isinstance(content, str) or not content.strip():
            raise EmptyLLMResponseError("SumoPod returned an empty text response. Check the configured MODEL.")
        return AIMessage(content=content.strip())

    def close(self) -> None:
        self.client.close()


def get_llm(provider: str = "sumopod", temperature: float = 0.2) -> SumoPodLLM:
    normalize_provider(provider)
    return SumoPodLLM(temperature=temperature)
