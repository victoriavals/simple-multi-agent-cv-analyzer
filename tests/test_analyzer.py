from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx
from ddgs.engines.duckduckgo import Duckduckgo
from ddgs.engines.yahoo import Yahoo
from ddgs.exceptions import DDGSException
from ddgs.results import TextResult
from langchain_core.messages import HumanMessage, SystemMessage
from openai import OpenAI
from streamlit.testing.v1 import AppTest

from main import main as cli_main
from src.agents.report_agent import ReportData, WeekPlan, render_markdown_en, render_markdown_id, validate_markdown
from src.graph.workflow import build_graph
from src.llm_provider import EmptyLLMResponseError, LLMProviderError, SumoPodLLM
from src.state import PipelineState
from src.tools.market_search import (
    MarketSearchError, _cached_web_requirements, fetch_market_results, get_market_requirements,
)


ROOT = Path(__file__).resolve().parent.parent
RESULTS = [{"title": "Python role", "content": "Python and Docker required", "url": "https://example.org/jobs/python"}]


class StubLLM:
    model = "unit-model"
    base_url = "https://unit-test.invalid/v1"

    def __init__(self):
        self.calls = []
        self.fail = False

    def invoke(self, messages):
        self.calls.append(messages)
        if self.fail:
            raise LLMProviderError("Provider unavailable")
        system = messages[0].content
        if system.startswith("You extract structured data"):
            content = json.dumps({
                "summary": None, "skills_explicit": ["python"],
                "experiences": [{"bullets": ["CAREER_ONLY_CUDA_TOOL"]}],
                "projects": [{"name": "Demo", "description": None}],
            })
        elif system.startswith("Extract concrete technical"):
            content = "sql"
        elif system.startswith("Extract required technical"):
            content = "python, docker"
        elif system.startswith("You are a reporting assistant"):
            content = json.dumps({
                "overview": "Candidate overview", "strengths": [{"skill": "python", "notes": "CV evidence"}],
                "gaps": [{"skill": "docker", "notes": "Practice deployment"}], "final_notes": "Continue learning.",
            })
        else:
            raise AssertionError("Unexpected agent prompt")
        return SimpleNamespace(content=content)


class ProviderTests(unittest.TestCase):
    def make_llm(self, handler):
        transport_client = httpx.Client(transport=httpx.MockTransport(handler))
        settings = {"API_KEY": "unit-test-key", "MODEL": "glm-unit-test", "BASE_URL": "https://ai.sumopod.com/v1"}
        with patch("src.llm_provider.get_setting", side_effect=lambda name, default="": settings.get(name, default)):
            llm = SumoPodLLM()
        llm.client.close()
        llm.client = OpenAI(api_key="unit-test-key", base_url=llm.base_url, http_client=transport_client, max_retries=0)
        self.addCleanup(llm.client.close)
        return llm

    def test_chat_completion_uses_sumopod_model_and_message_roles(self):
        requests = []

        def respond(request):
            requests.append(request)
            return httpx.Response(200, json={
                "id": "unit", "object": "chat.completion", "created": 0, "model": "glm-unit-test",
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "OK"}}],
            })

        llm = self.make_llm(respond)
        answer = llm.invoke([SystemMessage(content="System prompt"), HumanMessage(content="Hello")])
        self.assertEqual(answer.content, "OK")
        self.assertEqual(str(requests[0].url), "https://ai.sumopod.com/v1/chat/completions")
        self.assertEqual(requests[0].headers["authorization"], "Bearer unit-test-key")
        body = json.loads(requests[0].content)
        self.assertEqual(body["model"], "glm-unit-test")
        self.assertEqual([item["role"] for item in body["messages"]], ["system", "user"])

    def test_authentication_error_is_actionable_and_redacted(self):
        llm = self.make_llm(lambda request: httpx.Response(401, json={"error": {"message": "Unauthorized"}}))
        with self.assertRaises(LLMProviderError) as caught:
            llm.invoke([HumanMessage(content="Hello")])
        self.assertIn("HTTP 401", str(caught.exception))
        self.assertNotIn("unit-test-key", str(caught.exception))

    def test_streamlit_secrets_use_existing_variable_names(self):
        with patch("streamlit.secrets", {"API_KEY": "secret-from-cloud", "MODEL": "cloud-model"}), patch("src.llm_provider.OpenAI") as client:
            llm = SumoPodLLM()
        self.assertEqual(llm.model, "cloud-model")
        self.assertEqual(client.call_args.kwargs["api_key"], "secret-from-cloud")


class MarketTests(unittest.TestCase):
    def setUp(self):
        _cached_web_requirements.clear()
        self.addCleanup(_cached_web_requirements.clear)

    def test_search_and_skill_synthesis_are_reused_for_normalized_role(self):
        llm = StubLLM()
        with patch("src.tools.market_search.fetch_market_results", return_value=RESULTS) as search:
            first = get_market_requirements(" Data   Scientist ", llm)
            second = get_market_requirements("data scientist", llm)
        self.assertEqual(first["skills"], second["skills"])
        self.assertEqual(second["source"], "ddgs")
        self.assertEqual(search.call_count, 1)
        self.assertEqual(len(llm.calls), 1)

    def test_a_different_model_does_not_reuse_old_market_synthesis(self):
        llm = StubLLM()
        with patch("src.tools.market_search.fetch_market_results", return_value=RESULTS) as search:
            get_market_requirements("Data Scientist", llm)
            llm.model = "other-model"
            get_market_requirements("Data Scientist", llm)
        self.assertEqual(search.call_count, 2)

    def test_search_failure_uses_supplied_description_with_clear_provenance(self):
        llm = StubLLM()
        with patch("src.tools.market_search.fetch_market_results", side_effect=MarketSearchError("No connection")):
            result = get_market_requirements("Data Scientist", llm, "Private job: Python and Docker")
        self.assertEqual(result["source"], "job_description")
        self.assertTrue(result["warnings"])
        self.assertIn("Private job", llm.calls[0][1].content)

    def test_search_failure_without_description_does_not_invent_market_data(self):
        llm = StubLLM()
        with patch("src.tools.market_search.fetch_market_results", side_effect=MarketSearchError("No connection")):
            with self.assertRaises(MarketSearchError):
                get_market_requirements("Data Scientist", llm)
        self.assertFalse(llm.calls)

    def test_manual_descriptions_are_separate_and_skip_web_search(self):
        llm = StubLLM()
        with patch("src.tools.market_search.fetch_market_results") as search:
            get_market_requirements("Data Scientist", llm, "PRIVATE_DESCRIPTION_A", "job-description")
            get_market_requirements("Data Scientist", llm, "PRIVATE_DESCRIPTION_B", "job-description")
        search.assert_not_called()
        self.assertEqual(len(llm.calls), 2)
        self.assertIn("PRIVATE_DESCRIPTION_A", llm.calls[0][1].content)
        self.assertIn("PRIVATE_DESCRIPTION_B", llm.calls[1][1].content)

    def test_search_bounds_results_and_snippet_size(self):
        with patch("src.tools.market_search.DDGS") as factory:
            factory.return_value.__enter__.return_value.text.return_value = [
                {"title": "Job", "body": "a" * 5000, "href": "javascript:bad"} for _ in range(20)
            ]
            results = fetch_market_results("Engineer", 2026)
        self.assertLessEqual(len(results), 5)
        self.assertTrue(all(len(item["content"]) <= 600 and not item["url"] for item in results))

    def test_search_can_use_yahoo_when_duckduckgo_is_unreachable(self):
        result = TextResult(
            title="Engineer role", body="Engineer technical skills include Python and Docker.",
            href="https://example.org/jobs/engineer",
        )
        with patch.object(Duckduckgo, "search", side_effect=DDGSException("Unreachable")), patch.object(
            Yahoo, "search", return_value=[result]
        ):
            snippets = fetch_market_results("Engineer", 2026)
        self.assertEqual(snippets[0]["url"], result.href)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        _cached_web_requirements.clear()
        self.addCleanup(_cached_web_requirements.clear)

    def test_upskilling_weeks_have_distinct_sections_and_single_week_labels(self):
        for label, render in (("Minggu", render_markdown_id), ("Week", render_markdown_en)):
            with self.subTest(label=label):
                report = ReportData(plan_weeks=[
                    WeekPlan(title=f"{label} 1: Basics", tasks=["Practice basics"]),
                    WeekPlan(title="Mini project", tasks=["Build a project"]),
                ])
                markdown = render(report)
                self.assertIn(f"### {label} 1 — Basics\n\n- Practice basics\n\n### {label} 2 — Mini project", markdown)

    def test_reports_handle_nullable_cv_fields_and_include_work_experience(self):
        for language in ("indonesia", "english"):
            with self.subTest(language=language):
                llm = StubLLM()
                with patch("src.graph.workflow.get_llm", return_value=llm):
                    final = PipelineState.model_validate(build_graph()(PipelineState(
                        cv_path=str(ROOT / "samples/sample_cv.txt"), target_role="AI Engineer", language=language,
                        market_source="job-description", job_description="Python and Docker required",
                    )))
                self.assertFalse(final.errors)
                self.assertTrue(final.report_markdown)
                self.assertFalse(validate_markdown(final.report_markdown, language))
                self.assertIn("CAREER_ONLY_CUDA_TOOL", llm.calls[1][1].content)
                self.assertIn("deskripsi lowongan" if language == "indonesia" else "job description", final.report_markdown)

    def test_report_keeps_cv_skills_when_no_additional_skills_are_returned(self):
        for empty in (True, False):
            with self.subTest(empty_response=empty):
                llm = StubLLM()
                invoke = llm.invoke

                def respond(messages):
                    if messages[0].content.startswith("Extract concrete technical"):
                        if empty:
                            raise EmptyLLMResponseError("Empty text")
                        return SimpleNamespace(content="NONE")
                    return invoke(messages)

                with patch.object(llm, "invoke", side_effect=respond), patch(
                    "src.graph.workflow.get_llm", return_value=llm
                ):
                    final = PipelineState.model_validate(build_graph()(PipelineState(
                        cv_path=str(ROOT / "samples/sample_cv.txt"), target_role="AI Engineer",
                        market_source="job-description", job_description="Python and Docker required",
                    )))
                self.assertEqual(final.errors, [])
                self.assertEqual(final.analyzed_skills["explicit_skills"], ["python"])
                self.assertTrue(final.report_markdown)

    def test_provider_failure_stops_before_market_requests(self):
        llm = StubLLM()
        llm.fail = True
        with patch("src.graph.workflow.get_llm", return_value=llm), patch("src.tools.market_search.fetch_market_results") as search:
            final = PipelineState.model_validate(build_graph()(PipelineState(
                cv_path=str(ROOT / "samples/sample_cv.txt"), target_role="AI Engineer",
            )))
        self.assertTrue(final.errors)
        self.assertIsNone(final.report_markdown)
        self.assertEqual(len(llm.calls), 1)
        search.assert_not_called()

    def test_cli_returns_failure_when_cv_is_missing(self):
        with patch("sys.argv", ["main.py", "--cv", "definitely-missing-cv.txt", "--role", "Engineer"]), patch("main.load_dotenv"), redirect_stdout(StringIO()):
            self.assertEqual(cli_main(), 1)


class StreamlitTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch("dotenv.load_dotenv", return_value=False))
        self.stack.enter_context(patch.dict(os.environ, {"API_KEY": "unit-test-key"}))
        self.stack.enter_context(patch("streamlit.secrets", {}))
        self.llm = StubLLM()
        self.provider = self.stack.enter_context(patch("src.graph.workflow.get_llm", return_value=self.llm))
        self.search = self.stack.enter_context(patch("src.tools.market_search.fetch_market_results", return_value=RESULTS))
        _cached_web_requirements.clear()
        self.addCleanup(_cached_web_requirements.clear)
        self.app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()

    def prepare(self):
        self.app.text_input[0].set_value("AI Engineer")
        self.app.checkbox[0].check()

    def test_report_survives_widget_reruns_without_new_requests(self):
        self.prepare()
        self.app.button[0].click().run()
        self.assertFalse(self.app.exception)
        saved = self.app.session_state["analysis"]["markdown"]
        calls = len(self.llm.calls)
        self.app.selectbox[0].set_value("english").run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.session_state["analysis"]["markdown"], saved)
        self.assertTrue(any(item.value.strip() == saved.strip() for item in self.app.markdown))
        self.assertEqual(len(self.llm.calls), calls)
        self.assertEqual(len(self.app.get("download_button")), 1)

    def test_manual_mode_requires_description_before_requests(self):
        self.prepare()
        self.app.selectbox(key="market_source").set_value("job-description")
        self.app.button[0].click().run()
        self.assertFalse(self.app.exception)
        self.assertTrue(any("Tempel deskripsi lowongan" in item.value for item in self.app.warning))
        self.provider.assert_not_called()
        self.search.assert_not_called()

    def test_missing_credentials_leave_sample_report_available(self):
        self.prepare()
        with patch.dict(os.environ, {"API_KEY": ""}):
            self.app.button[0].click().run()
        self.assertFalse(self.app.exception)
        self.assertTrue(any("Streamlit Secrets" in item.value for item in self.app.error))
        self.assertTrue(any("## Ikhtisar" in item.value for item in self.app.markdown))
        self.provider.assert_not_called()
        self.search.assert_not_called()

    def test_search_fallback_displays_job_description_basis(self):
        self.prepare()
        self.app.text_area[0].set_value("Python and Docker required")
        self.search.side_effect = MarketSearchError("No connection")
        self.app.button[0].click().run()
        self.assertFalse(self.app.exception)
        self.assertFalse(self.app.error)
        self.assertEqual(self.app.session_state["analysis"]["source"], "job_description")
        self.assertTrue(any("deskripsi lowongan cadangan" in item.value for item in self.app.warning))

    def test_failed_rerun_clears_previous_successful_report(self):
        self.prepare()
        self.app.button[0].click().run()
        self.assertIn("analysis", self.app.session_state)
        self.llm.fail = True
        self.app.button[0].click().run()
        self.assertFalse(self.app.exception)
        self.assertNotIn("analysis", self.app.session_state)
        self.assertTrue(self.app.error)
        self.assertFalse(self.app.get("download_button"))


if __name__ == "__main__":
    unittest.main()
