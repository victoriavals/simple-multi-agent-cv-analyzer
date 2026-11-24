from pathlib import Path
import json
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.http_api import app


class AnalysisApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.addCleanup(self.client.close)

    def test_stream_emits_pipeline_progress_and_removes_uploaded_cv(self):
        captured_paths = []

        def graph(on_progress):
            def run(state):
                path = Path(state.cv_path)
                captured_paths.append(path)
                self.assertEqual(path.read_text(encoding="utf-8"), "Synthetic CV: Python")
                for stage in ("reading", "skills", "requirements", "report"):
                    on_progress(stage)
                state.report_markdown = "## Overview\nSynthetic report"
                state.market_requirements = {"source": "job_description"}
                return state

            return run

        with patch("src.http_api.get_setting", return_value="test-only"), patch("src.http_api.build_graph", graph):
            response = self.client.post(
                "/analyze",
                data={"role": "Engineer", "language": "english", "source": "ddgs", "description": "Needs Python"},
                files={"cv": ("candidate.txt", b"Synthetic CV: Python", "text/plain")},
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn("application/x-ndjson", response.headers["content-type"])
        self.assertEqual(response.headers["cache-control"], "no-store")
        events = [json.loads(line) for line in response.text.splitlines()]
        self.assertEqual([event["stage"] for event in events[:-1]], ["reading", "skills", "requirements", "report"])
        self.assertEqual(events[-1]["type"], "complete")
        self.assertTrue(events[-1]["report"]["usedFallback"])
        self.assertNotIn("Synthetic CV:", response.text)
        self.assertTrue(captured_paths)
        self.assertFalse(captured_paths[0].exists())

    def test_invalid_upload_is_rejected_without_starting_analysis(self):
        with patch("src.http_api.build_graph") as graph:
            response = self.client.post("/analyze", data={"role": "Engineer"}, files={"cv": ("cv.docx", b"test")})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["code"], "unsupported_file")
        graph.assert_not_called()

    def test_unconfigured_provider_returns_actionable_error(self):
        with patch("src.http_api.get_setting", return_value=""), patch("src.http_api.build_graph") as graph:
            response = self.client.post("/analyze", data={"role": "Engineer", "demo": "true"})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["code"], "provider_not_configured")
        graph.assert_not_called()
