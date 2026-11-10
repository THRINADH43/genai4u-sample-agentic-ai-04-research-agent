import json
import os
import threading
import time
import unittest
import urllib.request
from http.server import ThreadingHTTPServer

from research.agent import BudgetExceeded, Job, StepLimit, citation_problems, run_job
from research.models import FakeModel, Reply, model_from_env
from research.service import JobRunner, make_handler
from research.tools import DocSearch

DOCS = os.path.join(os.path.dirname(__file__), "..", "docs")


class ScriptedModel:
    """Returns given texts in order, for testing the revision and limits."""

    name = "scripted"

    def __init__(self, texts, cost=0.0):
        self.texts, self.cost = list(texts), cost

    def complete(self, system, prompt):
        return Reply(self.texts.pop(0) if self.texts else "{}", 10, 10, self.cost)


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.search = DocSearch(DOCS)

    def test_full_run_with_fake_model_cites_read_documents(self):
        job = Job("How much solar power has India installed?")
        out = run_job(job, FakeModel(), self.search)
        self.assertTrue(out["check_passed"])
        self.assertIn("solar-india.md", out["sources"])
        self.assertTrue(set(out["sources"]) <= job.read_docs)
        self.assertEqual([t["step"] for t in job.trace if t["step"] != "search"][:2], ["plan", "research"])

    def test_invented_source_triggers_one_revision(self):
        model = ScriptedModel([
            '{"sub_questions": ["solar capacity"]}',
            "[solar-india.md] India had about 90 GW",
            "- India has 90 GW of solar [made-up.md]",        # draft cites a document never read
            "- India has about 90 GW of solar [solar-india.md]",
            '{"ok": true, "problems": []}',
        ])
        job = Job("How much solar?")
        out = run_job(job, model, self.search)
        self.assertTrue(out["check_passed"])
        checks = [t for t in job.trace if t["step"] == "check"]
        self.assertIn("made-up.md", checks[0]["problems"][0])

    def test_citation_problems(self):
        self.assertEqual(citation_problems("- fact [a.md]", {"a.md"}), [])
        self.assertTrue(citation_problems("- fact", {"a.md"}))
        self.assertTrue(citation_problems("- fact [b.md]", {"a.md"}))

    def test_step_limit(self):
        with self.assertRaises(StepLimit):
            run_job(Job("How much solar?", max_steps=2), FakeModel(), self.search)

    def test_budget_per_job(self):
        with self.assertRaises(BudgetExceeded):
            run_job(Job("How much solar?", budget_usd=0.001), ScriptedModel([], cost=0.002), self.search)

    def test_bad_json_from_planner_falls_back(self):
        job = Job("How much solar power?")
        model = ScriptedModel(["not json", "[solar-india.md] 90 GW", "- 90 GW [solar-india.md]", '{"ok": true}'])
        self.assertTrue(run_job(job, model, self.search)["check_passed"])

    def test_model_chosen_by_configuration(self):
        self.assertIsInstance(model_from_env({}), FakeModel)


class ServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = JobRunner(FakeModel(), DocSearch(DOCS))
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(cls.runner))
        cls.url = f"http://127.0.0.1:{cls.server.server_address[1]}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def http(self, method, path, body=None):
        req = urllib.request.Request(self.url + path, method=method, data=json.dumps(body).encode() if body else None)
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_job_lifecycle(self):
        status, body = self.http("POST", "/jobs", {"question": "How many UPI transactions happen each month?"})
        self.assertEqual(status, 202)
        for _ in range(100):
            status, view = self.http("GET", f"/jobs/{body['id']}")
            if view["status"] in ("done", "failed"):
                break
            time.sleep(0.02)
        self.assertEqual(view["status"], "done")
        self.assertIn("upi-growth.md", view["result"]["sources"])
        status, trace = self.http("GET", f"/jobs/{body['id']}/trace")
        self.assertTrue(any(t["step"] == "plan" for t in trace))

    def test_errors(self):
        self.assertEqual(self.http("POST", "/jobs", {"question": "?"})[0], 400)
        self.assertEqual(self.http("GET", "/jobs/nope")[0], 404)

    def test_failed_job_reports_error(self):
        runner = JobRunner(FakeModel(), DocSearch(DOCS), max_steps=1)
        job_id = runner.submit("How much solar?")
        for _ in range(100):
            if runner.status[job_id] in ("done", "failed"):
                break
            time.sleep(0.02)
        self.assertEqual(runner.status[job_id], "failed")
        self.assertIn("StepLimit", runner.view(job_id)["error"])


if __name__ == "__main__":
    unittest.main()
