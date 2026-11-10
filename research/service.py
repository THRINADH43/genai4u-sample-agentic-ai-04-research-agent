"""HTTP API with background jobs.

POST /jobs        {"question": "..."}  -> 202 {"id", "status": "queued"}
GET  /jobs/<id>   -> 200 {"status", "progress", "steps", "cost_usd", "result", "error"}
GET  /jobs/<id>/trace -> 200 the step-by-step trace
"""

import json
import logging
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from research.agent import Job, run_job
from research.models import model_from_env
from research.tools import DocSearch

log = logging.getLogger("research")


class JobRunner:
    def __init__(self, model, search, workers=2, **limits):
        self.model, self.search, self.limits = model, search, limits
        self.jobs = {}
        self.status = {}
        self.results = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(workers)

    def submit(self, question):
        job_id = uuid.uuid4().hex[:12]
        job = Job(question, **self.limits)
        with self._lock:
            self.jobs[job_id], self.status[job_id] = job, "queued"
        self._pool.submit(self._run, job_id, job)
        return job_id

    def _run(self, job_id, job):
        self.status[job_id] = "running"
        try:
            self.results[job_id] = run_job(job, self.model, self.search)
            self.status[job_id] = "done"
        except Exception as err:  # step limit, budget, model errors: the job fails, the service lives on
            self.results[job_id] = {"error": f"{type(err).__name__}: {err}"}
            self.status[job_id] = "failed"
        job.progress = self.status[job_id]
        log.info("job %s %s in %d steps, $%.5f", job_id, self.status[job_id], job.steps, job.cost)

    def view(self, job_id):
        job = self.jobs.get(job_id)
        if job is None:
            return None
        result = self.results.get(job_id, {})
        return {"id": job_id, "question": job.question, "status": self.status[job_id], "progress": job.progress,
                "steps": job.steps, "tokens": job.tokens, "cost_usd": round(job.cost, 6),
                "result": None if "error" in result else result or None, "error": result.get("error")}


def make_handler(runner):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, status, body):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            if self.path != "/jobs":
                return self._json(404, {"error": "not found"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                question = body["question"].strip()
                assert 5 <= len(question) <= 500
            except (ValueError, KeyError, AttributeError, AssertionError):
                return self._json(400, {"error": "send JSON with a question of 5 to 500 characters"})
            self._json(202, {"id": runner.submit(question), "status": "queued"})

        def do_GET(self):
            parts = self.path.strip("/").split("/")
            if len(parts) >= 2 and parts[0] == "jobs":
                view = runner.view(parts[1])
                if view is None:
                    return self._json(404, {"error": "no such job"})
                if len(parts) == 3 and parts[2] == "trace":
                    return self._json(200, runner.jobs[parts[1]].trace)
                return self._json(200, view)
            self._json(404, {"error": "not found"})

        def log_message(self, *args):
            pass

    return Handler


def main():
    logging.basicConfig(level=logging.INFO)
    runner = JobRunner(model_from_env(), DocSearch("docs"),
                       max_steps=int(os.environ.get("MAX_STEPS", "10")),
                       budget_usd=float(os.environ.get("JOB_BUDGET_USD", "0.01")))
    port = int(os.environ.get("PORT", "8060"))
    print(f"listening on http://127.0.0.1:{port} using {runner.model.name}")
    ThreadingHTTPServer(("127.0.0.1", port), make_handler(runner)).serve_forever()


if __name__ == "__main__":
    main()
