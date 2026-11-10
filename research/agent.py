"""Orchestrator-workers with an evaluator check.

1. PLAN: the orchestrator splits the question into at most 3 sub-questions.
2. RESEARCH: one worker per sub-question searches docs/ and writes cited notes.
3. WRITE: the writer turns the notes into a short report; every bullet cites [file].
4. CHECK: code checks every citation is a document that was actually read; the model checks
   the draft answers the question. One revision is allowed if the check fails.
Every step is limited, counted against the job budget, and traced.
"""

import json
import re
import time


class BudgetExceeded(Exception):
    pass


class StepLimit(Exception):
    pass


class Job:
    def __init__(self, question, max_steps=10, budget_usd=0.01, max_tokens=40_000):
        self.question = question
        self.max_steps, self.budget_usd, self.max_tokens = max_steps, budget_usd, max_tokens
        self.steps = 0
        self.cost = 0.0
        self.tokens = 0
        self.trace = []
        self.progress = "queued"
        self.read_docs = set()

    def record(self, step, **data):
        self.trace.append({"t": round(time.time(), 3), "step": step, **data})


def call(job, model, step, system, prompt):
    if job.steps >= job.max_steps:
        raise StepLimit(f"step limit of {job.max_steps} reached")
    if job.cost >= job.budget_usd or job.tokens >= job.max_tokens:
        raise BudgetExceeded(f"budget used: ${job.cost:.5f}, {job.tokens} tokens")
    job.steps += 1
    started = time.monotonic()
    reply = model.complete(system, prompt)
    job.cost += reply.cost_usd
    job.tokens += reply.input_tokens + reply.output_tokens
    job.record(step, input_tokens=reply.input_tokens, output_tokens=reply.output_tokens,
               cost_usd=round(reply.cost_usd, 6), ms=int((time.monotonic() - started) * 1000),
               output=reply.text[:300])
    return reply.text


def parse_json(text, default):
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0)) if m else default
    except json.JSONDecodeError:
        return default


PLAN = 'PLAN: Split the research question into 1 to 3 short sub-questions. Reply JSON: {"sub_questions": [...]}'
NOTES = ("NOTES: From the documents below, copy the facts that help answer the sub-question. One fact per "
         "line, starting with the document name in brackets, like [file.md] fact. Use only these documents.")
WRITE = ("WRITE: Write a short report answering the question as 2 to 5 bullet points. End every bullet with "
         "the [file.md] it came from, using only the notes. If the notes do not answer, say so.")
CHECK = 'CHECK: Does the report answer the question using the notes? Reply JSON: {"ok": true|false, "problems": [...]}'


def cited_docs(report):
    return set(re.findall(r"\[([\w.-]+\.(?:md|txt))\]", report))


def citation_problems(report, read_docs):
    """Done in code, not by the model: each bullet needs a citation, and every citation must
    be a document this job actually read."""
    problems = []
    bullets = [l for l in report.splitlines() if l.strip().startswith(("-", "*"))]
    if not bullets:
        problems.append("the report has no bullet points")
    for b in bullets:
        if not cited_docs(b):
            problems.append(f"uncited bullet: {b[:60]}")
    for d in cited_docs(report) - read_docs:
        problems.append(f"cites {d}, which was never read")
    return problems


def run_job(job, model, search):
    job.progress = "planning"
    plan = parse_json(call(job, model, "plan", PLAN, f"Question: {job.question}"), {"sub_questions": [job.question]})
    subs = [s for s in plan.get("sub_questions", []) if isinstance(s, str)][:3] or [job.question]

    notes = []
    for i, sub in enumerate(subs, 1):
        job.progress = f"researching {i}/{len(subs)}"
        hits = search.search(sub)
        job.record("search", query=sub, found=[n for n, _ in hits])
        job.read_docs.update(n for n, _ in hits)
        if not hits:
            continue
        docs = "\n".join(f"[{n}] {t}" for n, t in hits)
        notes.append(call(job, model, "research", NOTES, f"Sub-question: {sub}\n\nDocuments:\n{docs}"))

    notes_text = "\n".join(notes)
    report = ""
    for attempt in range(2):  # the first draft plus at most one revision
        job.progress = "writing" if attempt == 0 else "revising"
        feedback = f"\n\nFix these problems from the last draft: {problems}" if attempt else ""
        report = call(job, model, "write", WRITE, f"Question: {job.question}\n\nNotes:\n{notes_text}{feedback}")
        job.progress = "checking"
        problems = citation_problems(report, job.read_docs)
        if not problems:
            verdict = parse_json(call(job, model, "check", CHECK, f"Question: {job.question}\n\nNotes:\n{notes_text}\n\nReport:\n{report}"),
                                 {"ok": False, "problems": ["the checker's reply was not JSON"]})
            problems = [] if verdict.get("ok") else [str(p) for p in verdict.get("problems", [])] or ["the checker said no"]
        job.record("check", attempt=attempt + 1, problems=problems)
        if not problems:
            break
    sources = sorted(cited_docs(report) & job.read_docs)
    return {"report": report, "sources": sources, "check_passed": not problems, "problems": problems}
