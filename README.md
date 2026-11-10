# Multi-step research agent service

A service that takes a research question, plans the work, searches a local folder of
documents (`docs/`, 11 short notes on Indian energy, payments and farming), writes a report
with sources, checks its own draft, and returns the report. Standard library only.

## Run

```
python -m research.service                                        # fake model, no key
MODEL_PROVIDER=groq GROQ_API_KEY=... python -m research.service   # real model
```

```
curl -X POST localhost:8060/jobs -d '{"question": "How many UPI transactions happen each month?"}'
# 202 {"id": "a1b2c3d4e5f6", "status": "queued"}
curl localhost:8060/jobs/a1b2c3d4e5f6
# {"status": "running", "progress": "researching 2/3", "steps": 3, "cost_usd": 0.0001, ...}
# {"status": "done", "result": {"report": "- UPI handled more than 13 billion transactions a month in 2024. [upi-growth.md]",
#   "sources": ["upi-growth.md"], "check_passed": true}, ...}
curl localhost:8060/jobs/a1b2c3d4e5f6/trace
```

| Variable | Default | |
|----------|---------|---|
| `MODEL_PROVIDER` | `fake` | `fake`, `groq` or `openrouter` |
| `MODEL_NAME` | `openai/gpt-oss-20b` | |
| `MAX_STEPS` | `10` | model calls per job |
| `JOB_BUDGET_USD` | `0.01` | spending limit per job |

## Design: orchestrator-workers with an evaluator check

1. **Plan:** the orchestrator splits the question into 1 to 3 sub-questions.
2. **Research:** one worker per sub-question searches `docs/` and copies facts as cited notes.
3. **Write:** the writer turns the notes into 2 to 5 bullets, each ending with `[file.md]`.
4. **Check:** first in code (every bullet has a citation, and every cited file was actually
   read by this job), then by the model (does the report answer the question?). If either
   fails, the writer gets the problems and revises once.

**Why this pattern:** a research question often has parts ("how much solar, and what is the
target?"), and orchestrator-workers handles each part separately. A full evaluator-optimizer
loop would keep rewriting until a judge is happy, which costs more and can loop; one checked
revision catches the common problem (invented or missing sources) at a fixed cost. The
citation check is code, not a model, because a model judging its own sources is easy to fool.

Sources in a report can only be documents this job read; invented ones are rejected.

## Reliability

- **Step limit:** each model call is a step; past `MAX_STEPS` the job fails with `StepLimit`.
- **Budget per job:** cost and tokens are counted after every call; past the limit the job
  fails with `BudgetExceeded`.
- **Retries:** timeouts, HTTP 429 and 5xx are retried twice with backoff; other errors fail.
- **Bad model output:** if the planner or checker does not reply with JSON, the agent falls
  back (use the question as the only sub-question; treat the check as failed).
- A failing job never stops the service: it is marked `failed` with the error.

## Model layer

`research/models.py`: `FakeModel` and `OpenAICompatibleModel` share one `complete` method;
`model_from_env` picks one from `MODEL_PROVIDER`, so changing the model is configuration only.

## Tracing

Each job keeps a trace: every step with tokens, cost, milliseconds and the start of the
output, plus each search and check. `GET /jobs/<id>/trace` returns it.

## Evaluation

`eval/questions.json`: 8 questions, each with facts the report must contain and the document
it must cite. `python evaluate.py` runs them and saves `results/eval_<model>.json`.

| model | passed | cost |
|-------|--------|------|
| fake (CI, `--min 0.75`) | 8/8 | $0 |
| Groq `openai/gpt-oss-20b` | 7/8 | $0.0017 for all 8 (about 6 steps each) |

**The Groq failure:** "Which states lead in wind power?" The planner rewrote it as a question
about U.S. states, so the search found nothing and the report said so. Giving the planner the
list of document titles would fix this.

**A bug I found in my own evaluation:** the first Groq run scored 5/8 because the model wrote
`90 GW` with a narrow no-break space, so a plain substring check failed. The evaluator now
normalises Unicode (NFKC) and whitespace before comparing.

## Tests

`python -m unittest discover -s tests -t .`, fake models only: a full run, a revision caused
by an invented source, citation checks, step limit, budget, bad JSON from the planner,
configuration, the HTTP job lifecycle with progress and trace, and failed jobs. GitHub
Actions runs the tests and the evaluation with the fake model on every push.
