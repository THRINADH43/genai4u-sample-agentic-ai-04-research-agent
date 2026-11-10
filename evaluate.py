"""Runs eval/questions.json through the agent. A question passes when the report contains every
expected fact, cites the expected document, and passed its own check.

MODEL_PROVIDER=groq python evaluate.py      # real model
python evaluate.py --min 0.75              # exit 1 below 75% (CI, fake model)"""

import argparse
import json
import os
import re
import unicodedata

from research.agent import Job, run_job
from research.models import model_from_env
from research.tools import DocSearch


def normalise(text):
    """Models often write '90\u202fGW' (a narrow no-break space) or non-breaking hyphens.
    NFKC turns those into plain characters, then all whitespace becomes one space."""
    text = unicodedata.normalize("NFKC", text).replace("\u2011", "-")
    return re.sub(r"\s+", " ", text).lower()


def evaluate(model, search, cases):
    rows, cost = [], 0.0
    for case in cases:
        job = Job(case["question"], budget_usd=0.02)
        try:
            out = run_job(job, model, search)
        except Exception as err:
            out = {"report": "", "sources": [], "check_passed": False, "error": str(err)}
        report = normalise(out["report"])
        ok = (all(normalise(f) in report for f in case["must_contain"]) and set(case["must_cite"]) <= set(out["sources"])
              and out["check_passed"])
        cost += job.cost
        rows.append({"question": case["question"], "ok": ok, "steps": job.steps, "cost_usd": round(job.cost, 6),
                     "sources": out["sources"], "report": out["report"][:300]})
    return rows, cost


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--min", type=float, default=None)
    args = p.parse_args()
    model = model_from_env()
    rows, cost = evaluate(model, DocSearch("docs"), json.load(open("eval/questions.json")))
    for r in rows:
        print(("PASS" if r["ok"] else "FAIL"), f"{r['steps']} steps ${r['cost_usd']:.5f}", r["question"], r["sources"])
    score = sum(r["ok"] for r in rows) / len(rows)
    print(f"\n{sum(r['ok'] for r in rows)}/{len(rows)} passed with {model.name}, total cost ${cost:.5f}")
    os.makedirs("results", exist_ok=True)
    json.dump({"model": model.name, "score": score, "cost_usd": cost, "rows": rows},
              open(f"results/eval_{model.name}.json", "w"), indent=1)
    if args.min is not None and score < args.min:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
