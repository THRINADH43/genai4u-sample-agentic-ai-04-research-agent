"""Provider-independent model layer. MODEL_PROVIDER=fake|groq|openrouter and MODEL_NAME choose
the model; nothing else in the code knows which one is used."""

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

PRICES = {"openai/gpt-oss-20b": (0.075, 0.30), "openai/gpt-oss-120b": (0.15, 0.60)}  # USD per 1M tokens


@dataclass
class Reply:
    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


class ModelError(Exception):
    pass


class FakeModel:
    """Deterministic stand-in. Answers by role, using the prompt's own content, so the whole
    pipeline (plan, research, write, check) can be tested without a key."""

    name = "fake"

    def complete(self, system, prompt):
        if system.startswith("PLAN"):
            q = prompt.split("Question:", 1)[1].strip()
            text = json.dumps({"sub_questions": [q]})
        elif system.startswith("NOTES"):
            lines = [l for l in prompt.splitlines() if l.startswith("[")]
            text = "\n".join(lines[:2]) or "nothing relevant"
        elif system.startswith("WRITE"):
            notes = [l for l in prompt.splitlines() if l.startswith("[")]
            text = "\n".join(f"- {l.split('] ', 1)[1]} {l.split('] ', 1)[0]}]" for l in notes[:3]) or "The documents do not cover this."
        elif system.startswith("CHECK"):
            text = json.dumps({"ok": True, "problems": []})
        else:
            text = "ok"
        return Reply(text, len(prompt) // 4, len(text) // 4, 0.0)


class OpenAICompatibleModel:
    URLS = {"groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
            "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY")}

    def __init__(self, provider, model, timeout=60, retries=2):
        base, key_var = self.URLS[provider]
        self.url, self.key = base + "/chat/completions", os.environ[key_var]
        self.name, self.model, self.timeout, self.retries = provider, model, timeout, retries

    def complete(self, system, prompt):
        body = {"model": self.model, "temperature": 0, "reasoning_effort": "low",
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]}
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json",
                                              "User-Agent": "research-agent/1.0"})
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    data = json.load(r)
                break
            except (urllib.error.URLError, TimeoutError) as err:
                code = getattr(err, "code", None)
                if (code is not None and code != 429 and code < 500) or attempt == self.retries:
                    raise ModelError(str(err)) from err
                time.sleep(2 ** attempt)
        u = data["usage"]
        pin, pout = PRICES.get(self.model, (0.0, 0.0))
        cost = u["prompt_tokens"] / 1e6 * pin + u["completion_tokens"] / 1e6 * pout
        return Reply(data["choices"][0]["message"]["content"] or "", u["prompt_tokens"], u["completion_tokens"], cost)


def model_from_env(env=os.environ):
    provider = env.get("MODEL_PROVIDER", "fake")
    if provider == "fake":
        return FakeModel()
    return OpenAICompatibleModel(provider, env.get("MODEL_NAME", "openai/gpt-oss-20b"))
