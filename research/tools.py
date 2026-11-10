"""The only tool: search the local docs/ folder. Sources in reports come only from here."""

import os
import re

STOP = {"the", "a", "an", "of", "in", "and", "is", "are", "to", "for", "how", "what", "does", "do", "by", "much"}


def words(text):
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP and len(w) > 1}


class DocSearch:
    def __init__(self, folder="docs"):
        self.docs = {n: open(os.path.join(folder, n), encoding="utf-8").read().strip()
                     for n in sorted(os.listdir(folder)) if n.endswith((".md", ".txt"))}

    def search(self, query, k=3):
        q = words(query)
        scored = [(len(q & words(t)), n) for n, t in self.docs.items()]
        return [(n, self.docs[n]) for s, n in sorted(scored, key=lambda x: (-x[0], x[1])) if s > 0][:k]
