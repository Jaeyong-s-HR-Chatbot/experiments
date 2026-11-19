"""BM25 색인 (numpy만 사용). 조각 수천 개 규모라 메모리에 올려 둬요."""
from __future__ import annotations

import math
from collections import Counter

import numpy as np


class BM25:
    def __init__(self, docs_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.n = len(docs_tokens)
        self.k1, self.b = k1, b
        self.dl = np.array([len(d) for d in docs_tokens], dtype=np.float32)
        self.avgdl = float(self.dl.mean()) if self.n else 0.0
        post: dict[str, list[tuple[int, int]]] = {}
        for i, toks in enumerate(docs_tokens):
            for t, f in Counter(toks).items():
                post.setdefault(t, []).append((i, f))
        self.post = {t: (np.array([i for i, _ in p], dtype=np.int32), np.array([f for _, f in p], dtype=np.float32))
                     for t, p in post.items()}
        self.idf = {t: math.log(1 + (self.n - len(p[0]) + 0.5) / (len(p[0]) + 0.5)) for t, p in self.post.items()}

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        s = np.zeros(self.n, dtype=np.float32)
        for t in dict.fromkeys(query_tokens):
            if t not in self.post:
                continue
            idx, tf = self.post[t]
            norm = tf + self.k1 * (1 - self.b + self.b * self.dl[idx] / (self.avgdl or 1.0))
            s[idx] += self.idf[t] * tf * (self.k1 + 1) / norm
        return s
