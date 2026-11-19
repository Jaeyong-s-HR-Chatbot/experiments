"""재정렬: 검색이 뽑은 후보를 질문과 다시 비교해 순서를 고쳐요.

- colbert: BGE-M3의 토큰 벡터로 비교 (질문 토큰마다 가장 가까운 조각 토큰의 유사도를 평균)
- cross  : 크로스 인코더(bge-reranker-v2-m3)가 질문과 조각을 함께 읽고 점수를 매김
"""
from __future__ import annotations

import numpy as np


def maxsim(q: np.ndarray, d: np.ndarray) -> float:
    """ColBERT 점수: 질문 토큰마다 조각에서 가장 가까운 토큰의 유사도를 구해 평균 (0~1 근처)"""
    if q.size == 0 or d.size == 0:
        return 0.0
    sim = q.astype(np.float32) @ d.astype(np.float32).T
    return float(sim.max(axis=1).mean())


def colbert_scores(embedder, query: str, docs: list[str]) -> list[float]:
    qv = embedder.colbert([query])[0]
    return [maxsim(qv, dv) for dv in embedder.colbert(docs)]


class CrossReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", max_len: int = 1024, batch_size: int = 16):
        from sentence_transformers import CrossEncoder
        from .embedder import _device
        self.device = _device()
        self.model = CrossEncoder(model_name, max_length=max_len, device=self.device)
        if self.device == "cuda":
            for target in (self.model, getattr(self.model, "model", None)):   # 반정밀도: 빠르고 메모리를 덜 써요
                try:
                    target.half()
                    break
                except Exception:
                    continue
        self.name, self.bs = model_name, batch_size

    def score(self, query: str, docs: list[str]) -> list[float]:
        s = np.asarray(self.model.predict([(query, d) for d in docs], batch_size=self.bs, show_progress_bar=False), dtype=np.float32)
        if s.min() < 0 or s.max() > 1:           # 점수가 로짓이면 0~1로 바꿔요
            s = 1 / (1 + np.exp(-s))
        return s.tolist()
