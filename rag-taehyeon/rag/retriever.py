"""④ 검색기: (필터) → Dense + BM25 → RRF → 조항 단위로 합치기.

필터 (구성 B·C)
- hard: 질문 카테고리(또는 '공통')와 하나라도 겹치는 조각만 남겨요. DB의 `categories && ARRAY[...]`와 같은 규칙이에요.
- adaptive: 분류를 믿을 만하면 hard, 아니면 soft(겹치는 조각의 점수만 올림).
  hard 결과가 너무 적거나 1등 점수가 낮으면 필터를 풀고 다시 검색해 더 나은 쪽을 써요.
"""
from __future__ import annotations

import hashlib
import logging
import pickle
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .bm25 import BM25
from .embedder import doc_text
from .reranker import CrossReranker, colbert_scores
from .text import COMMON, tokenize, tokenizer_name

log = logging.getLogger("rag")


@dataclass
class Candidate:
    chunk_id: str
    rank: int
    score: float                 # RRF 점수 (필터·가중 반영)
    cos: float | None            # 질문과의 코사인 (Dense가 꺼져 있으면 None)
    bm25: float | None
    dense_rank: int | None
    bm25_rank: int | None
    doc_kind: str
    categories: list[str]
    matched: bool                # 질문 카테고리와 겹치는지
    rerank: float | None = None  # 재정렬 점수 (재정렬을 안 했으면 None)

    def to_dict(self):
        return asdict(self)


@dataclass
class SearchResult:
    candidates: list[Candidate] = field(default_factory=list)
    used_filter: str = "none"            # none / hard / soft
    filter_categories: list[str] = field(default_factory=list)
    used_fallback: bool = False
    top_cos: float | None = None
    top_bm25: float | None = None
    doc_kind_counts: dict[str, int] = field(default_factory=dict)
    dense: bool = False
    pool: list[str] = field(default_factory=list)   # 재정렬 전 후보 순서 (chunk_id)
    rerank: str = "none"
    search_ms: float = 0.0
    rerank_ms: float = 0.0

    def to_dict(self):
        return asdict(self)


class Retriever:
    def __init__(self, cfg, rows: list[dict], emb: np.ndarray | None, embedder=None):
        self.cfg, self.embedder = cfg, embedder
        self.list = rows
        self.rows = {r["chunk_id"]: r for r in rows}        # chunk_id → 조각
        self.ids = [r["chunk_id"] for r in rows]
        self.emb = emb if (emb is not None and embedder is not None) else None
        if self.emb is not None and self.emb.shape[1] != embedder.encode(["확인"]).shape[1]:
            self.emb = None                                  # 저장된 임베딩과 모델 차원이 다르면 Dense를 꺼요
        self.bm25 = BM25(self._tokens())
        self.is_internal = np.array([r["doc_kind"] == "내부규정" for r in rows])
        self.group = [f'{r["document_id"]}|{r["article"] or r["chunk_id"]}' for r in rows]
        self._cross: dict[str, object] = {}

    def set_cross(self, model_name: str, reranker) -> None:
        """크로스 인코더를 직접 넣어요 (테스트, 미리 불러온 모델 재사용)"""
        self._cross[model_name] = reranker

    def rerank(self, cfg, text: str, cands: list[Candidate]) -> list[Candidate]:
        """후보 앞쪽 rerank_pool개를 다시 정렬해요. 나머지는 원래 순서대로 뒤에 둬요."""
        pool, rest = cands[:cfg.rerank_pool], cands[cfg.rerank_pool:]
        if not pool:
            return cands
        docs = [doc_text(self.rows[c.chunk_id]) for c in pool]
        if cfg.rerank == "colbert":
            if not hasattr(self.embedder, "colbert"):
                raise RuntimeError("colbert 재정렬은 FlagEmbedding(BGE-M3)이 필요해요. pip install FlagEmbedding")
            scores = colbert_scores(self.embedder, text, docs)
        elif cfg.rerank == "cross":
            if cfg.rerank_model not in self._cross:
                t0 = time.time()
                log.info("재정렬 모델 불러오는 중: %s. 처음이면 약 2.3GB를 내려받아요.", cfg.rerank_model)
                self._cross[cfg.rerank_model] = CrossReranker(cfg.rerank_model, cfg.max_len, cfg.batch_size)
                log.info("재정렬 모델 준비 완료: %.0f초", time.time() - t0)
            scores = self._cross[cfg.rerank_model].score(text, docs)
        else:
            raise ValueError(f"rerank는 none / colbert / cross 중 하나여야 해요: {cfg.rerank}")
        w = cfg.rerank_dense_weight
        for c, s in zip(pool, scores):
            c.rerank = round(float(s if (w <= 0 or c.cos is None) else w * c.cos + (1 - w) * s), 4)
        pool = sorted(pool, key=lambda c: -c.rerank)
        out = pool + rest
        for i, c in enumerate(out):
            c.rank = i + 1
        return out

    @property
    def dense_ready(self) -> bool:
        return self.emb is not None

    def _tokens(self) -> list[list[str]]:
        """형태소 분석은 오래 걸려서 결과를 저장해 두고 다시 써요."""
        texts = [f'{r["doc_name"]} {r["article"]} {r["article_title"]}\n{r["text"]}' for r in self.list]
        key = hashlib.md5((tokenizer_name() + "".join(self.ids) + str(sum(map(len, texts)))).encode()).hexdigest()[:16]
        path = Path(self.cfg.cache_dir) / f"tokens_{key}.pkl"
        if path.exists():
            try:
                toks = pickle.loads(path.read_bytes())
                log.info("저장해 둔 형태소 분석 결과를 써요 (%s)", tokenizer_name())
                return toks
            except Exception:
                pass
        log.info("BM25 색인용 형태소 분석 중: 조각 %d개 (%s, 처음 한 번)", len(texts), tokenizer_name())
        toks, t0 = [], time.time()
        for i, t in enumerate(texts, 1):
            toks.append(tokenize(t))
            if i % 500 == 0 or i == len(texts):
                log.info("  형태소 분석 %d/%d · 경과 %.0f초", i, len(texts), time.time() - t0)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(pickle.dumps(toks))
        except OSError:
            pass
        return toks

    def _mask(self, cats: list[str]) -> np.ndarray:
        want = set(cats) | {COMMON}
        return np.array([bool(want & set(r["categories"])) for r in self.list])

    def _rank(self, cfg, dense, bm, cats, mode) -> list[Candidate]:
        n = len(self.list)
        matched = self._mask(cats) if cats else np.ones(n, dtype=bool)
        allowed = matched if mode == "hard" else np.ones(n, dtype=bool)
        rrf = np.zeros(n, dtype=np.float64)
        ranks = {}
        for name, sc in (("dense", dense), ("bm25", bm)):
            if sc is None:
                continue
            s = np.where(allowed, sc, -np.inf)
            order = np.argsort(-s)[:cfg.n_candidates]
            order = [i for i in order if np.isfinite(s[i]) and (name == "dense" or s[i] > 0)]
            ranks[name] = {i: k + 1 for k, i in enumerate(order)}
            for k, i in enumerate(order):
                rrf[i] += 1.0 / (cfg.rrf_k + k + 1)
        if mode == "soft" and cats:
            rrf = np.where(matched, rrf * (1 + cfg.soft_boost), rrf)
        if cfg.internal_boost:
            rrf = np.where(self.is_internal, rrf * (1 + cfg.internal_boost), rrf)
        out, seen = [], set()
        for i in np.argsort(-rrf):
            if rrf[i] <= 0 or len(out) >= cfg.n_final:
                break
            if self.group[i] in seen:                        # 같은 조항의 조각은 가장 높은 것 하나만
                continue
            seen.add(self.group[i])
            r = self.list[i]
            out.append(Candidate(
                chunk_id=r["chunk_id"], rank=len(out) + 1, score=round(float(rrf[i]), 5),
                cos=round(float(dense[i]), 4) if dense is not None else None,
                bm25=round(float(bm[i]), 3) if bm is not None else None,
                dense_rank=ranks.get("dense", {}).get(i), bm25_rank=ranks.get("bm25", {}).get(i),
                doc_kind=r["doc_kind"], categories=r["categories"], matched=bool(matched[i]) if cats else False))
        return out

    def cos(self, text: str, chunk_ids: list[str]) -> list[float]:
        """질문과 조각의 코사인 (근거 판단용). Dense가 꺼져 있으면 빈 목록."""
        if not self.dense_ready or not chunk_ids:
            return []
        qv = self.embedder.encode([text])[0]
        idx = [self.ids.index(c) for c in chunk_ids]
        return (self.emb[idx] @ qv).tolist()

    def search(self, text: str, query_class=None, cfg=None) -> SearchResult:
        cfg = cfg or self.cfg
        t0 = time.perf_counter()
        use_dense = cfg.use_dense and self.dense_ready
        dense = self.emb @ self.embedder.encode([text])[0] if use_dense else None
        bm = self.bm25.scores(tokenize(text)) if (cfg.use_bm25 or not use_dense) else None   # Dense가 없으면 BM25로 대신
        cats = [c for c, _ in query_class.categories] if query_class else []
        mode = cfg.filter
        if mode == "adaptive":
            mode = ("hard" if query_class.confident else "soft") if cats else "none"
        if mode == "hard" and not cats:
            mode = "none"
        cands = self._rank(cfg, dense, bm, cats, mode)
        res = SearchResult(candidates=cands, used_filter=mode, filter_categories=cats if mode != "none" else [], dense=use_dense)

        def top(cs):
            if not cs:
                return -1.0
            return max(c.cos for c in cs[:3]) if use_dense else max(c.bm25 or 0 for c in cs[:3])

        if cfg.filter == "adaptive" and mode == "hard":
            weak = len(cands) < cfg.min_filtered or (use_dense and top(cands) < cfg.th_answer)
            if weak:
                free = self._rank(cfg, dense, bm, cats, "soft")
                if top(free) > top(cands) or len(cands) < cfg.min_filtered:
                    res.candidates, res.used_filter, res.used_fallback = free, "soft", True
        res.pool = [c.chunk_id for c in res.candidates]
        res.search_ms = round((time.perf_counter() - t0) * 1000, 2)
        if cfg.rerank != "none":
            t1 = time.perf_counter()
            res.candidates = self.rerank(cfg, text, res.candidates)
            res.rerank, res.rerank_ms = cfg.rerank, round((time.perf_counter() - t1) * 1000, 2)
        cs = res.candidates
        res.top_cos = max((c.cos for c in cs[:3]), default=None) if use_dense else None
        res.top_bm25 = max((c.bm25 or 0 for c in cs[:3]), default=None) if bm is not None else None
        res.doc_kind_counts = dict(Counter(c.doc_kind for c in cs))
        return res
