"""질문 하나를 ① 쪼개기 → ② 분류 → ③ 라우터 → ④ 검색 → LlmRequest 까지 처리해요. LLM은 부르지 않아요.

status
- ok            : 근거 조각을 찾음 (chunks, llm_request 사용)
- hr_only       : 본인 기록만 보면 되는 질문 (검색 안 함, hr_needed=True)
- clarify       : 조건이 빠져 되물어야 함 (message에 되묻는 문장)
- out_of_scope  : 인사·복리후생 범위 밖
- sensitive     : 고충·괴롭힘 등 담당자 연결
- team_scope    : 팀·부서 단위 조회 (권한 필요)
- no_evidence   : 검색했지만 근거가 약함
"""
from __future__ import annotations

import copy
import hashlib
import logging
from dataclasses import asdict, dataclass, field

from . import store
from .classifier import Classifier, QueryClass, route
from .embedder import doc_text, load_embedder
from .retriever import Retriever, SearchResult
from .splitter import carry_over, rewrite, split_question
from .text import CATEGORY_KEYWORDS, MSG, normalize

LAW_SOURCE = "법제처 국가법령정보센터"
log = logging.getLogger("rag")


@dataclass
class ChunkHit:
    rank: int
    chunk_id: str
    title: str
    score: float                 # 0~1. Dense가 있으면 코사인, 없으면 BM25를 1등 기준으로 나눈 값
    categories: list[str]
    doc_kind: str
    text: str
    effective_date: str
    source: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class LlmRequest:
    question: str
    sub_questions: list[str]
    slots: dict[str, str]
    chunks: list[ChunkHit]
    hr_needed: bool
    notes: list[str] = field(default_factory=list)


@dataclass
class SubResult:
    question: str
    rewritten: str
    query_class: QueryClass
    route: str
    search: SearchResult | None = None
    expanded: bool = False


@dataclass
class PipelineResult:
    question: str
    status: str
    message: str | None
    subs: list[SubResult]
    chunks: list[ChunkHit]
    llm_request: LlmRequest | None
    hr_needed: bool
    pool_ids: list[str] = field(default_factory=list)    # 재정렬 전 후보 순서 (쪼갠 질문이 여러 개면 번갈아 합침)
    search_ms: float = 0.0
    rerank_ms: float = 0.0


def _round_robin(lists: list[list], limit: int, key=lambda x: x) -> list:
    out, seen, i = [], set(), 0
    while len(out) < limit and any(i < len(l) for l in lists):
        for l in lists:
            if i < len(l) and key(l[i]) not in seen and len(out) < limit:
                seen.add(key(l[i]))
                out.append(l[i])
        i += 1
    return out


def embed_chunks(cfg, rows, embedder):
    """조각 임베딩을 만들어요 (DB 없이 파일로 돌릴 때). 결과는 cache_dir에 저장하고 다시 써요."""
    import numpy as np
    from pathlib import Path
    texts = [doc_text(r) for r in rows]
    key = hashlib.md5((getattr(embedder, "name", "") + str(cfg.max_len) + "".join(r["chunk_id"] for r in rows)
                       + str(sum(map(len, texts)))).encode()).hexdigest()[:16]
    path = Path(cfg.cache_dir) / f"emb_{key}.npy"
    if path.exists():
        log.info("저장해 둔 조각 임베딩을 써요: %s", path.name)
        return np.load(path)
    import time
    log.info("조각 %d개를 임베딩해요 (처음 한 번, 다음부터는 저장해 둔 걸 써요)", len(texts))
    t0, parts, step = time.time(), [], 128
    for i in range(0, len(texts), step):
        parts.append(embedder.encode(texts[i:i + step]))
        done = min(i + step, len(texts))
        el = time.time() - t0
        log.info("  임베딩 %d/%d · 경과 %.0f초 · 남은 시간 약 %.0f초", done, len(texts), el, el / done * (len(texts) - done))
    mat = np.concatenate(parts).astype(np.float32)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, mat)
    except OSError:
        pass
    return mat


class Pipeline:
    def __init__(self, cfg, embedder="auto", rows=None, emb=None):
        self.cfg = cfg
        self.embedder = load_embedder(cfg) if embedder == "auto" else embedder
        if rows is None:
            rows, emb, self.source = store.load(cfg)
            log.info("조각 %d개를 읽었어요 (%s)", len(rows), self.source)
        else:
            self.source = "memory"
        if emb is None and self.embedder is not None and self.source != "memory":
            emb = embed_chunks(cfg, rows, self.embedder)
            self.source += "+embedded"
        self.retriever = Retriever(cfg, rows, emb, self.embedder)
        self.classifier = Classifier(cfg, self.embedder)

    def with_config(self, cfg) -> "Pipeline":
        """모델과 색인은 그대로 두고 설정만 바꾼 파이프라인"""
        p = copy.copy(self)
        p.cfg = cfg
        p.classifier = copy.copy(self.classifier)
        p.classifier.cfg = cfg
        return p

    def reset_query_cache(self, keep_docs: bool = True) -> None:
        """시간을 공정하게 재려고 질문 쪽 저장분을 비워요. keep_docs=False면 조각의 ColBERT 벡터도 비워요."""
        e = self.embedder
        if hasattr(e, "_cache"):
            e._cache.clear()
        if hasattr(e, "_col"):
            docs = {doc_text(r) for r in self.retriever.list} if keep_docs else set()
            e._col = {k: v for k, v in e._col.items() if k in docs}

    def info(self) -> dict:
        return {"chunks": len(self.retriever.rows), "source": self.source, "dense": self.retriever.dense_ready,
                "embedder": getattr(self.embedder, "name", None), "device": getattr(self.embedder, "device", None),
                "preset": self.cfg.preset_name, "rerank": self.cfg.rerank}

    def _hit(self, c, rank, top_bm) -> ChunkHit:
        r = self.retriever.rows[c.chunk_id]
        art = r["article"]
        title = f'{r["doc_name"]} {art}'.strip() + (f'({r["article_title"][:20]})' if r["article_title"] and art.startswith("제") else "")
        score = c.cos if c.cos is not None else ((c.bm25 or 0) / top_bm if top_bm else 0.0)
        return ChunkHit(rank=rank, chunk_id=c.chunk_id, title=title, score=round(max(0.0, min(1.0, score)), 4),
                        categories=r["categories"], doc_kind=r["doc_kind"], text=r["text"], effective_date=r["effective_date"],
                        source=LAW_SOURCE if r["doc_kind"] == "법령" else "")

    def _search(self, sub: SubResult) -> bool:
        """검색하고 근거가 충분한지 돌려줘요. 약하면 카테고리 키워드를 붙여 한 번 더 찾아요."""
        cfg, rt = self.cfg, self.retriever
        sub.search = rt.search(sub.rewritten, sub.query_class, cfg)
        if not sub.search.candidates:
            return False
        if sub.search.top_cos is None or sub.search.top_cos >= cfg.th_answer:
            return True                                    # Dense가 없으면 점수로 막지 않아요
        cats = [c for c, _ in sub.query_class.categories][:2]
        extra = [w for c in cats for w in CATEGORY_KEYWORDS.get(c, [])[:3] if w not in sub.rewritten]
        if extra:
            again = rt.search(" ".join([sub.rewritten] + extra), sub.query_class, cfg)
            # 통과 여부는 키워드를 붙인 질의가 아니라 원래 질문 기준 점수로 봐요 (점수가 부풀지 않게)
            cos = rt.cos(sub.rewritten, [c.chunk_id for c in again.candidates[:3]])
            if cos and max(cos) >= cfg.th_answer:
                again.top_cos, sub.search, sub.expanded = round(max(cos), 4), again, True
                return True
        return False

    def run(self, question: str, history: list[str] | None = None, force_search: bool = False) -> PipelineResult:
        """force_search=True 이면 라우터 결과와 상관없이 검색해요 (검색기만 평가하거나, 분류를 다른 곳에서 할 때)."""
        q = normalize(question)
        carried = carry_over(q, history)
        base = " ".join(carried + [q])
        subs = []
        for part in split_question(base):
            rw = rewrite(part)
            qc = self.classifier.classify(rw, part)
            subs.append(SubResult(question=part, rewritten=rw, query_class=qc, route="search" if force_search else route(qc)))
        hr_needed = any(s.query_class.first_person or s.query_class.personal for s in subs)
        routes = [s.route for s in subs]

        def done(status, message=None, chunks=None, req=None, hr=hr_needed):
            return PipelineResult(question, status, message, subs, chunks or [], req, hr)

        for st in ("sensitive", "team_scope"):
            if st in routes:
                return done(st, MSG[st], hr=(st == "team_scope") or hr_needed)
        if "search" not in routes:
            if "clarify" in routes:
                return done("clarify", next(s.query_class.ask for s in subs if s.route == "clarify"))
            if "hr_only" in routes:
                return done("hr_only", MSG["hr_only"], hr=True)
            return done("out_of_scope", MSG["out_of_scope"])

        searched = [s for s in subs if s.route == "search"]
        ok = [s for s in searched if self._search(s)]
        if not ok:
            return done("no_evidence", MSG["no_evidence"])
        lists = []
        for s in ok:
            top_bm = max((c.bm25 or 0 for c in s.search.candidates), default=0)
            lists.append([(c, top_bm) for c in s.search.candidates])
        picked = _round_robin(lists, self.cfg.top_k, key=lambda x: x[0].chunk_id)       # 조각마다 번갈아 합침
        chunks = [self._hit(c, i + 1, top_bm) for i, (c, top_bm) in enumerate(picked)]
        pool_ids = _round_robin([s.search.pool for s in ok], self.cfg.n_final)
        slots = {k: v for s in subs for k, v in s.query_class.slots.items()}
        notes = []
        if any(c.doc_kind == "법령" for c in chunks):
            notes.append(f"법령을 근거로 답할 때는 출처({LAW_SOURCE})를 함께 적어요.")
        if any(s.route == "hr_only" for s in subs) or hr_needed:
            notes.append("본인 기록이 필요한 부분은 인사 DB 조회 결과를 함께 써요.")
        req = LlmRequest(question=question, sub_questions=[s.question for s in subs], slots=slots, chunks=chunks,
                         hr_needed=hr_needed, notes=notes)
        res = done("ok", None, chunks, req)
        res.pool_ids = pool_ids
        res.search_ms = round(sum(s.search.search_ms for s in ok), 2)
        res.rerank_ms = round(sum(s.search.rerank_ms for s in ok), 2)
        return res
