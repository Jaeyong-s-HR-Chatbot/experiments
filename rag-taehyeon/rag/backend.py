"""백엔드(app/graph/retriever.py)에 끼우는 연결부.

백엔드가 약속한 형식 [{id, citation, revised, text, score}] 으로 돌려줘요.
의도 분류와 거절은 백엔드가 하므로 여기서는 라우터를 건너뛰고 검색만 해요.
"""
from __future__ import annotations

from .config import load_config
from .pipeline import Pipeline

_pipe: Pipeline | None = None


def get_pipeline() -> Pipeline:
    global _pipe
    if _pipe is None:
        _pipe = Pipeline(load_config())
    return _pipe


def search(query: str, k: int = 5, wide: bool = False) -> list[dict]:
    pipe = get_pipeline()
    if wide or k != pipe.cfg.top_k:
        pipe = pipe.with_config(pipe.cfg.with_(top_k=k * 2 if wide else k))
    r = pipe.run(query, force_search=True)
    return [{"id": c.chunk_id, "citation": c.title, "revised": c.effective_date, "text": c.text, "score": c.score,
             "doc_kind": c.doc_kind, "source": c.source} for c in r.chunks]
