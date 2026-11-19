"""설정. config.toml을 읽고, 환경변수가 있으면 환경변수 값을 써요."""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 실험 구성: 모델과 색인은 그대로 두고 스위치만 바꿔요.
PRESETS = {
    "A0": {"use_dense": True, "use_bm25": False, "filter": "none"},      # Dense만
    "A1": {"use_dense": True, "use_bm25": True, "filter": "none"},       # + BM25, RRF로 합침
    "B": {"use_dense": True, "use_bm25": True, "filter": "hard"},        # + 카테고리 하드 필터
    "C": {"use_dense": True, "use_bm25": True, "filter": "adaptive"},    # + 적응형 필터, fallback
}


@dataclass(frozen=True)
class Config:
    hrdb_url: str = ""
    chunks: tuple[str, ...] = ()
    doc_kinds: tuple[str, ...] = ()
    cache_dir: str = ".rag_cache"
    embed_model: str = "BAAI/bge-m3"
    embed_enabled: bool = True
    goldenset_dir: str = ""
    preset_name: str = "C"
    use_dense: bool = True
    use_bm25: bool = True
    filter: str = "adaptive"            # none / hard / adaptive
    n_candidates: int = 50
    n_final: int = 20
    top_k: int = 8
    rrf_k: int = 60
    min_filtered: int = 3
    soft_boost: float = 0.3
    internal_boost: float = 0.15
    th_answer: float = 0.45
    th_in_scope: float = 0.40
    th_high_conf: float = 0.55
    proto_margin: float = 0.03
    rerank: str = "none"                # none / colbert / cross
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_pool: int = 20               # 재정렬할 후보 수
    rerank_dense_weight: float = 0.0    # 0이면 재정렬 점수만, 0.5면 코사인과 반반
    max_len: int = 1024                 # 모델에 넣는 최대 토큰 수
    batch_size: int = 16
    embed_backend: str = "auto"         # auto / flag(FlagEmbedding) / st(sentence-transformers)
    extra: dict = field(default_factory=dict, compare=False)

    def preset(self, name: str) -> "Config":
        """구성 이름(A0/A1/B/C)으로 스위치만 바꾼 설정을 돌려줘요."""
        if name not in PRESETS:
            raise ValueError(f"구성 이름은 {list(PRESETS)} 중 하나여야 해요: {name}")
        return replace(self, preset_name=name, **PRESETS[name])

    def with_(self, **kw) -> "Config":
        return replace(self, **kw)


def load_config(path: str | os.PathLike | None = None) -> Config:
    p = Path(path) if path else ROOT / "config.toml"
    t = tomllib.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    d, e, s, th = t.get("data", {}), t.get("embed", {}), t.get("search", {}), t.get("threshold", {})
    rr = t.get("rerank", {})
    env_chunks = [x for x in os.getenv("RAG_CHUNKS", "").split(os.pathsep) if x]

    def here(x):          # config.toml에 적은 상대 경로는 이 저장소 폴더 기준
        return str(x if Path(x).is_absolute() else ROOT / x)
    cfg = Config(
        hrdb_url=os.getenv("HRDB_URL", "").strip(),
        chunks=tuple(env_chunks or [here(x) for x in d.get("chunks", [])]),
        doc_kinds=tuple(d.get("doc_kinds", [])),
        cache_dir=here(d.get("cache_dir", ".rag_cache")),
        embed_model=os.getenv("RAG_EMBED_MODEL", e.get("model", "BAAI/bge-m3")),
        embed_enabled=(os.getenv("RAG_EMBED", "").strip().lower() not in {"0", "off", "false", "no"}) if os.getenv("RAG_EMBED")
        else bool(e.get("enabled", True)),
        goldenset_dir=os.getenv("GOLDENSET_DIR", "").strip(),
        n_candidates=int(s.get("n_candidates", 50)), n_final=int(s.get("n_final", 20)), top_k=int(s.get("top_k", 8)),
        rrf_k=int(s.get("rrf_k", 60)), min_filtered=int(s.get("min_filtered", 3)),
        soft_boost=float(s.get("soft_boost", 0.3)), internal_boost=float(s.get("internal_boost", 0.15)),
        th_answer=float(th.get("answer", 0.45)), th_in_scope=float(th.get("in_scope", 0.40)),
        th_high_conf=float(th.get("high_conf", 0.55)), proto_margin=float(th.get("proto_margin", 0.03)),
        rerank=rr.get("mode", "none"), rerank_model=rr.get("model", "BAAI/bge-reranker-v2-m3"),
        rerank_pool=int(rr.get("pool", 20)), rerank_dense_weight=float(rr.get("dense_weight", 0.0)),
        max_len=int(e.get("max_len", 1024)), batch_size=int(e.get("batch_size", 16)), embed_backend=e.get("backend", "auto"),
    )
    return cfg.preset(s.get("preset", "C"))
