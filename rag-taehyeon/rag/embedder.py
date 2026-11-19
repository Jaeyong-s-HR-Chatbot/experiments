"""BGE-M3 임베딩. 모델을 쓸 수 없으면 None을 돌려주고, 검색은 BM25만으로 동작해요.

- M3Embedder : FlagEmbedding. Dense와 ColBERT 벡터를 같은 모델에서 뽑아요 (재정렬 colbert에 필요)
- Embedder   : sentence-transformers. Dense만
- HashEmbedder: 테스트용 가짜
"""
from __future__ import annotations

import hashlib
import logging
import time

import numpy as np

log = logging.getLogger("rag")


def doc_text(r: dict) -> str:
    """조각을 모델에 넣는 글. DB 저장소의 rag_embed.py와 같은 형식이에요 ("규정명 조문 제목\\n본문")."""
    head = f'{r["doc_name"]} {r["article"]} {r["article_title"]}'.strip()
    return f'{head}\n{r["text"]}'


def _device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


class _Cached:
    def __init__(self):
        self._cache: dict[str, np.ndarray] = {}

    def encode(self, texts: list[str]) -> np.ndarray:
        miss = [t for t in dict.fromkeys(texts) if t not in self._cache]
        for i in range(0, len(miss), 512):
            part = miss[i:i + 512]
            for t, v in zip(part, self._dense(part)):
                self._cache[t] = v
        return np.stack([self._cache[t] for t in texts])


class Embedder(_Cached):
    """sentence-transformers로 BGE-M3 dense 벡터(1024차원, 정규화)를 만들어요."""

    def __init__(self, model_name: str = "BAAI/bge-m3", max_len: int = 1024, batch_size: int = 16):
        super().__init__()
        from sentence_transformers import SentenceTransformer
        self.device = _device()
        self.model = SentenceTransformer(model_name, device=self.device)
        self.model.max_seq_length = max_len
        self.name, self.bs = model_name, batch_size

    def _dense(self, texts):
        v = self.model.encode(texts, batch_size=self.bs, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
        return np.asarray(v, dtype=np.float32)


class M3Embedder(_Cached):
    """FlagEmbedding으로 BGE-M3의 Dense와 ColBERT 벡터를 뽑아요."""

    def __init__(self, model_name: str = "BAAI/bge-m3", max_len: int = 1024, batch_size: int = 16):
        super().__init__()
        from FlagEmbedding import BGEM3FlagModel
        self.device = _device()
        self.model = BGEM3FlagModel(model_name, use_fp16=(self.device == "cuda"), devices=self.device)
        self.name, self.max_len, self.bs = model_name, max_len, batch_size
        self._col: dict[str, np.ndarray] = {}

    def _run(self, texts, colbert):
        return self.model.encode(list(texts), batch_size=self.bs, max_length=self.max_len,
                                 return_dense=True, return_sparse=False, return_colbert_vecs=colbert)

    def _dense(self, texts):
        d = np.asarray(self._run(texts, False)["dense_vecs"], dtype=np.float32).reshape(len(texts), -1)
        return d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)

    def colbert(self, texts: list[str]) -> list[np.ndarray]:
        """글마다 토큰 벡터 묶음(토큰 수 × 차원, 정규화). 한 번 만든 건 다시 써요."""
        miss = [t for t in dict.fromkeys(texts) if t not in self._col]
        if miss:
            for t, c in zip(miss, self._run(miss, True)["colbert_vecs"]):
                c = np.asarray(c, dtype=np.float32)
                self._col[t] = (c / np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-12)).astype(np.float16)
        return [self._col[t] for t in texts]


class HashEmbedder:
    """테스트용 가짜 임베딩. 같은 글자 묶음이 많을수록 가깝게 나와요. 실제 검색에는 쓰지 않아요."""
    name, device = "hash", "cpu"

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _vec(self, t: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        t = "".join(t.split())
        for j in range(len(t) - 1):
            v[int(hashlib.md5(t[j:j + 2].encode()).hexdigest()[:8], 16) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts])

    def colbert(self, texts: list[str]) -> list[np.ndarray]:
        out = []
        for t in texts:
            words = [w for w in t.replace("\n", " ").split() if w][:256] or [t or " "]
            m = np.stack([self._vec(w + " ") for w in words])
            out.append(m.astype(np.float16))
        return out


def load_embedder(cfg):
    if not cfg.embed_enabled:
        return None
    order = {"auto": ["flag", "st"], "flag": ["flag"], "st": ["st"]}.get(cfg.embed_backend, ["flag", "st"])
    err = None
    for kind in order:
        try:
            cls = M3Embedder if kind == "flag" else Embedder
            t0 = time.time()
            log.info("임베딩 모델 불러오는 중: %s (%s). 처음이면 약 2.3GB를 내려받아요.", cfg.embed_model, cls.__name__)
            e = cls(cfg.embed_model, cfg.max_len, cfg.batch_size)
            log.info("임베딩 모델 준비 완료: %s, %.0f초", e.device, time.time() - t0)
            return e
        except Exception as e:  # 설치 안 됨, 모델을 못 받음 등
            err = e
    log.warning("임베딩 모델을 쓸 수 없어 BM25만으로 검색해요: %s", type(err).__name__)
    return None
