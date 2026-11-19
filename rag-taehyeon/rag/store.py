"""조각 불러오기. DB(rag.chunks) → chunks.jsonl 파일 순서로 찾아요.

- DB: 조각 + 저장된 BGE-M3 임베딩을 함께 읽어요.
- 파일: 조각만 읽어요 (임베딩이 없어 Dense 검색은 꺼져요).
검색 대상(searchable)만 읽어요.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from .text import COMMON

log = logging.getLogger("rag")


def _row(r: dict) -> dict:
    cats = r.get("categories") or ([r["category"]] if r.get("category") else [COMMON])
    eff = r.get("effective_date")
    return {
        "chunk_id": r["chunk_id"], "document_id": r.get("document_id") or r["chunk_id"].split("-")[0],
        "doc_kind": r.get("doc_kind") or "내부규정", "doc_name": r.get("doc_name") or "",
        "article": r.get("article") or "", "article_title": r.get("article_title") or "",
        "chunk_type": r.get("chunk_type") or "", "categories": list(cats), "text": r.get("text") or "",
        "effective_date": eff.isoformat() if hasattr(eff, "isoformat") else (eff or ""),
    }


def _from_db(url: str):
    import psycopg
    from psycopg.rows import dict_row
    with psycopg.connect(url, connect_timeout=5, row_factory=dict_row) as con:
        cols = {r["column_name"] for r in con.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_schema='rag' AND table_name='chunks'")}
        want = ["chunk_id", "document_id", "doc_kind", "doc_name", "article", "article_title", "chunk_type",
                "categories", "category", "text", "effective_date"]
        sel = ", ".join(c for c in want if c in cols)      # 예전 덤프에는 doc_kind, categories가 없어요
        emb = ", embedding::text AS emb" if "embedding" in cols else ""
        rows = con.execute(f"SELECT {sel}{emb} FROM rag.chunks WHERE searchable ORDER BY chunk_id").fetchall()
    out, vecs, ok = [], [], True
    for r in rows:
        out.append(_row(r))
        e = r.get("emb")
        if e:
            vecs.append(np.fromstring(e.strip("[]"), sep=",", dtype=np.float32))
        else:
            ok = False
    mat = np.stack(vecs) if (ok and vecs) else None
    if mat is not None:                                     # 코사인 = 내적이 되도록 정규화
        mat /= np.clip(np.linalg.norm(mat, axis=1, keepdims=True), 1e-9, None)
    return out, mat


def _from_files(paths) -> list[dict]:
    out = []
    for p in paths:
        with open(Path(p), encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r.get("searchable", True):
                        out.append(_row(r))
    return out


def load(cfg):
    """(조각 목록, 임베딩 행렬 또는 None, 출처)"""
    rows, mat, src = [], None, ""
    if cfg.hrdb_url:
        try:
            rows, mat = _from_db(cfg.hrdb_url)
            src = "db"
        except Exception as e:
            log.warning("DB에서 조각을 읽지 못했어요 (%s). 파일을 찾아볼게요.", type(e).__name__)
    if not rows and cfg.chunks:
        rows, src = _from_files(cfg.chunks), "file"
    if not rows:
        raise RuntimeError("조각을 찾지 못했어요. HRDB_URL을 넣거나 config.toml의 [data] chunks에 chunks.jsonl 경로를 적어 주세요.")
    if cfg.doc_kinds:
        keep = [i for i, r in enumerate(rows) if r["doc_kind"] in cfg.doc_kinds]
        rows = [rows[i] for i in keep]
        mat = mat[keep] if mat is not None else None
    return rows, mat, src
