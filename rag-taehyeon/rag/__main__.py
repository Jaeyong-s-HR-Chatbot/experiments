"""python -m rag "질문"            질문 하나를 단계별로 보여 줘요
python -m rag --eval              골든셋으로 검색 평가 (GOLDENSET_DIR 필요)
python -m rag --eval --presets A0 A1 B C
python -m rag --calibrate         answer 임계값 제안 (Dense 필요)
"""
from __future__ import annotations

import argparse
import json
import logging

from . import Pipeline, load_config
from .goldenset import calibrate, evaluate, load_items


def main():
    ap = argparse.ArgumentParser(prog="rag")
    ap.add_argument("question", nargs="?")
    ap.add_argument("--preset", default=None)
    ap.add_argument("--presets", nargs="*", default=None)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--router", action="store_true", help="평가에 라우터까지 포함")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--golden", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-embed", action="store_true")
    ap.add_argument("--rerank", choices=["none", "colbert", "cross"], default=None)
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    cfg = load_config()
    if a.no_embed:
        cfg = cfg.with_(embed_enabled=False)
    if a.preset:
        cfg = cfg.preset(a.preset)
    if a.rerank:
        cfg = cfg.with_(rerank=a.rerank)
    pipe = Pipeline(cfg)
    print("색인:", json.dumps(pipe.info(), ensure_ascii=False))
    if not pipe.retriever.dense_ready:
        print("※ Dense 검색이 꺼져 있어요 (임베딩 모델 또는 DB 임베딩 없음). BM25만으로 검색해요.")

    if a.eval or a.calibrate:
        items = load_items(a.golden)[: a.limit]
        if a.calibrate:
            print(json.dumps(calibrate(pipe, items), ensure_ascii=False, indent=1))
            return
        for name in (a.presets or [cfg.preset_name]):
            s = evaluate(pipe.with_config(cfg.preset(name)), items, force_search=not a.router)["summary"]
            by = s.pop("by_category")
            print(f"\n[{name}] " + json.dumps(s, ensure_ascii=False))
            for c, v in by.items():
                print(f"   {c:22s} {v}")
        return

    if not a.question:
        ap.print_help()
        return
    r = pipe.run(a.question)
    print(f"\n상태: {r.status} | hr_needed={r.hr_needed} | {r.message or ''}")
    for s in r.subs:
        qc = s.query_class
        print(f"- {s.question}\n    검색 질의: {s.rewritten}\n    카테고리: {qc.categories} confident={qc.confident} slots={qc.slots} → {s.route}")
        if s.search:
            print(f"    필터: {s.search.used_filter} {s.search.filter_categories} fallback={s.search.used_fallback} "
                  f"1등 코사인={s.search.top_cos} {s.search.doc_kind_counts}")
    for c in r.chunks:
        print(f"  {c.rank}. [{c.chunk_id}] {c.title}  {c.score}  {c.categories}")


if __name__ == "__main__":
    main()
