"""골든셋 연결: LLM에 넘길 메시지 만들기, 검색 평가.

골든셋 저장소 폴더는 환경변수 GOLDENSET_DIR 로 알려 줘요. 그 안에서 golden_set.json 을 찾아요.
평가는 조항 단위예요: 찾은 조각이 정답 조각과 같거나, 같은 문서·같은 조항이면 맞음으로 봐요.
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

log = logging.getLogger("rag")

SYSTEM_PROMPT = """당신은 누리철도㈜ 인사(HR) 안내 챗봇입니다. 아래 규칙을 반드시 지키세요.
1. [근거 조문]과 [직원 정보]에 있는 내용으로만 답하세요. 근거에 없는 내용, 금액, 일수는 지어내지 마세요.
2. 근거에서 답을 찾을 수 없으면 "규정에서 확인되지 않아요. 인사팀에 확인해 주세요."라고 답하세요.
3. 질문의 전제가 근거와 다를 때만 먼저 바로잡고 근거의 내용을 알려 주세요.
4. [직원 정보]의 조건(근속, 직급, 가족, 나이)을 근거의 기준에 하나씩 대입해 답하세요.
5. 금액이나 일수를 계산할 때는 계산식과 결과를 함께 쓰세요.
6. 답변 끝에 사용한 근거의 번호를 [근거: r03-0024] 형식으로 적으세요. 법령을 근거로 쓰면 출처도 적으세요.
7. 한국어로 간결하게 쓰고, 숫자(일수·금액·비율)는 근거와 똑같이 쓰세요."""


def build_messages(req, profile: str | None = None) -> list[dict]:
    """LlmRequest → OpenAI 형식 메시지 [system, user]"""
    blocks = []
    for c in req.chunks:
        head = f"[{c.chunk_id}] {c.title}" + (f" (시행 {c.effective_date})" if c.effective_date else "") \
               + (f" · 출처: {c.source}" if c.source else "")
        blocks.append(head + "\n" + c.text)
    parts = []
    if profile:
        parts.append("[직원 정보]\n" + profile)
    elif req.hr_needed:
        parts.append("[직원 정보]\n(인사 DB 조회 결과를 여기에 넣어요)")
    if req.slots:
        parts.append("[질문에서 읽은 조건]\n" + ", ".join(f"{k}: {v}" for k, v in req.slots.items()))
    parts.append("[근거 조문]\n" + "\n\n".join(blocks))
    if req.notes:
        parts.append("[참고]\n" + "\n".join("- " + n for n in req.notes))
    parts.append("[질문]\n" + req.question)
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "\n\n".join(parts)}]


def find_golden(directory: str | None = None) -> Path:
    d = Path(directory or os.getenv("GOLDENSET_DIR", "") or ".")
    if d.is_file():
        return d
    hits = sorted(d.rglob("golden_set.json"))
    if not hits:
        raise FileNotFoundError(f"golden_set.json을 찾지 못했어요: {d} (GOLDENSET_DIR을 확인해 주세요)")
    return hits[0]


def load_items(directory: str | None = None) -> list[dict]:
    g = json.loads(find_golden(directory).read_text(encoding="utf-8"))
    return g["items"] if isinstance(g, dict) else g


def evaluate(pipe, items: list[dict], ks=(1, 3, 5, 8), force_search: bool = True, label: str = "", every: int = 0) -> dict:
    """검색 평가. force_search=True면 검색기만, False면 라우터까지 포함해 재요.

    정답 조각(gold_chunks.primary)이 지금 색인에 있는 문항만 채점해요 (예: 법령 조각이 없는 DB면 법령 문항은 빠져요).
    """
    rows = pipe.retriever.rows

    def key(cid):
        r = rows.get(cid)
        return f'{r["document_id"]}|{r["article"] or cid}' if r else cid

    per, skipped, status, hr_ok, hr_n = [], 0, Counter(), 0, 0
    t0, total = time.time(), len(items)
    for n_done, it in enumerate(items, 1):
        if every and n_done > 1 and (n_done - 1) % every == 0:      # 진행 상황 (every=0이면 끔)
            el = time.time() - t0
            h3 = sum(r.get("hit@3", 0) for r in per) / len(per) if per and 3 in ks else float("nan")
            log.info("  [%s] %d/%d 문항 · 경과 %.0f초 · 남은 시간 약 %.0f초 · 지금까지 Hit@3 %.3f",
                     label or pipe.cfg.rerank, n_done - 1, total, el, el / (n_done - 1) * (total - n_done + 1), h3)
        gold = [g for g in (it.get("gold_chunks") or {}).get("primary", []) if g in rows]
        r = pipe.run(it["question"], force_search=force_search)
        status[r.status] += 1
        if "needs_hr_data" in it:
            hr_n += 1
            hr_ok += int(bool(it["needs_hr_data"]) == bool(r.hr_needed))
        if not gold:
            skipped += 1
            continue
        got = [key(c.chunk_id) for c in r.chunks]
        want = {key(g) for g in gold}
        rank = next((i + 1 for i, k in enumerate(got) if k in want), None)
        rank0 = next((i + 1 for i, c in enumerate(r.pool_ids) if key(c) in want), None)    # 재정렬 전 후보에서의 순위
        row = {"id": it.get("id"), "type": it.get("type"), "category": it.get("category"), "split": it.get("split"),
               "multi_hop": it.get("multi_hop"), "status": r.status, "question": it["question"],
               "rank": rank, "rank0": rank0, "in_pool": int(rank0 is not None), "rr": 1.0 / rank if rank else 0.0,
               "search_ms": r.search_ms, "rerank_ms": r.rerank_ms,
               "recall": len(want & set(got)) / len(want), "top": [c.chunk_id for c in r.chunks[:3]], "gold": gold}
        for k in ks:
            row[f"hit@{k}"] = int(rank is not None and rank <= k)
        per.append(row)
    if every:
        log.info("  [%s] %d/%d 문항 완료 · 전체 %.0f초", label or pipe.cfg.rerank, total, total, time.time() - t0)
    n = len(per) or 1
    ms = sorted(r["search_ms"] + r["rerank_ms"] for r in per) or [0.0]
    summary = {"preset": pipe.cfg.preset_name, "rerank": pipe.cfg.rerank, "dense": pipe.retriever.dense_ready,
               "scored": len(per), "skipped_no_gold": skipped,
               "in_pool": round(sum(r["in_pool"] for r in per) / n, 4),
               "ms_mean": round(sum(ms) / len(ms), 1), "ms_p95": round(ms[min(len(ms) - 1, int(len(ms) * 0.95))], 1),
               "rerank_ms_mean": round(sum(r["rerank_ms"] for r in per) / n, 1),
               **{f"hit@{k}": round(sum(r[f"hit@{k}"] for r in per) / n, 4) for k in ks},
               "recall": round(sum(r["recall"] for r in per) / n, 4), "mrr": round(sum(r["rr"] for r in per) / n, 4),
               "status": dict(status), "hr_needed_acc": round(hr_ok / hr_n, 4) if hr_n else None}
    by = defaultdict(list)
    for r in per:
        by[r["category"]].append(r)
    top_k = max(ks)
    summary["by_category"] = {c: {"n": len(v), f"hit@{top_k}": round(sum(x[f"hit@{top_k}"] for x in v) / len(v), 3),
                                  "hit@1": round(sum(x["hit@1"] for x in v) / len(v), 3)} for c, v in sorted(by.items())}
    return {"summary": summary, "rows": per}


def calibrate(pipe, items: list[dict]) -> dict:
    """answer 임계값 제안: '규정 없음' 문항과 나머지 문항의 1등 코사인을 가장 잘 가르는 값.

    Dense 검색이 켜져 있어야 해요. 임계값을 정한 문항으로 다시 채점하면 점수가 부풀어요. dev 문항으로만 정하세요.
    """
    if not pipe.retriever.dense_ready:
        raise RuntimeError("Dense 검색이 꺼져 있어요. 임베딩 모델과 DB 임베딩이 있어야 보정할 수 있어요.")
    pos, neg = [], []
    for it in items:
        s = pipe.retriever.search(it["question"], pipe.classifier.classify(it["question"]), pipe.cfg)
        if s.top_cos is not None:
            (neg if it.get("type") == "규정 없음" else pos).append(s.top_cos)
    if len(pos) < 2 or len(neg) < 2:
        raise RuntimeError("'규정 없음' 문항과 나머지 문항이 각각 2개 이상 있어야 해요.")
    best = max(sorted(set(pos + neg)), key=lambda t: sum(p >= t for p in pos) / len(pos) + sum(x < t for x in neg) / len(neg))
    return {"answer": round(best, 3), "answerable_min": min(pos), "no_rule_max": max(neg), "n_answerable": len(pos), "n_no_rule": len(neg),
            "kept_answerable": round(sum(p >= best for p in pos) / len(pos), 3), "blocked_no_rule": round(sum(x < best for x in neg) / len(neg), 3)}
