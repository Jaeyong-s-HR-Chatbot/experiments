"""python -m pytest -q   (DB·모델 없이 동작. HRDB_URL이 있으면 DB 테스트도 실행)"""
import os

import numpy as np
import pytest

from rag import Pipeline, load_config, split_question
from rag.classifier import Classifier, route
from rag.embedder import HashEmbedder
from rag.goldenset import build_messages, evaluate
from rag.store import _row

ROWS = [_row(r) for r in [
    {"chunk_id": "r03-0024", "document_id": "r03", "doc_name": "취업규칙", "article": "제23조", "article_title": "연차휴가",
     "categories": ["근태관리"], "text": "제23조(연차휴가) 1년간 80퍼센트 이상 출근한 직원에게 15일의 연차휴가를 준다."},
    {"chunk_id": "r03-0025", "document_id": "r03", "doc_name": "취업규칙", "article": "제23조", "article_title": "연차휴가",
     "categories": ["근태관리"], "text": "② 3년 이상 근속한 직원에게는 2년마다 1일을 가산한 연차휴가를 준다."},
    {"chunk_id": "r03-0026", "document_id": "r03", "doc_name": "취업규칙", "article": "제25조", "article_title": "병가",
     "categories": ["근태관리"], "text": "제25조(병가) 계속 3일 이상의 병가는 진단서를 제출한다. 병가는 연 60일 이내로 한다."},
    {"chunk_id": "r03-0068", "document_id": "r03", "doc_name": "취업규칙", "article": "별표 3", "article_title": "",
     "categories": ["생애이벤트-결혼", "근태관리"], "text": "청원휴가 대상 및 기간. 결혼 본인 5일, 자녀 1일. 사망 배우자 5일."},
    {"chunk_id": "r01-0013", "document_id": "r01", "doc_name": "복지후생 규정", "article": "제12조", "article_title": "경조비",
     "categories": ["생애이벤트-결혼", "기타 복리후생"], "text": "제12조(경조비) 직원에게 결혼축하금, 사망조위금 등 경조금을 지급할 수 있다."},
    {"chunk_id": "r05-0009", "document_id": "r05", "doc_name": "여비규정", "article": "별표 2", "article_title": "",
     "categories": ["출장"], "text": "국내 출장 여비. 서울 숙박비 상한 10만원, 광역시 8만원. 일비 2만5천원, 식비 2만5천원."},
    {"chunk_id": "r09-0004", "document_id": "r09", "doc_name": "퇴직급여 지급 규정", "article": "제4조", "article_title": "퇴직금",
     "categories": ["생애이벤트-휴직·복직·퇴직"], "text": "제4조(퇴직금) 계속근로기간 1년에 대하여 30일분 이상의 평균임금을 퇴직금으로 지급한다."},
    {"chunk_id": "L01-0060", "document_id": "L01", "doc_kind": "법령", "doc_name": "근로기준법", "article": "제60조", "article_title": "연차 유급휴가",
     "categories": ["근태관리"], "text": "사용자는 1년간 80퍼센트 이상 출근한 근로자에게 15일의 유급휴가를 주어야 한다."},
    {"chunk_id": "r30-0001", "document_id": "r30", "doc_name": "문서관리규정", "article": "제3조", "article_title": "보존",
     "categories": ["공통"], "text": "문서는 보존 연한에 따라 보존하고 폐기한다."},
]]


def make(preset="C", dense=True, **kw):
    cfg = load_config().with_(cache_dir=os.path.join(os.getenv("PYTEST_TMP", "/tmp"), "rag_test_cache"), hrdb_url="", **kw).preset(preset)
    emb = HashEmbedder() if dense else None
    mat = emb.encode([f'{r["doc_name"]} {r["article_title"]}\n{r["text"]}' for r in ROWS]) if dense else None
    return Pipeline(cfg, embedder=emb, rows=ROWS, emb=mat)


def test_split_only_when_categories_differ():
    assert len(split_question("결혼하면 휴가 며칠이고 경조금 얼마야?")) == 2
    assert split_question("5일 썼는데, 며칠까지 돼요?") == ["5일 썼는데, 며칠까지 돼요?"]
    assert len(split_question("연차는 며칠이고 병가는 며칠이야?")) == 1          # 같은 카테고리면 안 나눔


def test_split_shares_event_word():
    parts = split_question("결혼하면 휴가 며칠이고 경조금 얼마야?")
    assert all("결혼" in p for p in parts)


def test_masking_tags_removed():
    assert "[전화]" not in split_question("제 번호 [전화]인데 병가 며칠이야?")[0]


@pytest.mark.parametrize("q,status", [
    ("팀장한테 폭언을 들었어", "sensitive"),
    ("우리 팀 연차 현황 보여줘", "team_scope"),
    ("휴가 며칠이야?", "clarify"),
    ("내 연차 몇 개 남았어?", "hr_only"),
    ("병가 쓸 때 진단서 필요해?", "ok"),
])
def test_status(q, status):
    assert make(dense=False).run(q).status == status


def test_hr_needed_with_search():
    r = make(dense=False).run("나 올해 연차 몇 개야?")
    assert r.status == "ok" and r.hr_needed and r.chunks


def test_force_search_skips_router():
    r = make(dense=False).run("내 연차 몇 개 남았어?", force_search=True)
    assert r.status == "ok" and r.chunks[0].chunk_id.startswith("r03")


def test_bm25_finds_article():
    r = make(dense=False).run("병가 쓸 때 진단서 필요해?")
    assert r.chunks[0].chunk_id == "r03-0026"
    assert r.subs[0].search.top_cos is None and r.subs[0].search.dense is False


def test_same_article_merged():
    r = make(dense=False, filter="none").run("연차휴가 며칠 줘?", force_search=True)
    ids = [c.chunk_id for c in r.chunks]
    assert not ({"r03-0024", "r03-0025"} <= set(ids))


def test_hard_filter_keeps_matching_or_common():
    p = make("B", dense=False)
    r = p.run("서울 출장 가면 숙박비 얼마까지 나와?")
    s = r.subs[0].search
    assert s.used_filter == "hard" and "출장" in s.filter_categories
    assert all(set(c.categories) & {"출장", "공통"} for c in s.candidates)


def test_adaptive_falls_back_when_too_few():
    p = make("C", dense=False)
    r = p.run("퇴직하면 연차휴가 수당 줘?", force_search=True)
    assert r.subs[0].search.candidates


def test_presets_switch_without_reload():
    p = make(th_answer=0.0)        # 가짜 임베딩은 코사인이 낮아서 근거 판단을 끔
    cfg = p.cfg
    for name in ("A0", "A1", "B", "C"):
        q = p.with_config(cfg.preset(name))
        assert q.retriever is p.retriever and q.cfg.preset_name == name
        assert q.run("서울 출장 가면 숙박비 얼마까지 나와?", force_search=True).chunks[0].chunk_id == "r05-0009"
    assert p.cfg.preset_name == "C"


def test_dense_scores_and_no_evidence():
    p = make(th_answer=0.99)
    r = p.run("서울 출장 가면 숙박비 얼마까지 나와?")
    assert r.status == "no_evidence" and r.subs[0].search.top_cos is not None
    assert make(th_answer=0.01).run("서울 출장 가면 숙박비 얼마까지 나와?").status == "ok"


def test_internal_rule_before_law():
    r = make(dense=False, filter="none").run("연차휴가 15일 80퍼센트 출근", force_search=True)
    kinds = [c.doc_kind for c in r.chunks]
    assert kinds.index("내부규정") < kinds.index("법령")
    assert any(c.source for c in r.chunks if c.doc_kind == "법령")


def test_doc_kind_filter():
    cfg = load_config().with_(hrdb_url="", doc_kinds=("내부규정",))
    rows = [r for r in ROWS if r["doc_kind"] in cfg.doc_kinds]
    assert all(r["doc_kind"] == "내부규정" for r in rows) and len(rows) == len(ROWS) - 1


def test_build_messages():
    r = make(dense=False).run("나 결혼하는데 휴가 며칠이야?")
    m = build_messages(r.llm_request, profile="신호팀 대리, 근속 3년")
    assert m[0]["role"] == "system" and "[근거 조문]" in m[1]["content"] and "신호팀 대리" in m[1]["content"]
    assert r.chunks[0].chunk_id in m[1]["content"]


def test_multiturn_carry():
    r = make(dense=False).run("그럼 며칠 쉬어요?", history=["다음 달에 결혼해요"], force_search=True)
    assert "결혼" in r.subs[0].rewritten


def test_classifier_with_prototypes():
    cfg = load_config().with_(hrdb_url="")
    qc = Classifier(cfg, HashEmbedder()).classify("출장 가면 숙박비 얼마 나와?")
    assert qc.categories[0][0] == "출장" and qc.in_scope and qc.oos_score is not None
    assert route(qc) == "search"


def test_old_dump_row_defaults():
    r = _row({"chunk_id": "r03-0001", "doc_name": "취업규칙", "category": "근태관리", "text": "x"})
    assert r["categories"] == ["근태관리"] and r["doc_kind"] == "내부규정" and r["document_id"] == "r03"


def test_evaluate_counts():
    items = [{"id": "t1", "question": "병가 쓸 때 진단서 필요해?", "gold_chunks": {"primary": ["r03-0026"]}, "needs_hr_data": False, "category": "근태관리"},
             {"id": "t2", "question": "법령 문항", "gold_chunks": {"primary": ["L99-0001"]}, "needs_hr_data": False, "category": "근태관리"}]
    s = evaluate(make(dense=False), items)["summary"]
    assert s["scored"] == 1 and s["skipped_no_gold"] == 1 and s["hit@1"] == 1.0


def test_backend_adapter_format(monkeypatch):
    from rag import backend
    monkeypatch.setattr(backend, "_pipe", make(dense=False))
    hits = backend.search("병가 쓸 때 진단서 필요해?", k=3)
    assert len(hits) <= 3 and {"id", "citation", "revised", "text", "score"} <= set(hits[0])
    assert 0.0 <= hits[0]["score"] <= 1.0


@pytest.mark.skipif(not os.getenv("HRDB_URL"), reason="HRDB_URL 없음")
def test_db_load_has_embeddings():
    from rag import store
    rows, mat, src = store.load(load_config())
    assert src == "db" and len(rows) > 1000
    assert mat is None or (mat.shape[0] == len(rows) and abs(float(np.linalg.norm(mat[0])) - 1.0) < 1e-3)


# ───────── 재정렬 ─────────
class FakeCross:
    """질문에 '병가'가 있으면 병가 조문에, 아니면 글자가 많이 겹치는 조각에 높은 점수를 줘요."""
    name = "fake-cross"

    def __init__(self):
        self.calls = 0

    def score(self, query, docs):
        self.calls += 1
        return [1.0 if ("병가" in query and "제25조" in d) else len(set(query) & set(d)) / 100 for d in docs]


def test_rerank_cross_moves_target_to_top():
    p = make(th_answer=0.0, filter="none", rerank="cross")
    fake = FakeCross()
    p.retriever.set_cross(p.cfg.rerank_model, fake)
    r = p.run("연차휴가 말고 병가는 어떻게 돼?", force_search=True)
    s = r.subs[0].search
    assert fake.calls == 1 and s.rerank == "cross" and s.rerank_ms >= 0
    assert r.chunks[0].chunk_id == "r03-0026" and s.candidates[0].rerank == 1.0
    assert sorted(s.pool) == sorted(c.chunk_id for c in s.candidates)       # 재정렬은 순서만 바꿔요
    assert [c.rank for c in s.candidates] == list(range(1, len(s.candidates) + 1))


def test_rerank_colbert_runs_and_keeps_pool():
    p = make(th_answer=0.0, filter="none", rerank="colbert")
    r = p.run("서울 출장 가면 숙박비 얼마까지 나와?", force_search=True)
    s = r.subs[0].search
    assert s.rerank == "colbert" and all(c.rerank is not None for c in s.candidates[:p.cfg.rerank_pool])
    assert r.pool_ids and set(r.pool_ids) == set(s.pool)


def test_rerank_pool_limits_reordering():
    p = make(th_answer=0.0, filter="none", rerank="cross", rerank_pool=2)
    p.retriever.set_cross(p.cfg.rerank_model, FakeCross())
    s = p.run("연차휴가 며칠이야 병가", force_search=True).subs[0].search
    assert set(c.chunk_id for c in s.candidates[:2]) == set(s.pool[:2])
    assert [c.chunk_id for c in s.candidates[2:]] == s.pool[2:]
    assert all(c.rerank is None for c in s.candidates[2:])


def test_rerank_switch_without_reload():
    p = make(th_answer=0.0, filter="none")
    p.retriever.set_cross(p.cfg.rerank_model, FakeCross())
    q = "서울 출장 가면 숙박비 얼마까지 나와?"
    for mode in ("none", "colbert", "cross"):
        r = p.with_config(p.cfg.with_(rerank=mode)).run(q, force_search=True)
        assert r.subs[0].search.rerank == mode and r.chunks
    assert p.cfg.rerank == "none"


def test_colbert_needs_capable_embedder():
    class DenseOnly:
        name = "dense-only"
        def encode(self, texts):
            return HashEmbedder().encode(texts)
    cfg = load_config().with_(hrdb_url="", th_answer=0.0, rerank="colbert").preset("A1")
    mat = HashEmbedder().encode([r["text"] for r in ROWS])
    p = Pipeline(cfg, embedder=DenseOnly(), rows=ROWS, emb=mat)
    with pytest.raises(RuntimeError):
        p.run("병가 며칠이야?", force_search=True)


def test_evaluate_reports_pool_and_time():
    p = make(th_answer=0.0, filter="none", rerank="cross")
    p.retriever.set_cross(p.cfg.rerank_model, FakeCross())
    items = [{"id": "t1", "question": "병가 쓸 때 진단서 필요해?", "gold_chunks": {"primary": ["r03-0026"]}, "split": "dev"}]
    out = evaluate(p, items)
    s, row = out["summary"], out["rows"][0]
    assert s["rerank"] == "cross" and s["in_pool"] == 1.0 and "ms_mean" in s and "rerank_ms_mean" in s
    assert row["rank"] == 1 and row["rank0"] is not None and row["split"] == "dev"


def test_file_mode_embeds_and_caches(tmp_path):
    import json
    f = tmp_path / "chunks.jsonl"
    f.write_text("\n".join(json.dumps({**r, "searchable": True}, ensure_ascii=False) for r in ROWS), encoding="utf-8")
    cfg = load_config().with_(hrdb_url="", chunks=(str(f),), cache_dir=str(tmp_path / "cache"), th_answer=0.0).preset("A0")
    p = Pipeline(cfg, embedder=HashEmbedder())
    assert p.retriever.dense_ready and p.source == "file+embedded"
    assert list((tmp_path / "cache").glob("emb_*.npy"))
    assert p.run("서울 출장 가면 숙박비 얼마까지 나와?", force_search=True).chunks[0].chunk_id == "r05-0009"
    assert Pipeline(cfg, embedder=HashEmbedder()).retriever.dense_ready        # 두 번째는 저장해 둔 걸 써요
