"""② 질문 분류 (키워드 + BGE-M3 대표 질문).

- 카테고리(여러 개): 키워드가 있으면 키워드 카테고리, 없으면 대표 질문과 가장 가까운 카테고리
- confident: 키워드와 대표 질문이 같은 카테고리를 가리키는지로 정해요 (LLM이 매긴 점수가 아니에요)
- 임베딩 모델이 없으면 키워드만으로 판단하고, 범위 밖 여부는 판단하지 않아요(in_scope=None).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from . import text as T


@dataclass
class QueryClass:
    categories: list[tuple[str, float]] = field(default_factory=list)   # (카테고리, 점수) 높은 순
    scores: dict[str, float] = field(default_factory=dict)              # 전체 카테고리 점수
    confident: bool = False
    slots: dict[str, str] = field(default_factory=dict)
    in_scope: bool | None = None       # None = 판단 불가 (임베딩 없음)
    personal: bool = False             # 본인 기록 조회
    both: bool = False                 # 본인 기록 + 규정
    first_person: bool = False
    sensitive: bool = False
    team: bool = False
    ambiguous: str | None = None       # "조건" / "주제"
    ask: str | None = None
    oos_score: float | None = None

    def to_dict(self):
        return asdict(self)


class Classifier:
    def __init__(self, cfg, embedder=None):
        self.cfg, self.embedder = cfg, embedder
        self.labels, self.proto = [], None
        if embedder is not None:
            self.labels = list(T.PROTO_QUESTIONS)
            mats = [embedder.encode(qs).mean(axis=0) for qs in T.PROTO_QUESTIONS.values()]
            m = np.stack(mats)
            self.proto = m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-9, None)

    def classify(self, text: str, original: str | None = None) -> QueryClass:
        cfg, orig = self.cfg, T.normalize(original or text)
        kw = T.kw_hits(text, T.CATEGORY_KEYWORDS)
        kw_cats = sorted(kw, key=lambda c: -kw[c])
        qc = QueryClass(slots=T.extract_slots(text))
        qc.first_person = bool(T.FIRST_PERSON_RE.search(orig))
        qc.personal = qc.first_person and any(w in orig for w in T.LOOKUP_WORDS)
        qc.both = qc.personal and any(w in orig for w in T.REG_CUES)
        qc.sensitive = any(w in orig for w in T.SENSITIVE_WORDS)
        qc.team = any(w in orig for w in T.TEAM_WORDS) and any(w in orig for w in T.LOOKUP_WORDS + ["보여", "알려"])

        best_s = margin = None
        if self.proto is not None:
            sims = dict(zip(self.labels, (self.proto @ self.embedder.encode([text])[0]).tolist()))
            qc.oos_score = round(sims.pop("범위밖"), 4)
            ranked = sorted(sims.items(), key=lambda x: -x[1])
            (best_c, best_s), margin = ranked[0], ranked[0][1] - ranked[1][1]
            qc.scores = {c: round(s, 4) for c, s in sims.items()}
            if kw_cats:
                cats = kw_cats + ([best_c] if best_c not in kw_cats and best_s >= cfg.th_high_conf else [])
                qc.confident = best_c in kw_cats[:2]
            else:
                cats = [c for c, s in ranked if s >= best_s - cfg.proto_margin]
                qc.confident = best_s >= cfg.th_high_conf and margin >= cfg.proto_margin
            qc.categories = [(c, qc.scores[c]) for c in cats]
            qc.in_scope = bool(kw_cats) or (best_s >= cfg.th_in_scope and best_s > qc.oos_score)
        else:
            total = sum(kw.values()) or 1
            qc.scores = {c: round(kw.get(c, 0) / total, 4) for c in T.CATEGORIES}
            qc.categories = [(c, qc.scores[c]) for c in kw_cats]
            qc.confident = bool(kw_cats)
            qc.in_scope = True if kw_cats else None

        for rule in T.CLARIFY_RULES:
            if any(w in text for w in rule["if_any"]) and (not rule["need_any"] or any(w in text for w in rule["need_any"])) \
                    and not any(w in text for w in rule["unless_any"]):
                qc.ambiguous, qc.ask = "조건", rule["ask"]
                break
        if not qc.ambiguous and self.proto is not None and qc.in_scope and not kw_cats \
                and best_s < cfg.th_high_conf and margin < cfg.proto_margin:
            qc.ambiguous = "주제"
            qc.ask = f"어떤 내용이 궁금하신지 조금 더 알려주시겠어요? 예: {ranked[0][0]} 관련인지, {ranked[1][0]} 관련인지"
        return qc


def route(qc: QueryClass) -> str:
    """③ 라우터: 민감 → 팀 단위 → 범위 밖 → 본인 조회만 → 되묻기 → 검색"""
    if qc.sensitive:
        return "sensitive"
    if qc.team:
        return "team_scope"
    if qc.in_scope is False:
        return "out_of_scope"
    if qc.personal and not qc.both:
        return "hr_only"
    if qc.ambiguous:
        return "clarify"
    return "search"
