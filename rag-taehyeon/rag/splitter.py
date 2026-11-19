"""① 질문 쪼개기와 재작성 (규칙).

"~이고 / ~랑 / 그리고 / ?" 로 나눴을 때, 조각마다 카테고리 키워드가 있고 카테고리가 서로 다를 때만 나눠요.
"""
from __future__ import annotations

import re

from .text import ANAPHORA, CATEGORY_KEYWORDS, expansions, extract_slots, kw_hits, normalize, topic_words

SPLIT_RE = re.compile(r"(?<=[가-힣])(?:이고|하고|이랑|랑|고)\s+|\s*그리고\s+|[?？]\s+|(?<=[가-힣]),\s+")
_EVENTS = ["결혼", "혼인", "출산", "임신", "사망", "부고", "휴직", "퇴직", "퇴사", "입사", "승진", "출장", "이사"]


def event_words(text: str) -> list[str]:
    return [w for w in _EVENTS if w in text]


def split_question(question: str) -> list[str]:
    """복합 질문을 검색 단위로 나눠요. 나눌 필요가 없으면 [질문] 그대로."""
    q = normalize(question)
    parts = [p.strip() for p in SPLIT_RE.split(q) if p and p.strip()]
    merged: list[str] = []
    has: list[bool] = []
    for p in parts:
        t = bool(topic_words(p))
        if merged and (not t or not has[-1]):          # 주제어가 없는 조각은 이웃과 합침
            merged[-1] += " " + p
            has[-1] = has[-1] or t
        else:
            merged.append(p)
            has.append(t)
    cats = [set(kw_hits(p, CATEGORY_KEYWORDS)) for p in merged]
    if len(merged) < 2 or all(c == cats[0] for c in cats):   # 카테고리가 같으면 한 번에 검색
        return [q]
    shared = event_words(q) + list(extract_slots(q).values())  # 이벤트·조건 단어는 조각마다 공유
    return [" ".join([s for s in shared if s not in p] + [p]) for p in merged]


def carry_over(question: str, history: list[str] | None) -> list[str]:
    """멀티턴: 지시어가 있거나 주제어가 없으면 직전 발화의 이벤트·조건·주제어를 이어받아요."""
    if not history:
        return []
    q, prev = normalize(question), normalize(history[-1])
    if not (any(a in q for a in ANAPHORA) or not topic_words(q)):
        return []
    c = event_words(prev) + list(extract_slots(prev).values()) + topic_words(prev)
    return [w for i, w in enumerate(c) if w not in q and w not in c[:i]]


def rewrite(sub: str) -> str:
    """검색 질의: 조각 + 사전에 있는 규정 용어"""
    return " ".join([sub] + expansions(sub))
