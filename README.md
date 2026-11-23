# experiments: 예전 RAG 코드·실험 노트북 보관

누리철도 HR 챗봇 팀이 **지금은 쓰지 않는** 예전 검색(RAG) 코드와 실험 노트북을 모아 둔 곳이에요. 기록용이라 여기서 개발하지 않아요.

| 폴더 | 원래 위치 | 내용 | 옮긴 이유 |
| --- | --- | --- | --- |
| `rag-taehyeon/` | `Jaeyong-s-HR-Chatbot/RAG-taehyeon` (main `238d3b3`, 히스토리 그대로) | 초기 `rag` 검색 패키지(Dense + BM25, RRF, 카테고리 필터, 선택 재정렬)와 재정렬 비교 실험(`experiments/rerank`) | 검색 방식은 2026-11-19에 "조문 검색 + 항 리랭크(B)"로 정함. backend 기본 검색은 이 패키지를 안 씀(`SEARCH_MODE=rag`일 때만 썼음) |
| `db-welfare-rag-baseline/` | `Jaeyong-s-HR-Chatbot/DB` `Welfare Rag Baseline.ipynb` (커밋 `3aade37`) | 초기 복리후생 RAG 베이스라인 노트북 (가상 샘플 규정, BGE-M3) | 실제 DB·골든셋 기반 평가로 대체됨 |
| `backend-archive/` | `Jaeyong-s-HR-Chatbot/backend` (main `0a1ceda`) | 예전 의도 분류 측정 결과(`docs/intent_eval.md`), 마스킹 시연 스크립트(`scripts/send_demo_questions.py`) | 측정 결과는 지금 구조와 맞지 않음(최신은 backend `docs/intent_classification.md`), 시연 스크립트는 안 씀 |

지금 쓰는 것
- 규정·인사 DB: `Jaeyong-s-HR-Chatbot/DB`
- 골든셋·평가 스크립트·Colab: `Jaeyong-s-HR-Chatbot/RAG` (예전 이름 `RAG-hyerim`)
- 챗봇 서버: `Jaeyong-s-HR-Chatbot/backend`

모든 인물·값은 가상 데이터예요.
