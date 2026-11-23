# backend-archive

backend 저장소에서 더 쓰지 않는 파일을 옮겨 둔 곳이에요. 원본은 backend main `0a1ceda`(2026-11-23) 기준이에요.

| 파일 | 뭐였나 | 왜 옮겼나 |
| --- | --- | --- |
| `docs/intent_eval.md` | 규칙만 쓰던 때의 임직원 의도 분류 측정 결과 (dev 92문장, 83.7%) | 지금 구조·숫자와 맞지 않아요. 최신은 backend `docs/intent_classification.md` |
| `scripts/send_demo_questions.py` | 마스킹 시연 질문을 보내고 대화 로그(chat_log) 새 행을 비교하던 스크립트 | 어디서도 안 쓰는 일회성 시연 도구 |

`send_demo_questions.py`는 backend 폴더 안(`scripts/`)에 있을 때를 기준으로 경로를 잡아요. 다시 쓰려면 backend `scripts/`로 복사해서 실행하세요.
