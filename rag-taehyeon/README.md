# rag: 누리철도 HR 챗봇 규정 검색

질문 하나를 **① 쪼개기 → ② 분류 → ③ 라우터 → ④ 검색 → `LlmRequest`** 까지 처리해요. LLM은 부르지 않아요.

`Welfare Rag Baseline.ipynb`의 설계(규칙 + BGE-M3)와 `Demo.ipynb`가 부르는 형식에 맞춰 **새로 작성한 코드**예요.
재용님의 원본 `rag` 패키지를 보고 만든 게 아니라서, 내부 구현과 점수는 원본과 다를 수 있어요.

## 노트북과 다른 점

| 항목 | 노트북 (Welfare Rag Baseline) | 이 패키지 |
| --- | --- | --- |
| 데이터 | 가상 샘플 규정 | 팀 DB `rag.chunks` (규정 1,124 + 법령 1,459 조각) |
| 카테고리 | 업무 영역 5개 | 팀 카테고리 9개 + 공통 (DB `categories`와 같은 이름) |
| Sparse | BGE-M3 sparse | BM25 (kiwi 형태소) |
| 재정렬 | BGE-M3 ColBERT | 선택: 없음(기본) / ColBERT / 크로스 인코더(bge-reranker-v2-m3) |
| Dense | 직접 인코딩 | DB에 저장된 BGE-M3 임베딩 사용. DB가 없으면 조각 파일을 직접 임베딩해 저장 |
| 추가 | | 팀 단위 조회 상태, 내부규정 우선, 법령 출처 표기 |

## 설치

```bash
pip install -r requirements.txt
```

- `kiwipiepy`가 없으면 글자 2개 묶음으로 대신 검색해요 (정확도가 떨어져요).
- `sentence-transformers`가 없거나 모델을 못 받으면 **BM25만으로** 동작해요. 이때는 범위 밖·근거 없음을 판단하지 못해요.

## 설정

| 환경변수 | 뜻 |
| --- | --- |
| `HRDB_URL` | 팀 DB 주소. 있으면 `rag.chunks`에서 조각과 임베딩을 읽어요 |
| `RAG_CHUNKS` | DB가 없을 때 쓸 `chunks.jsonl` 경로 (여러 개면 `;`(Windows) 또는 `:`로 구분). 임베딩이 없어 BM25만 동작 |
| `GOLDENSET_DIR` | 골든셋 저장소 폴더 (`golden_set.json`을 찾아요) |
| `RAG_EMBED` | `off`면 임베딩 모델을 안 불러오고 BM25만 써요 (GPU 메모리가 작은 PC용) |

나머지 값(구성, 후보 수, 임계값)은 `config.toml`에 있어요.

## 쓰는 법

```python
from rag import Pipeline, load_config, split_question

cfg = load_config()
pipe = Pipeline(cfg)                 # 조각 불러오기 + BM25 색인 (처음 20초쯤, 다음부터는 저장해 둔 걸 써요)

r = pipe.run("결혼하면 휴가 며칠이고 경조금 얼마야?")
r.status          # ok / hr_only / clarify / out_of_scope / sensitive / team_scope / no_evidence
r.hr_needed       # 본인 기록이 필요한 질문인지
r.subs            # 쪼갠 질문별 분류, 라우터, 검색 후보(최대 20개)
r.chunks          # LLM에 넘길 조각 (최대 8개): rank, chunk_id, title, score, categories, text
r.llm_request     # 질문, 조건(slots), 조각, 참고 사항

from rag.goldenset import build_messages
messages = build_messages(r.llm_request, profile="신호팀 대리, 근속 3년")   # OpenAI 형식 [system, user]
```

터미널에서:

```bash
python -m rag "서울 출장 가면 숙박비 얼마까지 나와?"
python -m rag --eval --presets A0 A1 B C      # 골든셋 검색 평가
python -m rag --eval --presets C --router     # 라우터까지 포함
python -m rag --calibrate                     # answer 임계값 제안 (Dense 필요)
python -m pytest -q
```

## 실험 구성

| 구성 | 내용 |
| --- | --- |
| A0 | Dense만 |
| A1 | + BM25, RRF로 합침 |
| B | + 카테고리 하드 필터 (질문 카테고리 또는 `공통`과 겹치는 조각만) |
| C | + 적응형 필터: 분류를 믿을 만하면 하드, 아니면 소프트. 결과가 약하면 필터를 풀어요 |

```python
for name in ["A0", "A1", "B", "C"]:
    pipe.with_config(cfg.preset(name)).run(q)     # 모델과 색인은 그대로, 스위치만 바꿈
```

## 재정렬

`config.toml`의 `[rerank] mode`를 바꾸거나, 코드에서 스위치만 바꿔요.

| mode | 방식 | 필요한 것 |
| --- | --- | --- |
| `none` | 재정렬 없음 (기본) | |
| `colbert` | BGE-M3의 토큰 벡터로 다시 비교 | `pip install FlagEmbedding` |
| `cross` | 크로스 인코더(bge-reranker-v2-m3)가 질문과 조각을 함께 읽음 | `sentence-transformers` |

```python
pipe.with_config(cfg.with_(rerank="cross", rerank_pool=20)).run(q)
```

```bash
python -m rag --eval --presets A1 --rerank cross
```

검색 결과에 재정렬 전 후보 순서(`search.pool`)와 걸린 시간(`search_ms`, `rerank_ms`)이 함께 들어 있어요.
세 방식의 비교 실험은 [experiments/rerank](./experiments/rerank)에 있어요.

## 백엔드에 연결

`rag/backend.py`가 백엔드의 검색 형식 `[{id, citation, revised, text, score}]`으로 돌려줘요.
의도 분류와 거절은 백엔드가 하므로, 여기서는 라우터를 건너뛰고 검색만 해요.

백엔드에는 이미 연결돼 있어요. 백엔드 `.env`에 `SEARCH_MODE=rag`를 넣으면 옆 폴더(`rag-main`)의 이 패키지로 검색해요.
폴더 위치가 다르면 `RAG_PATH`에 경로를 적어요.

## 측정 결과

골든셋, 조항 단위 채점, 검색 구성 A1(Dense + BM25, 필터 없음), Colab L4. 노트북과 결과 표는 [experiments/rerank](./experiments/rerank).

### 재정렬 비교 (최종 198문항: test 151 + 불확실 29 + 법령 18, 후보 50개)

| 구성 | Hit@1 | Hit@3 | Hit@5 | Hit@8 | MRR | 재정렬 시간(문항당) |
| --- | --- | --- | --- | --- | --- | --- |
| R0 재정렬 없음 | 0.354 | 0.646 | 0.712 | 0.778 | 0.505 | 0 |
| R1 ColBERT (BGE-M3) | 0.374 | 0.621 | 0.707 | 0.788 | 0.509 | 186ms (조각 벡터 미리 계산) |
| R2 크로스 인코더 (bge-reranker-v2-m3) | **0.404** | **0.682** | **0.793** | **0.828** | **0.553** | 572ms |

- 후보 50개 안에 정답이 있는 비율은 0.924예요. 재정렬로 올릴 수 있는 한계가 이 값이에요.
- test 151문항만 보면 Hit@8이 R0 0.788, R2 0.828이에요.
- 법령이 정답인 18문항은 Hit@1이 R0 0.111, R2 0.444로 차이가 커요. R0는 내부규정을 앞에 두는 설정(`internal_boost`) 때문에 법령이 뒤로 밀리고, 재정렬이 이를 바로잡아요.
- ColBERT는 R0와 차이가 없어 쓰지 않아요.
- dev 72문항(후보 20개)에서는 R0 0.875, R2 0.889(Hit@8)였어요.

### 판단

- 검색 순위 기준으로는 크로스 인코더가 낫지만, 답변 정답률까지 오르는지는 아직 확인하지 않았어요.
- 그래서 기본값은 `rerank = "none"`으로 두고, GPU 서버에서 쓸 때 `[rerank] mode = "cross"`, `pool = 50`으로 켜요.
  CPU에서는 조각 50개를 다시 읽는 데 오래 걸려서 로컬 개발에는 맞지 않아요.

### 참고: BM25만 썼을 때 (270문항)

Hit@1 0.296, Hit@8 0.648. Dense를 켜야 의미 있는 점수가 나와요.

- 내부규정 우선(`internal_boost` 0.15)은 BM25만으로 골든셋 전체에서 0 / 0.05 / 0.15를 비교해 고른 값이에요.
  평가 문항을 보고 고른 값이라 R0 점수에 조금 유리할 수 있고, Dense를 켠 상태에서 dev로 다시 정해야 해요.

## 아직 안 된 것

- **분류기(대표 질문)와 라우터의 실제 점수**: BGE-M3로 측정 필요
- **카테고리 필터 구성(B, C) 비교**: Dense를 켠 상태에서 측정 필요
- **답변까지 포함한 평가**: 검색 결과로 LLM 답변을 만들어 정답률 측정
- **임계값 3개**(`answer`, `in_scope`, `high_conf`): 실제 점수 분포를 보고 정한 값이 아니에요. dev 문항으로 보정 필요
- 사전(`rag/text.py`)은 규정을 보고 만든 초안이에요. 놓친 표현은 평가 문항이 아닌 규정을 보고 추가하세요.
