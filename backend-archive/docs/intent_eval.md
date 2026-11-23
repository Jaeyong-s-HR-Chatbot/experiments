# 의도 분류 평가 (dev)

- 문장 수: 92
- 전체 정확도: 83.7%
- 문장당 평균 처리 시간: 0.199 ms

## 라벨별 지표

| 라벨 | 정밀도 | 재현율 | F1 | 문장 수 |
| --- | ---: | ---: | ---: | ---: |
| rule | 65.2% | 83.3% | 73.2% | 18 |
| lookup | 100.0% | 84.6% | 91.7% | 13 |
| both | 65.0% | 100.0% | 78.8% | 13 |
| event | 100.0% | 100.0% | 100.0% | 12 |
| sensitive | 100.0% | 58.3% | 73.7% | 12 |
| refuse | 100.0% | 75.0% | 85.7% | 12 |
| no_question | 100.0% | 83.3% | 90.9% | 12 |

## 혼동 행렬

| 정답 \ 예측 | rule | lookup | both | event | sensitive | refuse | no_question |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rule | 15 | 0 | 3 | 0 | 0 | 0 | 0 |
| lookup | 0 | 11 | 2 | 0 | 0 | 0 | 0 |
| both | 0 | 0 | 13 | 0 | 0 | 0 | 0 |
| event | 0 | 0 | 0 | 12 | 0 | 0 | 0 |
| sensitive | 4 | 0 | 1 | 0 | 7 | 0 | 0 |
| refuse | 3 | 0 | 0 | 0 | 0 | 9 | 0 |
| no_question | 1 | 0 | 1 | 0 | 0 | 0 | 10 |

## 도구·계산 일치율

- 조회 도구 (gold `lookup`·`both`): 22/26 (84.6%)
- 계산 종류: 9/14 (64.3%)
- 계산 슬롯 값 전체 일치: 7/14 (50.0%)

## 가장 많은 혼동

- sensitive → rule: 4건
- refuse → rule: 3건
- rule → both: 3건

## 오분류 문장

| id | 문장 | 정답 | 예측 |
| --- | --- | --- | --- |
| lookup-06-a | 내 입사일 알려 줘 | lookup | both |
| lookup-06-b | 제 부서와 직급이 어떻게 돼? | lookup | both |
| sensitive-04-a | 동료들이 저를 따돌리는 것 같아 너무 힘들어요 | sensitive | rule |
| sensitive-07-a | 회의마다 제 의견만 무시하고 단체 대화방에서도 빼요 | sensitive | both |
| sensitive-07-b | 팀원들이 저를 일부러 피하고 업무 정보를 공유하지 않아요 | sensitive | rule |
| sensitive-08-a | 상사가 큰소리로 위협해 출근하는 게 두려워요 | sensitive | rule |
| sensitive-08-b | 관리자가 반복해서 모욕적인 말을 해 잠을 못 자요 | sensitive | rule |
| refuse-08-a | 동료의 급여와 수당 얼마 나왔는지 알려 줘 | refuse | rule |
| refuse-09-a | 박지현의 가족수당과 부양가족 정보를 알려 줘 | refuse | rule |
| refuse-09-b | 최민수가 받은 가족수당 금액을 확인해 줘 | refuse | rule |
| no_question-07-b | 제 메일 주소와 계좌번호를 알려 드릴게요: seojun@example.invalid, 999-12-345678 | no_question | both |
| no_question-10-a | 주민번호와 연락처를 알려 드립니다: 900101-1234567, 010-1234-5678 | no_question | rule |
| both-11-a | 내 급여명세의 통상임금은 3,200,000원, 연장 10시간이면 예상액과 계산 기준 알려 줘 | rule | both |
| both-11-b | 내 급여자료의 통상임금 3.2백만원, 밤 10시부터 새벽 2시까지 일한 야간수당을 계산해 줘 | rule | both |
| both-14-a | 내 근무지에서 광역시 2박 3일 출장비 계산해 줘 | rule | both |
