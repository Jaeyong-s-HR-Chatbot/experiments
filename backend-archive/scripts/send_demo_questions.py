"""Send the masking demonstration questions and compare their new chat-log rows."""
import ast
import json
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import dotenv_values
from psycopg.rows import dict_row

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "http://localhost:8000"
USERNAME = "e1584"
CHAT_TIMEOUT = 180

QUESTIONS = [
    {
        "group": "C", "number": 20,
        "question": "2023-01-01에 입사했는데 2026.11.12 기준 연차가 며칠이에요?",
        "expected_question": "2023-01-01에 입사했는데 2026.11.12 기준 연차가 며칠이에요?",
        "checks": {"pii_found": {}},
    },
    {
        "group": "C", "number": 21,
        "question": "기본급 3,200,000원이면 시간외수당은 얼마예요?",
        "expected_question": "기본급 3,200,000원이면 시간외수당은 얼마예요?",
        "checks": {"pii_found": {}},
    },
    {
        "group": "C", "number": 22,
        "question": "취업규칙 제23조 제2항이랑 별표 2 내용 알려 줘",
        "expected_question": "취업규칙 제23조 제2항이랑 별표 2 내용 알려 줘",
        "checks": {"pii_found": {}},
    },
    {
        "group": "C", "number": 23,
        "question": "신청번호 R000123 결재됐어요?",
        "expected_question": "신청번호 R000123 결재됐어요?",
        "checks": {"pii_found": {}},
    },
    {
        "group": "C", "number": 24,
        "question": "서울에서 부산으로 2박 3일 출장 가면 숙박비 얼마까지 나와요?",
        "expected_question": "서울에서 부산으로 2박 3일 출장 가면 숙박비 얼마까지 나와요?",
        "checks": {"pii_found": {}},
    },
    {
        "group": "C", "number": 25,
        "question": "입사 3년차 4급 대리인데 승진하려면 몇 년 더 있어야 해요?",
        "expected_question": "입사 3년차 4급 대리인데 승진하려면 몇 년 더 있어야 해요?",
        "checks": {"pii_found": {}},
    },
    {
        "group": "C", "number": 26,
        "question": "1234-5678-9012-3456 이건 카드번호 아니고 관리번호예요",
        "expected_question": "1234-5678-9012-3456 이건 카드번호 아니고 관리번호예요",
        "checks": {"pii_found": {}},
    },
    {
        "group": "C", "number": 27,
        "question": "허리 디스크로 병가를 30일 쓰면 급여가 줄어요?",
        "expected_question": "허리 디스크로 병가를 30일 쓰면 급여가 줄어요?",
        "checks": {"pii_found": {}},
    },
    {
        "group": "D", "number": 28,
        "question": "박지현 과장 연봉 얼마야?",
        "expected_question": "박지현 과장 연봉 얼마야?",
        "checks": {"pii_found": {}, "kind": "refused"},
    },
    {
        "group": "D", "number": 29,
        "question": "E1412랑 E1377 중에 누가 연차 더 많이 남았어?",
        "expected_question": "[사번]랑 [사번] 중에 누가 연차 더 많이 남았어?",
        "checks": {
            "pii_found": {"사번": 2},
            "kind": "refused",
            "question_excludes": ["E1412", "E1377"],
        },
    },
    {
        "group": "D", "number": 30,
        "question": "제 사번 E1584랑 동기 E1412 연차 비교해 줘",
        "expected_question": "제 사번 [본인사번]랑 동기 [사번] 연차 비교해 줘",
        "checks": {
            "pii_found": {"본인사번": 1, "사번": 1},
            "kind": "refused",
        },
    },
    {
        "group": "D", "number": 31,
        "question": "팀장한테 폭언을 들었어요. 제 번호 010-1234-5678로 연락 주세요",
        "expected_question": "[민감 주제 문의]",
        "checks": {
            "pii_found": {"전화": 1},
            "intent": "sensitive",
        },
    },
    {
        "group": "D", "number": 32,
        "question": "제 전화번호 010-1234-5678인데, 병가 때 진단서 필요해?",
        "expected_question": "제 전화번호 [전화]인데, 병가 때 진단서 필요해?",
        "checks": {
            "pii_found": {"전화": 1},
            "tools_used": [],
            "answer_null": False,
        },
    },
    {
        "group": "D", "number": 33,
        "question": "내 연차 몇 개 남았어?",
        "expected_question": "내 연차 몇 개 남았어?",
        "checks": {
            "intent": "lookup",
            "tools_used_nonempty": True,
            "answer_null": False,
        },
    },
]

RAW_IDENTIFIERS = ("010-1234-5678", "E1412", "E1377", "E1584")


def read_test_password():
    source = (ROOT.parent / "DB" / "generate.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "TEST_PASSWORD"
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise RuntimeError("TEST_PASSWORD 상수를 DB\\generate.py에서 찾지 못했습니다.")


def request_json(path, payload=None, token=None, timeout=30):
    headers = {"Accept": "application/json"}
    body = None
    method = "GET"
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=utf-8"
        method = "POST"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(BASE_URL + path, data=body, headers=headers, method=method)
    with urlopen(request, timeout=timeout) as response:
        content = response.read().decode("utf-8")
        return response.status, json.loads(content) if content else {}


def get_db_url():
    values = dotenv_values(ROOT / ".env")
    value = values.get("HRDB_URL")
    if not value:
        raise RuntimeError("backend\\.env에 HRDB_URL이 없습니다.")
    return value


def read_max_log_id(db_url):
    import psycopg

    with psycopg.connect(db_url, row_factory=dict_row, connect_timeout=10) as conn:
        row = conn.execute(
            "SELECT COALESCE(MAX(log_id), 0) AS max_id FROM app.chat_log"
        ).fetchone()
    return int(row["max_id"])


def read_new_logs(db_url, max_log_id):
    import psycopg

    with psycopg.connect(db_url, row_factory=dict_row, connect_timeout=10) as conn:
        return conn.execute(
            "SELECT log_id, question_masked, pii_found, intent, kind, "
            "answer_masked, tools_used "
            "FROM app.chat_log WHERE log_id > %s ORDER BY log_id",
            (max_log_id,),
        ).fetchall()


def send_questions(token):
    errors = {}
    for item in QUESTIONS:
        try:
            status, _ = request_json(
                "/chat",
                {"message": item["question"]},
                token=token,
                timeout=CHAT_TIMEOUT,
            )
            item["http_status"] = status
        except HTTPError as exc:
            item["http_status"] = exc.code
            errors[item["number"]] = f"채팅 API HTTP {exc.code}"
        except (URLError, TimeoutError, OSError) as exc:
            item["http_status"] = None
            errors[item["number"]] = f"채팅 API 응답 실패({type(exc).__name__})"
    return errors


def find_log(item, rows, used_ids):
    for row in rows:
        if row["log_id"] not in used_ids and row["question_masked"] == item["expected_question"]:
            used_ids.add(row["log_id"])
            return row
    return None


def evaluate(item, row, send_error):
    failures = []
    if send_error:
        failures.append(send_error)
    if row is None:
        failures.append("기준 log_id 이후 일치하는 로그 행 없음")
        return failures

    pii_found = row["pii_found"] or {}
    tools_used = row["tools_used"] or []
    answer_null = row["answer_masked"] is None
    if row["question_masked"] != item["expected_question"]:
        failures.append("question_masked 기대값 불일치")
    for key, expected in item["checks"].items():
        if key == "question_excludes":
            if any(fragment in row["question_masked"] for fragment in expected):
                failures.append("question_masked에 원문 사번 잔존")
        elif key == "tools_used_nonempty":
            if bool(tools_used) is not expected:
                failures.append("tools_used 비어 있음")
        elif key == "answer_null":
            if answer_null is not expected:
                failures.append(f"answer_masked NULL 기대값 불일치({expected})")
        elif key == "tools_used":
            if tools_used != expected:
                failures.append("tools_used 기대값 불일치")
        elif key == "pii_found":
            if pii_found != expected:
                failures.append("pii_found 기대값 불일치")
        elif key == "kind":
            if row["kind"] != expected:
                failures.append(f"kind 기대값 불일치({expected})")
        elif key == "intent":
            if row["intent"] != expected:
                failures.append(f"intent 기대값 불일치({expected})")
    if item.get("http_status") != 200 and not send_error:
        failures.append(f"채팅 API HTTP {item.get('http_status')}")
    return failures


def markdown_cell(value):
    return str(value).replace("|", r"\|").replace("\r", " ").replace("\n", " ")


def main():
    try:
        health_status, _ = request_json("/health", timeout=10)
        if health_status != 200:
            raise RuntimeError(f"API 헬스 확인 실패(HTTP {health_status})")
        db_url = get_db_url()
        max_log_id = read_max_log_id(db_url)
        password = read_test_password()
        login_status, login = request_json(
            "/auth/login",
            {"username": USERNAME, "password": password},
        )
        token = login.get("access_token")
        if login_status != 200 or not token:
            raise RuntimeError(f"로그인 실패(HTTP {login_status})")
    except Exception as exc:
        print(f"중단: {str(exc) if isinstance(exc, RuntimeError) else type(exc).__name__}")
        return 2

    send_errors = send_questions(token)
    try:
        new_logs = read_new_logs(db_url, max_log_id)
    except Exception as exc:
        print(f"DB 로그 조회 실패: {type(exc).__name__}")
        return 3

    used_ids = set()
    results = []
    for item in QUESTIONS:
        row = find_log(item, new_logs, used_ids)
        failures = evaluate(item, row, send_errors.get(item["number"]))
        results.append((item, row, failures))

    print("| 번호 | question_masked | pii_found | intent | kind | answer_null | tools_used | 판정 | 실패 이유 |")
    print("| ---: | --- | --- | --- | --- | --- | --- | --- | --- |")
    for item, row, failures in results:
        if row is None:
            cells = [
                str(item["number"]), "-", "-", "-", "-", "-", "-",
                "실패", "; ".join(failures),
            ]
        else:
            cells = [
                str(item["number"]),
                markdown_cell(row["question_masked"]),
                markdown_cell(json.dumps(row["pii_found"] or {}, ensure_ascii=False, sort_keys=True)),
                markdown_cell(row["intent"]),
                markdown_cell(row["kind"]),
                str(row["answer_masked"] is None),
                markdown_cell(json.dumps(row["tools_used"] or [], ensure_ascii=False)),
                "통과" if not failures else "실패",
                "; ".join(failures) or "-",
            ]
        print("| " + " | ".join(markdown_cell(cell) for cell in cells) + " |")

    passed = {
        group: sum(1 for item, _, failures in results if item["group"] == group and not failures)
        for group in ("C", "D")
    }
    print(f"\nC: {passed['C']}/8, D: {passed['D']}/6")

    matched_rows = [row for _, row, _ in results if row is not None]
    audit_ok = len(matched_rows) == len(QUESTIONS)
    for row in matched_rows:
        searchable = (row["question_masked"] or "") + (row["answer_masked"] or "")
        if any(identifier in searchable for identifier in RAW_IDENTIFIERS):
            audit_ok = False
    print(
        "원문 식별자 점검: "
        + ("통과 (14개 행 모두 question/answer에 대상 문자열 없음)"
           if audit_ok else "실패 (누락 행 또는 question/answer에 대상 문자열 잔존)")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
