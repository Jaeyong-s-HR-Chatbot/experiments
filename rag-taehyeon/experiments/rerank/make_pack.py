"""재정렬 실험 묶음(rerank_exp_pack.zip)을 다시 만들어요. 노트북(rerank_eval.ipynb)이 이 묶음을 읽어요.

    python experiments/rerank/make_pack.py --db ../DB-main/DB-main --golden ../goldenset

묶음에 들어가는 것
  rag_pkg/   이 저장소의 검색 패키지 (rag/, config.toml 등)
  data/      규정·법령 조각(DB 저장소의 data/regs, data/laws), 골든셋(goldenset 저장소) + 문항 묶음(split)
문항 묶음은 splits.json 을 써요. LLM 평가와 같은 나눔(seed 42)이에요.
"""
import argparse
import json
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PKG_FILES = ["config.toml", "requirements.txt", "pyproject.toml", "README.md"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="DB 저장소 폴더 (data/regs/chunks.jsonl 이 있는 곳)")
    ap.add_argument("--golden", required=True, help="goldenset 저장소 폴더 또는 golden_set.json 파일")
    ap.add_argument("--out", default=str(HERE / "rerank_exp_pack.zip"))
    a = ap.parse_args()

    db = Path(a.db)
    chunks = {"chunks_regs.jsonl": db / "data/regs/chunks.jsonl", "chunks_laws.jsonl": db / "data/laws/chunks.jsonl"}
    for p in chunks.values():
        if not p.exists():
            raise SystemExit(f"조각 파일이 없어요: {p}")
    g = Path(a.golden)
    gpath = g if g.is_file() else next(iter(sorted(g.rglob("golden_set.json"))), None)
    if gpath is None:
        raise SystemExit(f"golden_set.json 을 찾지 못했어요: {g}")
    golden = json.loads(gpath.read_text(encoding="utf-8"))
    items = golden["items"] if isinstance(golden, dict) else golden
    splits = json.loads((HERE / "splits.json").read_text(encoding="utf-8"))
    missing = [i["id"] for i in items if i["id"] not in splits]
    if missing:
        print(f"※ 묶음이 정해지지 않은 문항 {len(missing)}개는 'new'로 표시해요 (골든셋이 바뀐 경우): {missing[:5]}")
    for i in items:
        i["split"] = splits.get(i["id"], "new")

    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted((ROOT / "rag").glob("*.py")):
            z.write(f, f"rag_pkg/rag/{f.name}")
        for name in PKG_FILES:
            if (ROOT / name).exists():
                z.write(ROOT / name, f"rag_pkg/{name}")
        for name, p in chunks.items():
            z.write(p, f"data/{name}")
        z.writestr("data/golden_set.json", json.dumps(golden, ensure_ascii=False))
    n = {}
    for i in items:
        n[i["split"]] = n.get(i["split"], 0) + 1
    print(f"만들었어요: {a.out} | 문항 {len(items)}개 {n}")


if __name__ == "__main__":
    main()
