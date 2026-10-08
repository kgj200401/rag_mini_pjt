"""평가: (1) 검색 평가 Hit@K / Recall@K / MRR, (2) 답변 평가(LLM 채점).

사용:
  python -m eval.evaluate                      # 검색 평가: 청킹 3종 × 검색 3종 비교
  python -m eval.evaluate --answer [--n 10]    # 답변 평가: 정답성/관련성/근거충실성/Hallucination
  옵션: --corpora article,split_long,hier  --cand 50  --verbose

지표 정의 (Top-5 조문 기준, 같은 조의 조각은 하나로 합침)
  Hit@K    : 정답 조문이 하나라도 상위 K개에 있는 질문의 비율
  Recall@K : 질문별 (찾은 정답 수 / 필요한 정답 수)의 평균.
             require_all 질문은 정답 조문 전부가 필요하고, 그 외 질문은 정답 중 하나면 충분(대체 가능한 조문)
  MRR      : 정답이 처음 나온 순위의 역수 평균
"""
import json
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

from common.ai_model import get_llm_model
from rag.corpus import CORPORA, load_chunks
from rag.reranker import rerank
from rag.retriever import HybridRetriever
from rag.vectorstore import build_index, dense_search

K = 5
GOLDEN = Path("eval/golden_set.jsonl")
RESULT_DIR = Path("eval/results")


def arg(name: str, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


CAND = int(arg("--cand", 50))  # 리랭킹 후보 수


def load_golden() -> list[dict]:
    return [json.loads(l) for l in GOLDEN.read_text(encoding="utf-8").splitlines() if l.strip()]


def articles(docs: list[dict]) -> list[str]:
    """청크 순위를 조 번호 순위로 바꾼다 (같은 조의 중복은 첫 순위만 남김)."""
    seen: list[str] = []
    for d in docs:
        if d["article_no"] not in seen:
            seen.append(d["article_no"])
    return seen[:K]


# ───────────────────────── 1. 검색 평가 ─────────────────────────
def score_retrieval(item: dict, arts: list[str]) -> dict:
    gold = item["gold_articles"]
    found = [g for g in gold if g in arts]
    ranks = [arts.index(g) + 1 for g in found]
    need = len(gold) if item["require_all"] else 1
    return {
        "rank": min(ranks) if ranks else None,
        "recall": min(len(found), need) / need,
        "full": len(found) >= need,
    }


def retrieval_eval(corpus: str, items: list[dict]) -> dict:
    build_index(corpus)
    hybrid = HybridRetriever("bigram", corpus)
    methods = {
        "dense": lambda q: dense_search(q, CAND, corpus),
        "hybrid": lambda q: hybrid.search(q, CAND, CAND),
        "hybrid+rerank": lambda q: rerank(q, hybrid.search(q, CAND, CAND), 10),
    }
    out = {}
    for group, sub in (("쉬움", [i for i in items if i["set"] == "easy"]),
                       ("어려움", [i for i in items if i["set"] == "hard"])):
        for m, fn in methods.items():
            rs = [score_retrieval(it, articles(fn(it["question"]))) for it in sub]
            n = len(sub)
            multi = [r for it, r in zip(sub, rs) if it["require_all"]]
            out[(group, m)] = {
                "n": n,
                "hit1": sum(1 for r in rs if r["rank"] == 1) / n,
                "hit3": sum(1 for r in rs if r["rank"] and r["rank"] <= 3) / n,
                "hit5": sum(1 for r in rs if r["rank"]) / n,
                "recall5": sum(r["recall"] for r in rs) / n,
                "mrr": sum(1 / r["rank"] for r in rs if r["rank"]) / n,
                "full_multi": (sum(r["full"] for r in multi) / len(multi)) if multi else None,
                "misses": [it["question"] for it, r in zip(sub, rs) if not r["rank"]],
            }
    return out


def run_retrieval(items: list[dict]) -> None:
    corpora = arg("--corpora", ",".join(CORPORA)).split(",")
    results = {c: retrieval_eval(c, items) for c in corpora}
    for group in ("쉬움", "어려움"):
        n = next(iter(results[corpora[0]][(group, "dense")].values()))
        print(f"\n■ {group} 질문 ({n}개)")
        print(f"{'청킹':<12}{'검색':<15}{'Hit@1':>7}{'Hit@3':>7}{'Hit@5':>7}{'Recall@5':>10}{'MRR':>7}")
        for c in corpora:
            for m in ("dense", "hybrid", "hybrid+rerank"):
                r = results[c][(group, m)]
                print(f"{c:<12}{m:<15}{r['hit1']:>7.2f}{r['hit3']:>7.2f}{r['hit5']:>7.2f}"
                      f"{r['recall5']:>10.2f}{r['mrr']:>7.2f}")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "retrieval.json").write_text(json.dumps(
        {c: {f"{g}|{m}": v for (g, m), v in r.items()} for c, r in results.items()},
        ensure_ascii=False, indent=1), encoding="utf-8")
    if "--verbose" in sys.argv:
        print("\n[Top-5에 정답이 하나도 없던 질문]")
        for c in corpora:
            for (g, m), r in results[c].items():
                if m == "hybrid+rerank" and r["misses"]:
                    print(f"- {c} / {g}: {r['misses']}")


# ───────────────────────── 2. 답변 평가 (LLM 채점) ─────────────────────────
JUDGE_PROMPT = """당신은 법령 QA 시스템의 엄격한 채점자입니다. 아래 정보를 보고 [답변]을 채점하세요.

[질문]
{question}

[정답 조문 원문] (정답의 근거가 되는 조문)
{gold}

[시스템이 답변 생성에 사용한 조문]
{context}

[답변]
{answer}

채점 기준 (각 1~5점, 5가 최고):
- correctness(정답성): [정답 조문 원문]의 내용과 일치하는가. 엄격하게 채점하세요.
  · 질문이 요구한 내용 중 [정답 조문 원문]에 있는 것을 빠뜨리면 최대 3점.
  · [정답 조문 원문]에 답이 있는데 "확인할 수 없다"고 하거나 일부를 확인할 수 없다고 하면 최대 2점.
  · 핵심 사실(숫자, 주체, 요건)이 틀리면 1~2점.
  · 정답 조문이 "없음"인 질문은 "제공된 조문에서는 확인할 수 없습니다"라고만 거절해야 5점이다.
- relevance(관련성): 질문이 묻는 것에 맞게 답했는가. 불필요한 내용이 많으면 감점.
- faithfulness(근거 충실성): 답변의 모든 주장이 [시스템이 사용한 조문]에 실제로 있는가. 근거로 든 조문 번호가 맞는지도 본다.
  · 거절하면서 근거 조문을 붙이거나, 근거 표기가 실제 내용과 다르면 최대 3점.
- hallucination: [시스템이 사용한 조문]에 없는 내용을 답변이 사실처럼 말하면 true, 아니면 false.

반드시 아래 JSON 한 개만 출력하세요 (설명 문장 금지):
{{"correctness": 정수, "relevance": 정수, "faithfulness": 정수, "hallucination": true또는false, "reason": "한두 문장 근거"}}"""


def parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group()) if m else {}


def run_answer(items: list[dict]) -> None:
    from rag.pipeline import RagPipeline

    n = int(arg("--n", len(items)))
    items = items[:n] if "--n" in sys.argv and "--head" in sys.argv else items
    if "--n" in sys.argv and "--head" not in sys.argv:  # 유형이 고르게 섞이도록 균등 추출
        step = max(len(items) // n, 1)
        items = items[::step][:n]
    rag = RagPipeline()
    judge = get_llm_model(max_tokens=400)
    parents = {c["metadata"]["article_no"]: c["text"] for c in load_chunks("article")}

    rows = []
    for i, it in enumerate(items, 1):
        res = rag.ask(it["question"])
        gold = "\n\n".join(parents[g] for g in it["gold_articles"]) or "없음 (이 법과 무관하거나 이 법에 없는 질문)"
        raw = judge.invoke(JUDGE_PROMPT.format(
            question=it["question"], gold=gold, context=res["context"], answer=res["answer"])).content
        try:
            score = parse_json(raw)
        except json.JSONDecodeError:
            score = {}
        rows.append({"id": it["id"], "set": it["set"], "type": it["type"], "question": it["question"],
                     "gold_articles": it["gold_articles"], "answer": res["answer"], "context": res["context"],
                     "sources": [s["article"] + (f" {s['unit']}" if s["unit"] else "") for s in res["sources"]],
                     "score": score})
        print(f"[{i}/{len(items)}] {it['question'][:30]:<32} {score.get('correctness')}/{score.get('relevance')}/"
              f"{score.get('faithfulness')} 환각={score.get('hallucination')}", flush=True)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "answer_eval.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")

    ok = [r for r in rows if r["score"]]
    def mean(key, sub): return sum(r["score"][key] for r in sub) / len(sub) if sub else float("nan")
    print(f"\n■ 답변 평가 (채점 성공 {len(ok)}/{len(rows)}개, 1~5점)")
    print(f"{'구분':<12}{'개수':>5}{'정답성':>8}{'관련성':>8}{'근거충실':>9}{'환각 비율':>10}")
    for label, sub in [("전체", ok)] + [(s, [r for r in ok if r["set"] == s]) for s in ("easy", "hard", "out_of_scope")]:
        if sub:
            hall = sum(bool(r["score"].get("hallucination")) for r in sub) / len(sub)
            print(f"{label:<12}{len(sub):>5}{mean('correctness', sub):>8.2f}{mean('relevance', sub):>8.2f}"
                  f"{mean('faithfulness', sub):>9.2f}{hall:>10.0%}")
    print("\n[낮은 점수(정답성 또는 근거충실성 ≤ 3) 또는 환각 의심 — 직접 확인 필요]")
    for r in ok:
        s = r["score"]
        if s.get("correctness", 5) <= 3 or s.get("faithfulness", 5) <= 3 or s.get("hallucination"):
            print(f"- {r['id']} {r['question']}\n    점수 {s.get('correctness')}/{s.get('relevance')}/{s.get('faithfulness')}"
                  f" 환각={s.get('hallucination')} | {s.get('reason')}")


if __name__ == "__main__":
    load_dotenv(override=True)
    golden = load_golden()
    run_answer(golden) if "--answer" in sys.argv else run_retrieval(golden)
