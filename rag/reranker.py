"""Cross-encoder 리랭킹: 질문과 조문을 한 쌍으로 읽고 점수를 매겨 다시 줄 세운다."""
from functools import lru_cache

RERANK_MODEL = "BAAI/bge-reranker-v2-m3"


@lru_cache(maxsize=1)
def _model():
    import torch
    from sentence_transformers import CrossEncoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return CrossEncoder(RERANK_MODEL, device=device, max_length=1024)


def rerank(query: str, docs: list[dict], top_n: int = 5) -> list[dict]:
    """docs를 질문과의 관련도 순으로 재정렬해 top_n개만 반환한다."""
    if not docs:
        return []
    scores = _model().predict([(query, d["text"]) for d in docs], batch_size=8)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)[:top_n]
    return [{**d, "rerank_score": float(s)} for d, s in ranked]


if __name__ == "__main__":
    import sys

    from dotenv import load_dotenv

    from rag.retriever import HybridRetriever

    load_dotenv(override=True)
    retriever = HybridRetriever()
    for q in sys.argv[1:]:
        cands = retriever.search(q, top_k=20)
        print(f"\n질문: {q}")
        print("  hybrid :", [d["article_no"] for d in cands[:5]])
        print("  rerank :", [(d["article_no"], round(d["rerank_score"], 2)) for d in rerank(q, cands, 5)])
