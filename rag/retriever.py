"""BM25(키워드) 검색 + Dense 검색을 RRF로 합치는 하이브리드 검색."""
import re
from functools import lru_cache

from rank_bm25 import BM25Okapi

from rag.corpus import load_chunks
from rag.vectorstore import dense_search

RRF_K = 60


def tokenize_bigram(text: str) -> list[str]:
    """공백·기호를 지우고 2글자씩 자른다. PDF 줄바꿈으로 단어가 끊겨도 같은 토큰이 나온다."""
    s = "".join(re.findall(r"[가-힣A-Za-z0-9]+", text))
    return [s[i : i + 2] for i in range(len(s) - 1)] or [s]


@lru_cache(maxsize=1)
def _kiwi():
    from kiwipiepy import Kiwi

    return Kiwi()


def tokenize_kiwi(text: str) -> list[str]:
    """형태소 분석 후 명사·동사·어근만 남긴다."""
    return [
        t.form for t in _kiwi().tokenize(text) if t.tag[0] in "NV" or t.tag == "XR"
    ]


TOKENIZERS = {"bigram": tokenize_bigram, "kiwi": tokenize_kiwi}


class BM25Index:
    def __init__(self, tokenizer: str = "bigram", corpus: str = "article"):
        self.tokenize = TOKENIZERS[tokenizer]
        self.chunks = load_chunks(corpus)
        self.bm25 = BM25Okapi([self.tokenize(c["text"]) for c in self.chunks])

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        scores = self.bm25.get_scores(self.tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            {"score": float(scores[i]), "chunk_id": self.chunks[i]["id"],
             "text": self.chunks[i]["text"], **self.chunks[i]["metadata"]}
            for i in ranked
            if scores[i] > 0
        ]


def rrf_fuse(result_lists: list[list[dict]], k: int = RRF_K) -> list[dict]:
    """여러 검색 결과를 순위만으로 합친다: score(d) = Σ 1 / (k + rank)."""
    fused: dict[str, dict] = {}
    for results in result_lists:
        for rank, doc in enumerate(results, start=1):
            entry = fused.setdefault(doc["chunk_id"], {**doc, "rrf_score": 0.0})
            entry["rrf_score"] += 1.0 / (k + rank)
    return sorted(fused.values(), key=lambda d: d["rrf_score"], reverse=True)


class HybridRetriever:
    def __init__(self, tokenizer: str = "bigram", corpus: str = "article"):
        self.corpus = corpus
        self.bm25 = BM25Index(tokenizer, corpus)

    def search(self, query: str, top_k: int = 20, candidates: int = 20) -> list[dict]:
        dense = dense_search(query, top_k=candidates, corpus=self.corpus)
        sparse = self.bm25.search(query, top_k=candidates)
        return rrf_fuse([dense, sparse])[:top_k]


if __name__ == "__main__":
    import sys

    from dotenv import load_dotenv

    load_dotenv(override=True)
    retriever = HybridRetriever()
    for q in sys.argv[1:]:
        print(f"\n질문: {q}")
        dense = [d["article_no"] for d in dense_search(q, 5)]
        sparse = [d["article_no"] for d in retriever.bm25.search(q, 5)]
        print("  dense :", dense)
        print("  bm25  :", sparse)
        print("  hybrid:", [d["article_no"] for d in retriever.search(q, 5)])
