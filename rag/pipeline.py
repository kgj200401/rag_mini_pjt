"""RAG 파이프라인: 하이브리드 검색 → 리랭킹 → LLM 답변(조문 인용)."""
from common.ai_model import get_llm_model
from rag.corpus import load_chunks
from rag.reranker import rerank
from rag.retriever import HybridRetriever

SYSTEM_PROMPT = """당신은 대한민국 법령 안내 도우미입니다.
아래 [조문]에 있는 내용만 근거로 질문에 답하세요.
- 답변 끝에 근거 조문을 "(근거: 제N조 제M항)" 형태로 표시하세요.
- 조문이 다른 조문을 가리키면("제32조를 위반한 경우", "제31조제1항에 따른") 그 연결을 따라 함께 답하세요.
  예: 어떤 의무를 위반했을 때의 조사·과태료·처벌 조문이 [조문]에 있으면 그것까지 답하세요.
- [조문]에서 답을 찾을 수 없으면 추측하지 말고 "제공된 조문에서는 확인할 수 없습니다."라고만 답하세요.
  이때 근거 조문은 절대 붙이지 마세요.
- 질문이 묻지 않은 내용은 언급하지 마세요. 질문에 답이 되는 부분만 간결하게 답하세요.
- 쉬운 우리말로 간결하게 답하세요."""

REFUSAL = "확인할 수 없습니다"


def is_refusal(answer: str) -> bool:
    """근거를 찾지 못해 거절한 답변인지 (일부만 거절한 긴 답변은 해당 없음)."""
    return REFUSAL in answer and len(answer) < 80


class RagPipeline:
    """corpus="hier": 항/호 단위로 검색하고, LLM에는 해당 조 전체를 전달한다(부모-자식)."""

    def __init__(self, tokenizer: str = "bigram", corpus: str = "hier"):
        self.retriever = HybridRetriever(tokenizer, corpus)
        self.parents = {c["id"]: c for c in load_chunks("article")}
        self.use_parent = corpus != "article"
        self.llm = get_llm_model(max_tokens=1024)

    def retrieve(self, question: str, candidates: int = 50, top_n: int = 5) -> list[dict]:
        """후보를 리랭킹한 뒤 같은 조의 조각은 하나로 합쳐 조 단위 top_n개를 반환한다."""
        ranked = rerank(question, self.retriever.search(question, candidates, candidates), candidates)
        docs: list[dict] = []
        for d in ranked:
            parent_id = d.get("parent_id", d["chunk_id"])
            if any(x["parent_id"] == parent_id for x in docs):
                continue
            parent = self.parents[parent_id]
            docs.append({**d, "parent_id": parent_id, "matched_text": d["text"],
                         "text": parent["text"] if self.use_parent else d["text"]})
            if len(docs) == top_n:
                break
        return docs

    def ask(self, question: str, top_n: int = 5) -> dict:
        docs = self.retrieve(question, top_n=top_n)
        context = "\n\n".join(d["text"] for d in docs)
        answer = self.llm.invoke(
            [
                ("system", SYSTEM_PROMPT),
                ("human", f"[조문]\n{context}\n\n[질문]\n{question}"),
            ]
        ).content
        refused = is_refusal(answer)
        return {
            "question": question,
            "answer": answer,
            "context": context,  # LLM에 실제로 전달한 조문 (평가용)
            "sources": [] if refused else [
                {
                    "article": d["article"],  # 예: 제2조(정의)
                    "unit": d.get("unit"),  # 검색에 걸린 항/호 (예: 제4호), 없으면 None
                    "chapter": d["chapter"],
                    "content": d["matched_text"],  # 실제로 검색된 조문 내용
                    "score": d["rerank_score"],
                }
                for d in docs
            ],
        }


if __name__ == "__main__":
    import sys

    from dotenv import load_dotenv

    load_dotenv(override=True)
    rag = RagPipeline()
    for q in sys.argv[1:]:
        r = rag.ask(q)
        print(f"\n질문: {q}\n답변: {r['answer']}")
        print("출처:", ", ".join(f"{s['article']}{' ' + s['unit'] if s['unit'] else ''}({s['score']:.2f})" for s in r["sources"]))
