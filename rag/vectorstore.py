"""Dense 인덱싱/검색: 청크를 임베딩해 Qdrant에 저장하고 질문과 가까운 조문을 찾는다."""
from qdrant_client.models import Distance, PointStruct, VectorParams

from common.ai_model import get_embedding_model
from common.qdrant import get_qdrant_client
from rag.corpus import CORPORA, load_chunks


def build_index(corpus: str = "article", recreate: bool = False) -> int:
    """청크를 임베딩해 컬렉션에 저장한다. 이미 같은 개수가 있으면 건너뛴다."""
    client = get_qdrant_client()
    chunks = load_chunks(corpus)
    COLLECTION_NAME = CORPORA[corpus]["collection"]

    if client.collection_exists(COLLECTION_NAME):
        if not recreate and client.count(COLLECTION_NAME).count == len(chunks):
            return len(chunks)
        client.delete_collection(COLLECTION_NAME)

    vectors = get_embedding_model().embed_documents([c["text"] for c in chunks])
    client.create_collection(
        COLLECTION_NAME,
        vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE),
    )
    client.upsert(
        COLLECTION_NAME,
        points=[
            PointStruct(
                id=i,
                vector=vec,
                payload={"chunk_id": c["id"], "text": c["text"], **c["metadata"]},
            )
            for i, (c, vec) in enumerate(zip(chunks, vectors))
        ],
    )
    return len(chunks)


def dense_search(query: str, top_k: int = 5, corpus: str = "article") -> list[dict]:
    """질문을 임베딩해 의미가 가까운 조문 top_k개를 반환한다."""
    vec = get_embedding_model().embed_query(query)
    res = get_qdrant_client().query_points(
        CORPORA[corpus]["collection"], query=vec, limit=top_k, with_payload=True
    )
    return [{"score": p.score, **p.payload} for p in res.points]


if __name__ == "__main__":
    import sys

    from dotenv import load_dotenv

    load_dotenv(override=True)
    for name in CORPORA:
        print(f"{name}: 저장된 청크 {build_index(name, recreate='--recreate' in sys.argv)}개")
    for q in sys.argv[1:]:
        if q.startswith("--"):
            continue
        print(f"\n질문: {q}")
        for r in dense_search(q):
            print(f"  {r['score']:.3f}  {r['article']}")
