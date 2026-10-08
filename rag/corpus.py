"""청킹 방식별 코퍼스 설정: 청크 파일 경로와 Qdrant 컬렉션 이름."""
import json
from pathlib import Path

CORPORA = {
    # A. 조 단위 (현재 방식)
    "article": {"path": "data/chunks.json", "collection": "law_ai_basic"},
    # B. 조 단위 + 긴 조문(1,200자 초과)만 항/호로 분할
    "split_long": {"path": "data/chunks_split.json", "collection": "law_ai_split"},
    # C. 계층형: 항/호 단위로 검색하고, LLM에는 조 전체를 전달 (500자 초과 조문을 분할)
    "hier": {"path": "data/chunks_hier.json", "collection": "law_ai_hier"},
}


def load_chunks(corpus: str = "article") -> list[dict]:
    return json.loads(Path(CORPORA[corpus]["path"]).read_text(encoding="utf-8"))
