"""조 단위 청크를 항(①)·호(1.) 단위의 작은 청크로 나눈다 (부모 조 정보 유지).

- 항이 있으면 항 단위로, 항이 없고 호만 있으면(예: 제2조 정의) 호 단위로 나눈다.
- 호 단위로 나눌 때는 조 첫머리 문장("다음과 같다")을 각 조각에 붙인다.
- 각 조각의 prefix는 "[장 > 절 > 조(제목) > 제N항]" 형태가 된다.
"""
import json
import re
from pathlib import Path

from rag.corpus import CORPORA, load_chunks

CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"
HO_RE = re.compile(r"^(\d+(?:의\d+)?)\.\s*\S")


def _groups(body: list[str], starts: list[int]) -> list[list[str]]:
    """시작 줄 번호들을 기준으로 줄들을 묶는다. 첫 시작 앞의 줄은 첫 묶음에 붙인다."""
    bounds = starts + [len(body)]
    groups = [body[bounds[i] : bounds[i + 1]] for i in range(len(starts))]
    groups[0] = body[: starts[0]] + groups[0]
    return groups


def split_chunk(chunk: dict, min_len: int) -> list[dict]:
    """min_len보다 긴 조를 항/호로 나눈다. 나눌 수 없거나 짧으면 조 전체를 그대로 반환한다."""
    meta = chunk["metadata"]
    whole = {**chunk, "metadata": {**meta, "parent_id": chunk["id"], "unit": None}}
    head, *body = chunk["text"].split("\n")
    if len(chunk["text"]) <= min_len or not body:
        return [whole]
    prefix = head[1:-1]  # 바깥 대괄호 제거

    hang = [i for i, line in enumerate(body) if line[:1] in CIRCLED]
    if len(hang) >= 2 or (hang and hang[0] > 0):
        units = [
            (f"제{CIRCLED.index(g[0][0]) + 1}항", "p", CIRCLED.index(g[0][0]) + 1, g)
            for g in _groups(body, hang)
        ]
    else:
        ho = [i for i, line in enumerate(body) if HO_RE.match(line)]
        if len(ho) < 2:
            return [whole]
        header = body[: ho[0]]
        units = []
        for i, ln in enumerate(ho):
            end = ho[i + 1] if i + 1 < len(ho) else len(body)
            no = HO_RE.match(body[ln]).group(1)
            units.append((f"제{no}호", "h", no, header + body[ln:end]))

    out = []
    for label, kind, no, lines in units:
        out.append(
            {
                "id": f"{chunk['id']}__{kind}{no}",
                "text": f"[{prefix} > {label}]\n" + "\n".join(lines),
                "metadata": {
                    **meta,
                    "parent_id": chunk["id"],
                    "unit": label,
                    "char_len": sum(len(x) for x in lines),
                },
            }
        )
    return out


def build_corpus(name: str, min_len: int) -> list[dict]:
    parents = load_chunks("article")
    chunks = [c for p in parents for c in split_chunk(p, min_len)]
    path = Path(CORPORA[name]["path"])
    path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    return chunks


if __name__ == "__main__":
    for name, min_len in (("split_long", 1200), ("hier", 500)):
        chunks = build_corpus(name, min_len)
        lens = [len(c["text"]) for c in chunks]
        print(f"{name}: 청크 {len(chunks)}개, 최대 {max(lens)}자, 평균 {sum(lens) // len(lens)}자")
