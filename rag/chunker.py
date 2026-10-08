"""법률 구조(장·절·조) 기반 청킹: 조 단위 + 조 제목 prefix.

청크 형식 (팀 합의용):
{
  "id": "article_2",
  "text": "[제1장 총칙 > 제2조(정의)]\n이 법에서 사용하는 용어의 뜻은 ...",
  "metadata": {
    "chapter": "제1장 총칙",
    "section": None,                # 절이 없으면 None
    "article_no": "제2조",          # 제17조의2 같은 가지번호 포함
    "article_title": "정의",
    "article": "제2조(정의)",       # sources 출력용
    "char_len": 1234
  }
}
"""
import re

CHAPTER_RE = re.compile(r"^제(\d+)장\s+(.+)$")
SECTION_RE = re.compile(r"^제(\d+)절\s+(.+)$")
ARTICLE_RE = re.compile(r"^제(\d+)조(?:의(\d+))?\(([^)]+)\)\s*(.*)$")
SUPPLEMENT_RE = re.compile(r"^부칙")

FOOTER_RE = re.compile(r"법제처\s*\d+\s*국가법령정보센터")
RUNNING_HEADER = "인공지능 발전과 신뢰 기반 조성 등에 관한 기본법"

# 새 줄로 시작해야 하는 항(①)·호(1.)·목(가.)·주석([...], <...>)
NEWLINE_START_RE = re.compile(
    r"^(?:[①-⑳]|\d+(?:의\d+)?\.\s*\S|[가-힣]\.\s*\S|\[|<)"
)
# 줄바꿈으로 잘린 짧은 조각("다.", ".")은 공백 없이 앞 줄에 붙인다
TINY_FRAGMENT_RE = re.compile(r"[가-힣]{0,2}[.,;]")


def clean_lines(text: str) -> list[str]:
    """페이지 머리글/바닥글과 마크다운 기호(Docling 출력 대비)를 제거한다."""
    lines = []
    for raw in text.splitlines():
        line = FOOTER_RE.sub("", raw)
        line = re.sub(r"\*\*", "", line)
        line = re.sub(r"^\s*#+\s*", "", line)
        line = re.sub(r"^\s*[-*]\s+(?=\S)", "", line)
        line = line.strip()
        if not line or line == RUNNING_HEADER:
            continue
        lines.append(line)
    return lines


# 줄 끝에서 끊겨도 단독으로 쓰이는 한 글자 단어(이 뒤에는 공백을 둔다)
STANDALONE_SYLLABLES = set("및등수중때바데것자그이각별또")
JOSA_FRAGMENTS = {"을", "를", "은", "는", "과", "와", "로", "도", "에", "서", "게", "며", "고", "다"}


def build_vocab(lines: list[str]) -> set[str]:
    """줄의 첫/끝 토큰(잘렸을 수 있음)을 뺀, 문서 안에서 온전히 쓰인 단어 집합."""
    vocab: set[str] = set()
    for line in lines:
        tokens = line.split()
        vocab.update(tokens[1:-1])
    return vocab


def need_space(prev: str, nxt: str, vocab: set[str]) -> bool:
    """줄바꿈 지점에 공백이 있었는지 추정한다. 단어 중간에서 끊겼다면 False."""
    if TINY_FRAGMENT_RE.fullmatch(nxt):
        return False
    tail, head = prev.split()[-1], nxt.split()[0]
    if tail + head in vocab:  # 예: "인"+"공지능제품", "영향"+"을", "말"+"한다."
        return False
    if head in JOSA_FRAGMENTS:
        return False  # 조사/어미 한 글자가 줄 맨 앞으로 넘어온 경우
    if len(tail) == 1 and tail not in STANDALONE_SYLLABLES:
        return False  # 한 글자 조각이 줄 끝에 남은 경우 (예: "고"+"지하여야")
    return True


def join_wrapped(lines: list[str], vocab: set[str] | None = None) -> str:
    """PDF 줄바꿈으로 끊긴 문장을 이어 붙이되, 항/호/목은 줄을 유지한다."""
    vocab = vocab or set()
    out: list[str] = []
    for line in lines:
        if not out or NEWLINE_START_RE.match(line):
            out.append(line)
        elif need_space(out[-1], line, vocab):
            out[-1] += " " + line
        else:
            out[-1] += line
    return "\n".join(out)


def build_text(chapter, section, label, body_lines, vocab=None) -> str:
    crumbs = " > ".join(c for c in (chapter, section, label) if c)
    return f"[{crumbs}]\n{join_wrapped(body_lines, vocab)}".strip()


def chunk_law(text: str) -> list[dict]:
    chapter = section = None
    chunks: list[dict] = []
    cur = None  # 현재 조: dict(no, title, chapter, section, lines)
    all_lines = clean_lines(text)
    vocab = build_vocab(all_lines)

    def flush():
        nonlocal cur
        if cur is None:
            return
        label = f"{cur['no']}({cur['title']})" if cur["title"] else cur["no"]
        body = build_text(cur["chapter"], cur["section"], label, cur["lines"], vocab)
        chunks.append(
            {
                "id": "article_" + cur["no"].replace("제", "").replace("조", "").replace("의", "_")
                if cur["no"] != "부칙"
                else "supplementary",
                "text": body,
                "metadata": {
                    "chapter": cur["chapter"],
                    "section": cur["section"],
                    "article_no": cur["no"],
                    "article_title": cur["title"],
                    "article": label,
                    "char_len": len(body),
                },
            }
        )
        cur = None

    for line in all_lines:
        m = CHAPTER_RE.match(line)
        if m:
            flush()
            chapter, section = f"제{m.group(1)}장 {m.group(2).strip()}", None
            continue
        m = SECTION_RE.match(line)
        if m:
            flush()
            section = f"제{m.group(1)}절 {m.group(2).strip()}"
            continue
        m = ARTICLE_RE.match(line)
        if m:
            flush()
            no = f"제{m.group(1)}조" + (f"의{m.group(2)}" if m.group(2) else "")
            cur = {
                "no": no,
                "title": m.group(3).strip(),
                "chapter": chapter,
                "section": section,
                "lines": [m.group(4)] if m.group(4) else [],
            }
            continue
        if SUPPLEMENT_RE.match(line):
            flush()
            cur = {"no": "부칙", "title": "", "chapter": None, "section": None, "lines": [line]}
            continue
        if cur is not None:  # 제1장 앞의 표지(법 이름, 시행일 등)는 버린다
            cur["lines"].append(line)

    flush()
    return chunks


if __name__ == "__main__":
    import sys

    from rag.loader import load_document

    path = sys.argv[1]
    backend = sys.argv[2] if len(sys.argv) > 2 else "auto"
    result = chunk_law(load_document(path, backend))
    print(f"청크 수: {len(result)}")
    for c in result[:3]:
        print("-" * 60)
        print(c["id"], c["metadata"])
        print(c["text"][:300])
