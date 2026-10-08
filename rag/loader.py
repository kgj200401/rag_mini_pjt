"""법령 PDF 로더.

기본은 Docling(RFP 안내)을 쓰고, 설치되어 있지 않으면 pdfplumber로 대체한다.
반환값은 "문서 전체 텍스트(str)" 하나이며, 구조 분석은 chunker.py가 담당한다.
"""
from pathlib import Path


def load_with_docling(pdf_path: str | Path) -> str:
    from docling.document_converter import DocumentConverter

    result = DocumentConverter().convert(str(pdf_path))
    return result.document.export_to_markdown()


def load_with_pdfplumber(pdf_path: str | Path) -> str:
    import pdfplumber

    pages = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return "\n".join(pages)


def load_document(pdf_path: str | Path, backend: str = "auto") -> str:
    """backend: "docling" | "pdfplumber" | "auto"(docling 시도 후 실패하면 pdfplumber)"""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    if backend == "docling":
        return load_with_docling(pdf_path)
    if backend == "pdfplumber":
        return load_with_pdfplumber(pdf_path)

    try:
        return load_with_docling(pdf_path)
    except ImportError:
        return load_with_pdfplumber(pdf_path)
