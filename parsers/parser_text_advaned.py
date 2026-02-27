import json
from pathlib import Path

try:
    import pdfplumber
except Exception:
    pdfplumber = None

from langchain_core.documents import Document

PDF_DIR = Path(__file__).parent.parent / "data" / "pdfs"
TEXT_DIR = Path(__file__).parent.parent / "data" / "parsed" / "text"


def parse_texts(pdf_dir: Path = PDF_DIR) -> list[Document]:
    """PDF에서 텍스트를 추출합니다. 표 영역은 제외하여 중복을 방지합니다.
    결과는 data/parsed/text/*.jsonl에 저장되며, 이미 완료된 PDF는 skip합니다."""

    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    docs = []


    if pdfplumber is None:
        print("  [warn] pdfplumber 미설치: 기존 data/parsed/text jsonl을 로드합니다.")
        for pdf_path in sorted(Path(pdf_dir).glob("*.pdf")):
            out_path = TEXT_DIR / (pdf_path.stem + ".jsonl")
            if not out_path.exists():
                print(f"  [skip] {pdf_path.name} (기존 파싱 결과 없음)")
                continue
            with open(out_path, "r", encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    docs.append(Document(page_content=r["text"], metadata={
                        "source_file": r["source_file"],
                        "page": r["page"],
                        "type": "text",
                    }))
        return docs

    for pdf_path in sorted(Path(pdf_dir).glob("*.pdf")):
        out_path = TEXT_DIR / (pdf_path.stem + ".jsonl")

        # 이미 완료된 PDF는 skip
        if out_path.exists():
            print(f"  [skip] {pdf_path.name}")
            with open(out_path, "r", encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    docs.append(Document(page_content=r["text"], metadata={
                        "source_file": r["source_file"],
                        "page": r["page"],
                        "type": "text",
                    }))
            continue

        print(f"  파싱 중: {pdf_path.name}")
        pdf_docs = []

        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # 표 bounding box 수집
                table_bboxes = [t.bbox for t in page.find_tables()]

                if table_bboxes:
                    # 표 영역 안의 글자 객체 제거 후 텍스트 추출
                    def not_in_table(obj):
                        for x0, top, x1, bottom in table_bboxes:
                            if (obj.get("x0", 0) >= x0 and obj.get("top", 0) >= top and
                                    obj.get("x1", 0) <= x1 and obj.get("bottom", 0) <= bottom):
                                return False
                        return True
                    text = page.filter(not_in_table).extract_text() or ""
                else:
                    text = page.extract_text() or ""

                pdf_docs.append(Document(
                    page_content=text.strip(),
                    metadata={"source_file": pdf_path.name, "page": page_num, "type": "text"},
                ))

        # PDF 완료 즉시 저장 (중단해도 보존)
        with open(out_path, "w", encoding="utf-8") as f:
            for d in pdf_docs:
                f.write(json.dumps({
                    "source_file": d.metadata["source_file"],
                    "page": d.metadata["page"],
                    "type": "text",
                    "text": d.page_content,
                }, ensure_ascii=False) + "\n")

        docs.extend(pdf_docs)
        print(f"  → {pdf_path.name}: {len(pdf_docs)}페이지 완료")

    return docs
