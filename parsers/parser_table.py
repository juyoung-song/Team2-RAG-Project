from pathlib import Path

import pdfplumber
from langchain_core.documents import Document

PDF_DIR = Path(__file__).parent.parent / "data" / "pdfs"


def _to_markdown(table: list[list]) -> str:
    def clean(cell) -> str:
        return str(cell).strip().replace("\n", " ") if cell else ""

    rows = [[clean(c) for c in row] for row in table]
    col_n = max(len(r) for r in rows)
    rows = [r + [""] * (col_n - len(r)) for r in rows]

    lines = ["| " + " | ".join(rows[0]) + " |",
             "| " + " | ".join(["---"] * col_n) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in rows[1:]]
    return "\n".join(lines)


def parse_tables(pdf_dir: Path = PDF_DIR) -> list[Document]:
    docs = []
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_num, page in enumerate(pdf.pages):
                for tbl_idx, table in enumerate(page.extract_tables()):
                    md = _to_markdown(table)
                    if not md.strip():
                        continue
                    docs.append(Document(
                        page_content=md,
                        metadata={
                            "source_file": pdf_path.name,
                            "page": page_num,
                            "table_index": tbl_idx + 1,
                            "type": "table",
                        },
                    ))
    return docs
