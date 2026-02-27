from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

PDF_DIR = Path(__file__).parent.parent / "data" / "pdfs"


def parse_texts(pdf_dir: Path = PDF_DIR) -> list[Document]:
    docs = []
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        pages = PyPDFLoader(str(pdf_path)).load()
        for page in pages:
            page.metadata["source_file"] = pdf_path.name
        docs.extend(pages)
    return docs
