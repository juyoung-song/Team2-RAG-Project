import json
import re
import unicodedata
from pathlib import Path

import pandas as pd
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document


def parse_pdfs_to_jsonl(pdf_dir, parsed_dir):
    """data/pdfs -> data/parsed (page 단위 jsonl 저장)"""
    pdf_dir = Path(pdf_dir)
    parsed_dir = Path(parsed_dir)
    parsed_dir.mkdir(parents=True, exist_ok=True)

    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        docs = PyMuPDFLoader(str(pdf_path)).load()
        out_path = parsed_dir / f"{pdf_path.stem}.jsonl"

        with out_path.open("w", encoding="utf-8") as f:
            for d in docs:
                row = {
                    "source_file": pdf_path.name,
                    "page": d.metadata.get("page"),
                    "text": d.page_content,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _file_key(name):
    s = str(name).rsplit(".", 1)[0]
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_documents_with_metadata(preprocessed_dir, csv_path):
    """data/preprocessed + data_list.csv 결합 -> Document 리스트"""
    preprocessed_dir = Path(preprocessed_dir)
    meta_df = pd.read_csv(csv_path)

    meta_df["file_key"] = meta_df["파일명"].apply(_file_key)
    meta_map = meta_df.set_index("file_key").to_dict("index")

    documents = []
    for fp in sorted(preprocessed_dir.glob("*.jsonl")):
        base_meta = meta_map.get(_file_key(fp.name), {})

        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                text = (row.get("text") or "").strip()
                if not text:
                    continue

                md = dict(base_meta)
                md["source_file"] = row.get("source_file")
                md["page"] = row.get("page")
                md["author"] = row.get("author")
                documents.append(Document(page_content=text, metadata=md))

    return documents
