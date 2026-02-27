import json
import re
import unicodedata
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document
from parsers.parser_text_advaned import parse_texts


def parse_pdfs_to_jsonl(pdf_dir, parsed_dir):
    """data/pdfs -> data/parsed (page 단위 jsonl 저장)"""
    from collections import defaultdict

    pdf_dir = Path(pdf_dir)
    parsed_dir = Path(parsed_dir)
    parsed_dir.mkdir(parents=True, exist_ok=True)

    by_file = defaultdict(list)
    docs = parse_texts(pdf_dir)
    for d in docs:
        source_file = d.metadata.get("source_file")
        if not source_file:
            continue
        by_file[source_file].append({
            "source_file": source_file,
            "page": d.metadata.get("page"),
            "text": d.page_content,
        })

    for source_file, rows in sorted(by_file.items()):
        out_path = parsed_dir / f"{Path(source_file).stem}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for row in rows:
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


def load_all_parsed(
    parsed_dir=None,
    csv_path=None,
    save_merged=True,
) -> list[Document]:
    """
    텍스트 + 표 + 이미지 파싱 결과를 통합하여 Document 리스트로 반환.

    - 텍스트: data/parsed/*.jsonl
    - 표:     data/parsed/table/*.jsonl
    - 이미지: data/parsed/image/*.jsonl
    - 결합본: data/parsed/merged/*.jsonl (source_file 기준)

    Returns:
        list[Document] — 전체 레코드 (metadata에 type 포함)
    """
    from collections import defaultdict

    ROOT = Path(__file__).parent.parent
    parsed_dir  = Path(parsed_dir)  if parsed_dir  else ROOT / "data" / "parsed"
    table_dir   = parsed_dir / "table"
    image_dir   = parsed_dir / "image"
    merged_dir  = parsed_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    # csv 메타데이터 (선택)
    meta_map = {}
    if csv_path:
        import pandas as pd
        meta_df = pd.read_csv(csv_path)
        meta_df["file_key"] = meta_df["파일명"].apply(_file_key)
        meta_map = meta_df.set_index("file_key").to_dict("index")

    def _read_jsonl(path, default_type):
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                r.setdefault("type", default_type)
                records.append(r)
        return records

    # ── 1. 세 가지 파싱 결과 수집 ────────────────────────
    all_records = []

    # 텍스트: data/parsed/text/*.jsonl
    text_dir = parsed_dir / "text"
    if text_dir.exists():
        for fp in sorted(text_dir.glob("*.jsonl")):
            all_records.extend(_read_jsonl(fp, "text"))
    else:
        # 이전 방식 호환 (data/parsed/*.jsonl)
        for fp in sorted(parsed_dir.glob("*.jsonl")):
            all_records.extend(_read_jsonl(fp, "text"))

    # 표
    for fp in sorted(table_dir.glob("*.jsonl")):
        all_records.extend(_read_jsonl(fp, "table"))

    # 이미지
    for fp in sorted(image_dir.glob("*.jsonl")):
        all_records.extend(_read_jsonl(fp, "image"))

    # ── 2. merged/ 저장 ──────────────────────────────────
    if save_merged:
        by_file = defaultdict(list)
        for r in all_records:
            by_file[r.get("source_file", "unknown")].append(r)

        for source_file, records in by_file.items():
            out = merged_dir / (Path(source_file).stem + ".jsonl")
            with open(out, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ── 3. Document 변환 ─────────────────────────────────
    documents = []
    for r in all_records:
        text = (r.get("text") or "").strip()
        if not text:
            continue

        base_meta = dict(meta_map.get(_file_key(r.get("source_file", "")), {}))
        base_meta.update({
            "source_file": r.get("source_file"),
            "page":        r.get("page"),
            "type":        r.get("type", "text"),
        })
        if "table_index" in r:
            base_meta["table_index"] = r["table_index"]

        documents.append(Document(page_content=text, metadata=base_meta))

    text_n  = sum(1 for d in documents if d.metadata.get("type") == "text")
    table_n = sum(1 for d in documents if d.metadata.get("type") == "table")
    image_n = sum(1 for d in documents if d.metadata.get("type") == "image")
    print(f"load_all_parsed 완료: 텍스트 {text_n} / 표 {table_n} / 이미지 {image_n} = 총 {len(documents)}개")

    return documents
