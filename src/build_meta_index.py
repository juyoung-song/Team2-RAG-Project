import re
import os
from dotenv import load_dotenv
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


# =========================
# Config
# =========================
ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "data_list.csv"

load_dotenv(ROOT / ".env")
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다. ROOT/.env 또는 환경변수를 확인하세요.")

OUT_DIR = ROOT / "out" / "faiss_meta_router"
EMBED_MODEL = "text-embedding-3-small"


# =========================
# Helpers
# =========================
def _clean_str(x: Any) -> str:
    """nan/None -> '' + 문자열 정리"""
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    s = str(x).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def _money_to_krw_text(x: Any) -> str:
    """130000000.0 / '130,000,000' / '130000000' -> '130,000,000원'"""
    s = _clean_str(x)
    if not s:
        return ""
    digits = re.sub(r"[^\d]", "", s)
    if not digits:
        return s
    n = int(digits)
    return f"{n:,}원"


def _norm_fname(x: str) -> str:
    """경로/공백 차이를 흡수하는 파일명 정규화(basename + 공백정리)"""
    s = (x or "").strip().replace("\\", "/")
    s = Path(s).name  # 경로가 섞여도 파일명만
    s = re.sub(r"\s+", " ", s)
    return s


def make_doc_key(row: Dict[str, Any]) -> str:
    """
    CSV에 document_id가 없으므로 안정적인 doc_key를 생성:
    doc_key = 공고번호_차수_파일명(basename)
    """
    notice = _clean_str(row.get("공고 번호"))
    rnd = _clean_str(row.get("공고 차수"))
    rnd_norm = re.sub(r"\.0$", "", rnd) if rnd else ""
    fname = _norm_fname(_clean_str(row.get("파일명"))).lower()

    if notice:
        return f"{notice}_{rnd_norm}_{fname}"
    # 공고번호가 없으면 파일명만이라도
    return fname


def _build_search_text(row: Dict[str, Any], doc_key: str, file_name_norm: str) -> str:
    """
    메타 검색이 잘 되도록 텍스트를 풍부하게 구성.
    - 사업명/기관/공고번호/파일명/금액/날짜 + doc_key + file_name_norm
    """
    사업명 = _clean_str(row.get("사업명"))
    발주기관 = _clean_str(row.get("발주 기관"))
    공고번호 = _clean_str(row.get("공고 번호"))
    공고차수 = _clean_str(row.get("공고 차수"))
    파일명 = _clean_str(row.get("파일명"))
    파일형식 = _clean_str(row.get("파일형식"))
    공개일자 = _clean_str(row.get("공개 일자"))
    사업금액 = _money_to_krw_text(row.get("사업 금액"))

    파일명_no_ext = re.sub(r"\.[A-Za-z0-9]+$", "", 파일명) if 파일명 else ""

    lines = [
        f"[사업명] {사업명}",
        f"[발주기관] {발주기관}",
        f"[공고번호] {공고번호}",
        f"[공고차수] {공고차수}",
        f"[파일명] {파일명}",
        f"[파일명_확장자제거] {파일명_no_ext}",
        f"[파일형식] {파일형식}",
        f"[사업금액] {사업금액}",
        f"[공개일자] {공개일자}",
        f"[doc_key] {doc_key}",
        f"[file_name_norm] {file_name_norm}",
    ]

    compact = " | ".join([
        f"사업명={사업명}",
        f"발주기관={발주기관}",
        f"공고번호={공고번호}",
        f"차수={공고차수}",
        f"금액={사업금액}",
        f"파일명={파일명}",
        f"doc_key={doc_key}",
    ])

    return "\n".join(lines + ["", "[COMPACT]", compact]).strip()


def meta_row_to_doc(row: Dict[str, Any]) -> Document:
    doc_key = make_doc_key(row)

    raw_file = _clean_str(row.get("파일명"))
    file_name_norm = _norm_fname(raw_file).lower()

    text = _build_search_text(row, doc_key=doc_key, file_name_norm=file_name_norm)

    md = {
        # 라우팅에 쓰는 핵심 키
        "doc_key": doc_key,
        "file_name": raw_file,
        "file_name_norm": file_name_norm,

        # 참고/분석용
        "raw_path": _clean_str(row.get("raw_path")),
        "notice_no": _clean_str(row.get("공고 번호")),
        "notice_round": _clean_str(row.get("공고 차수")),
        "buyer": _clean_str(row.get("발주 기관")),
        "project_name": _clean_str(row.get("사업명")),
        "budget_krw": _money_to_krw_text(row.get("사업 금액")),
        "open_date": _clean_str(row.get("공개 일자")),
        "file_type": _clean_str(row.get("파일형식")),
        "source_type": "csv_summary",
        "row_index": row.get("row_index"),
        "match_type": _clean_str(row.get("match_type")),
        "type": _clean_str(row.get("type")),
        "chunk": row.get("chunk"),
    }

    return Document(page_content=text, metadata=md)


def build_meta_faiss_from_csv(csv_path: Path, out_dir: Path) -> None:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    if "row_index" not in df.columns:
        df.insert(0, "row_index", range(len(df)))

    records = df.to_dict(orient="records")

    docs: List[Document] = []
    skipped = 0
    for r in records:
        fname = _clean_str(r.get("파일명"))
        pname = _clean_str(r.get("사업명"))
        doc_key = make_doc_key(r)

        # 정보가 너무 없으면 스킵
        if not (doc_key or fname or pname):
            skipped += 1
            continue

        docs.append(meta_row_to_doc(r))

    print(f"[META] rows={len(records)} | docs={len(docs)} | skipped={skipped}")

    out_dir.mkdir(parents=True, exist_ok=True)

    emb = OpenAIEmbeddings(model=EMBED_MODEL)
    db = FAISS.from_documents(docs, emb)
    db.save_local(out_dir)

    print(f"[META] saved FAISS meta index to: {out_dir}")


if __name__ == "__main__":
    build_meta_faiss_from_csv(CSV_PATH, OUT_DIR)