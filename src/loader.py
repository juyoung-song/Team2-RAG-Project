# src/loader.py
import json
from pathlib import Path
from functools import lru_cache
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from src.retriever_utils import load_chunk_jsonl_as_documents


# -----------------------
# 0) 프로젝트/환경 설정
# -----------------------
ROOT = Path(__file__).resolve().parents[1]

# .env 로드 (OPENAI_API_KEY 등)
load_dotenv(ROOT / ".env")

# (기본값) 필요하면 노트북에서 loader.* 상수만 덮어써도 되지만,
# 가장 깔끔한 건 .env 또는 별도 settings 모듈로 빼는 것입니다.
FAISS_DIR = ROOT / "out" / "faiss_merged_hwp_pdf_cs600_ov100"
CHUNK_JSONL_PATHS = [
    ROOT / "out" / "chunks_merged_hwp_pdf_cs600_ov100.jsonl",
]

EMBED_MODEL = "text-embedding-3-small"


# -----------------------
# 1) 유틸: 메타 정규화/포맷
# -----------------------
def _first_non_empty(*vals):
    for v in vals:
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        return v
    return None


def normalize_meta(md: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    서로 다른 파서/적재 코드에서 메타 키 이름이 흔들릴 때,
    generator 단계에서 SOURCES를 안정적으로 찍기 위한 최소 메타만 표준화합니다.
    """
    md = md or {}

    file_name = _first_non_empty(md.get("file_name"), md.get("filename"), md.get("source"), md.get("path"))
    doc_id = _first_non_empty(md.get("doc_id"), md.get("document_id"), md.get("id"))
    chunk = _first_non_empty(md.get("chunk"), md.get("chunk_id"), md.get("chunk_index"), md.get("chunk_no"))
    source_type = _first_non_empty(md.get("source_type"), md.get("type"), md.get("parser"))

    return {
        "file_name": file_name,
        "doc_id": doc_id,
        "chunk": chunk,
        "source_type": source_type,
    }


def format_docs_json_meta(docs: List[Any], max_chars_per_doc: int = 1000) -> str:
    """
    LLM에게 CONTEXT를 줄 때,
    각 chunk 앞에 META(JSON)를 붙여서 SOURCES를 '추측 없이' 출력하도록 만듭니다.
    """
    blocks = []
    for d in docs:
        meta = normalize_meta(getattr(d, "metadata", None))
        meta_line = "META: " + json.dumps(meta, ensure_ascii=False)
        text = (getattr(d, "page_content", "") or "")[:max_chars_per_doc]
        blocks.append(f"-----\n{meta_line}\nTEXT:\n{text}")
    return "\n\n".join(blocks)


# -----------------------
# 2) 캐시: 임베딩 / DB / chunk docs
# -----------------------
@lru_cache(maxsize=1)
def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBED_MODEL)


@lru_cache(maxsize=1)
def get_db() -> FAISS:
    emb = get_embeddings()
    return FAISS.load_local(FAISS_DIR, emb, allow_dangerous_deserialization=True)


@lru_cache(maxsize=1)
def get_chunk_docs(min_chars: int = 30):
    """
    BM25 용으로 jsonl chunk들을 LangChain Document 리스트로 로드합니다.
    """
    docs = load_chunk_jsonl_as_documents(CHUNK_JSONL_PATHS, min_chars=min_chars)
    return docs