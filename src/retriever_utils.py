from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever


# ---------------------------
# JSONL loader (schema-flexible)
# ---------------------------
def _pick_first(d: Dict[str, Any], keys: Iterable[str]) -> Optional[Any]:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _coerce_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    # list/tuple면 join
    if isinstance(x, (list, tuple)):
        return "\n".join(str(v) for v in x if v is not None)
    return str(x)


def _normalize_metadata(md: Any) -> Dict[str, Any]:
    if md is None:
        return {}
    if isinstance(md, dict):
        return md
    # metadata가 문자열/기타일 수 있어 안전 처리
    return {"_metadata": str(md)}


def load_chunk_jsonl_as_documents(
    jsonl_paths: List[Path],
    *,
    min_chars: int = 30,
    add_source_path: bool = True,
) -> List[Document]:
    """
    다양한 chunk jsonl 스키마를 흡수하는 로더.
    - page_content 후보 키: page_content, content, text, chunk, chunk_text, md
    - metadata 후보 키: metadata, meta
    - doc_id/file_name/source_type/chunk/page/page_range 등은 metadata로 합쳐짐
    """
    docs: List[Document] = []

    for p in jsonl_paths:
        p = Path(p)
        if not p.exists():
            continue

        with p.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    obj = json.loads(line)
                except Exception:
                    # JSON 파싱 실패 라인은 스킵
                    continue

                if not isinstance(obj, dict):
                    continue

                # 1) 본문 텍스트 추출 (스키마 유연)
                text = _pick_first(
                    obj,
                    keys=[
                        "page_content",
                        "content",
                        "text",
                        "chunk",
                        "chunk_text",
                        "md",
                        "markdown",
                        "body",
                    ],
                )
                page_content = _coerce_text(text).strip()

                if len(page_content) < min_chars:
                    continue

                # 2) metadata 추출 + 상위 필드 merge
                md = _normalize_metadata(_pick_first(obj, keys=["metadata", "meta"]))

                # 흔한 상위 필드들도 metadata에 합침
                for k in [
                    "doc_id",
                    "file_name",
                    "source_type",
                    "chunk",
                    "chunk_id",
                    "page",
                    "page_index",
                    "page_start",
                    "page_end",
                    "sub_page",
                    "section",
                    "title",
                ]:
                    if k in obj and k not in md:
                        md[k] = obj[k]

                if add_source_path:
                    md["_jsonl_path"] = str(p)
                    md["_jsonl_line"] = line_no

                docs.append(Document(page_content=page_content, metadata=md))

    return docs


def build_bm25_retriever(
    docs: List[Document],
    *,
    k: int = 8,
) -> BM25Retriever:
    """
    BM25Retriever 구성.
    - docs는 chunk 단위 Document 리스트
    """
    r = BM25Retriever.from_documents(docs)
    r.k = k
    return r
