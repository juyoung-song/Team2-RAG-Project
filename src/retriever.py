from functools import lru_cache
from typing import Any, List, Optional
from collections import Counter

from flashrank import Ranker, RerankRequest

from src.retriever_utils import build_bm25_retriever
from src.loader import (
    get_db,
    get_chunk_docs,
    normalize_meta,
    _first_non_empty,
)

BM25_K = 50
DENSE_K = 50

MMR_FETCH_K = 200
MMR_LAMBDA = 0.25

RERANK_MODEL = "ms-marco-MiniLM-L-12-v2"


@lru_cache(maxsize=1)
def get_bm25_retriever():
    docs = get_chunk_docs(min_chars=30)
    print(f"[BM25] loaded docs={len(docs)}")
    return build_bm25_retriever(docs, k=BM25_K)


@lru_cache(maxsize=1)
def get_dense_retriever():
    db = get_db()
    return db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": DENSE_K,
            "fetch_k": MMR_FETCH_K,
            "lambda_mult": MMR_LAMBDA,
        },
    )


def _rrf_merge(docs_a: List[Any], docs_b: List[Any], k: int = 80, rrf_k: int = 60) -> List[Any]:
    scores = {}

    def key(d):
        m = getattr(d, "metadata", None) or {}
        return (
            m.get("doc_id") or m.get("document_id") or m.get("id"),
            m.get("chunk") or m.get("chunk_id") or m.get("chunk_index"),
            (getattr(d, "page_content", "") or "")[:80],
        )

    for rank, d in enumerate(docs_a, 1):
        scores[key(d)] = scores.get(key(d), 0.0) + 1.0 / (rrf_k + rank)
    for rank, d in enumerate(docs_b, 1):
        scores[key(d)] = scores.get(key(d), 0.0) + 1.0 / (rrf_k + rank)

    rep = {}
    for d in docs_a + docs_b:
        rep.setdefault(key(d), d)

    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [rep[k_] for k_, _ in merged[:k]]


def get_hybrid_docs(question: str, k: int = 60) -> List[Any]:
    sparse = get_bm25_retriever()
    dense = get_dense_retriever()

    sparse_docs = sparse.invoke(question)  # BM25_K
    dense_docs  = dense.invoke(question)   # DENSE_K
    return _rrf_merge(sparse_docs, dense_docs, k=k)


@lru_cache(maxsize=1)
def get_ranker():
    return Ranker(model_name=RERANK_MODEL)


def _doc_id_of(d) -> Optional[str]:
    md = getattr(d, "metadata", None) or {}
    return _first_non_empty(md.get("doc_id"), md.get("document_id"), md.get("id"))


def _filter_by_top_docid(docs: List[Any], top_n: int = 2) -> List[Any]:
    c = Counter(_doc_id_of(d) for d in docs)
    top = {doc_id for doc_id, _ in c.most_common(top_n) if doc_id}
    if not top:
        return docs
    return [d for d in docs if _doc_id_of(d) in top]


def _inject_normalized_meta(docs: List[Any]) -> List[Any]:
    """
    평가/출력에서 file_name/doc_id/chunk/source_type을 안정적으로 쓰기 위해
    normalize_meta 결과를 실제 doc.metadata에 주입.
    """
    for d in docs:
        nm = normalize_meta(getattr(d, "metadata", None))
        md = getattr(d, "metadata", None) or {}
        md.update({k: v for k, v in nm.items() if v is not None})
        d.metadata = md
    return docs


def rerank_docs(
    question: str,
    top_n: int = 10,
    candidate_k: int = 60,
    docid_scope_top_n: Optional[int] = None,
) -> List[Any]:
    """
    BM25 + Dense(RRF) 후보(candidate_k)를 만든 뒤 FlashRank로 재정렬 후 top_n 반환.

    - 평가(Recall/MRR): docid_scope_top_n=None 추천 (필터 끔)
    - 생성(답변 품질): docid_scope_top_n=2 같은 스코프 제한을 켤 수 있음
    """
    docs = get_hybrid_docs(question, k=candidate_k)
    if not docs:
        return []

    docs = _inject_normalized_meta(docs)

    ranker = get_ranker()

    passages = []
    for i, d in enumerate(docs):
        meta = d.metadata  # 주입된 메타
        header = f"[file={meta.get('file_name')} doc_id={meta.get('doc_id')} chunk={meta.get('chunk')}]"
        passages.append({"id": i, "text": header + "\n" + (getattr(d, "page_content", "") or "")})

    ranked = ranker.rerank(RerankRequest(query=question, passages=passages))
    top_ids = [r["id"] for r in ranked[:top_n]]
    top_docs = [docs[i] for i in top_ids]

    if docid_scope_top_n is not None:
        top_docs = _filter_by_top_docid(top_docs, top_n=docid_scope_top_n)

    return top_docs