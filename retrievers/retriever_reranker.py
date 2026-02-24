import re

from retrievers.retriever_dense import build_retriever_dense
"""reranker 1차 검색 결과를 다시 점수 매겨서 순서를 재정렬하는 2차 정렬기"""

# [STEP 1] 텍스트를 간단 토큰 집합으로 변환
def _tokenize(text):
    return set(re.findall(r"[가-힣A-Za-z0-9]+", str(text or "").lower()))


# [STEP 2] 쿼리-문서 간 유사도(간단 Jaccard) 계산
def _jaccard(query_tokens, doc_text):
    doc_tokens = _tokenize(doc_text)
    if not query_tokens or not doc_tokens:
        return 0.0
    inter = len(query_tokens & doc_tokens)
    union = len(query_tokens | doc_tokens)
    return inter / union if union else 0.0


class _SimpleRerankerRetriever:
    def __init__(self, base_retriever, top_n=6):
        self.base_retriever = base_retriever
        self.top_n = top_n

    # [STEP 4] 1차 검색 결과를 점수순으로 재정렬 후 상위 top_n 반환
    def invoke(self, query):
        docs = self.base_retriever.invoke(query)
        q_tokens = _tokenize(query)
        ranked_docs = sorted(
            docs,
            key=lambda d: _jaccard(q_tokens, getattr(d, "page_content", "")),
            reverse=True,
        )
        return ranked_docs[: self.top_n]


def build_retriever_reranker(vectorstore, first_k=20, top_n=6):
    """Dense top-k 검색 후 간단 점수로 재정렬하는 retriever."""
    # [STEP 3] 먼저 Dense retriever로 넉넉히 후보를 가져옴
    base_retriever = build_retriever_dense(vectorstore, k=first_k)
    return _SimpleRerankerRetriever(base_retriever, top_n=top_n)
