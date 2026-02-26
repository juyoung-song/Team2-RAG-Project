from sentence_transformers import CrossEncoder

from retrievers.retriever_dense import build_retriever_dense

"""reranker: 1차 검색 결과를 BAAI/bge-reranker-large 모델로 재정렬하는 2차 정렬기"""


class _BGERerankerRetriever:
    def __init__(self, base_retriever, top_n=6):
        self.base_retriever = base_retriever
        self.top_n = top_n
        # [STEP 1] CrossEncoder 모델 로드 (HuggingFace에서 자동 다운로드)
        self.reranker = CrossEncoder("BAAI/bge-reranker-large")

    def invoke(self, query):
        # [STEP 2] 1차 Dense 검색으로 후보 문서 가져오기
        docs = self.base_retriever.invoke(query)

        # [STEP 3] 쿼리-문서 쌍 구성 후 CrossEncoder로 점수 계산
        pairs = [(query, getattr(d, "page_content", "")) for d in docs]
        scores = self.reranker.predict(pairs)

        # [STEP 4] 점수 높은 순으로 정렬 후 상위 top_n 반환
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[: self.top_n]]


def build_retriever_reranker(vectorstore, first_k=20, top_n=6):
    """Dense top-k 검색 후 BAAI/bge-reranker-large CrossEncoder로 재정렬하는 retriever."""
    # [STEP 0] Dense retriever로 넉넉히 후보를 가져옴
    base_retriever = build_retriever_dense(vectorstore, k=first_k)
    return _BGERerankerRetriever(base_retriever, top_n=top_n)
