"""최종으로 결정된 retriever: Hybrid"""

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever


def build_dense_retriever(vectorstore, k=12):
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k},
    )


def build_bm25_retriever(split_documents, k=12):
    """Sparse retriever"""
    bm25_retriever = BM25Retriever.from_documents(split_documents)
    bm25_retriever.k = k
    return bm25_retriever


def build_hybrid_retriever(
    vectorstore,
    split_documents,
    dense_k=12,
    sparse_k=12,
    dense_weight=0.7,
    sparse_weight=0.3,
):
    dense_retriever = build_dense_retriever(vectorstore, k=dense_k)
    bm25_retriever = build_bm25_retriever(split_documents, k=sparse_k)
    return EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        weights=[dense_weight, sparse_weight],
    )
