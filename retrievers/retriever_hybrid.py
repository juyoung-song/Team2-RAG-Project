from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

from retrievers.retriever_dense import build_retriever_dense


def build_retriever_hybrid(
    vectorstore,
    split_documents,
    dense_k=6,
    sparse_k=6,
    dense_weight=0.7,
    sparse_weight=0.3,
):
    """Hybrid retriever (Dense + Sparse)"""
    dense_retriever = build_retriever_dense(vectorstore, k=dense_k)
    sparse_retriever = BM25Retriever.from_documents(split_documents)
    sparse_retriever.k = sparse_k

    return EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=[dense_weight, sparse_weight],
    )
