def build_retriever_dense(vectorstore, k=6):
    """Dense retriever (FAISS MMR 검색)"""
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k},
    )
