def build_retriever_naive(vectorstore, k=None):
    """NaiveRAG 방식: vectorstore 기본 retriever"""
    if k is None:
        return vectorstore.as_retriever()
    return vectorstore.as_retriever(search_kwargs={"k": k})
