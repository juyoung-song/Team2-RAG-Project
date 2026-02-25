from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents, chunk_size=700, chunk_overlap=70):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return text_splitter.split_documents(documents)


def create_embeddings(model="text-embedding-3-small", chunk_size=100):
    return OpenAIEmbeddings(model=model, chunk_size=chunk_size)


def build_faiss_vectorstore(split_docs, embeddings):
    return FAISS.from_documents(documents=split_docs, embedding=embeddings)
