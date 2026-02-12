# rag_pipeline.py
# ------------------------------------------------------------
# 목적:
# - evaluate.py 등에서 `from rag_pipeline import rag_chain`만으로 사용 가능하게 구성
# - 최초 1회 로딩 후 캐싱(FAISS 로드 / rerank 모델 로드 등)
# ------------------------------------------------------------

import os
import json
from pathlib import Path
from dotenv import load_dotenv
from functools import lru_cache
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough,
    RunnableBranch,
)
from langchain_core.output_parsers import StrOutputParser

# FlashRank (직접 rerank 방식: LangChain 래퍼 충돌 회피)
from flashrank import Ranker, RerankRequest

# -----------------------
# 0) 환경 설정 (여기만 수정)
# -----------------------

ROOT = Path(__file__).resolve().parents[1]
FAISS_DIR = ROOT / "out" / "faiss_merged_hwp_pdf_cs600_ov100"  # <- 본인 경로로
EMBED_MODEL = "text-embedding-3-small"                # <- 사용중인 임베딩
LLM_MODEL = "gpt-5-mini"
TEMPERATURE = 0

MMR_K = 12
MMR_FETCH_K = 80
MMR_LAMBDA = 0.3

RERANK_TOP_N = 8
RERANK_MODEL = "ms-marco-MiniLM-L-12-v2"

NO_EVIDENCE = "근거 문서에서 확인할 수 없습니다"

load_dotenv(ROOT / ".env")

# -----------------------
# 1) 유틸: 메타 정규화
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
    md = md or {}

    file_name = _first_non_empty(md.get("file_name"), md.get("filename"), md.get("source"), md.get("path"))
    doc_id = _first_non_empty(md.get("doc_id"), md.get("document_id"), md.get("id"))
    chunk = _first_non_empty(md.get("chunk"), md.get("chunk_id"), md.get("chunk_index"), md.get("chunk_no"))
    source_type = _first_non_empty(md.get("source_type"), md.get("type"), md.get("parser"))

    # SOURCES에 쓰는 4개 메타만 고정
    return {
        "file_name": file_name,
        "doc_id": doc_id,
        "chunk": chunk,
        "source_type": source_type,
    }

def format_docs_json_meta(docs: List[Any], max_chars_per_doc: int = 2600) -> str:
    blocks = []
    for d in docs:
        meta = normalize_meta(getattr(d, "metadata", None))
        meta_line = "META: " + json.dumps(meta, ensure_ascii=False)
        text = (getattr(d, "page_content", "") or "")[:max_chars_per_doc]
        blocks.append(f"-----\n{meta_line}\nTEXT:\n{text}")
    return "\n\n".join(blocks)


# -----------------------
# 2) 캐시: 임베딩 / DB / Retriever / Reranker
# -----------------------
@lru_cache(maxsize=1)
def get_embeddings():
    # OpenAI API Key는 환경변수 OPENAI_API_KEY 사용 권장
    # os.environ["OPENAI_API_KEY"] = "..."  # 직접 박아도 되지만 비추천
    return OpenAIEmbeddings(model=EMBED_MODEL)

@lru_cache(maxsize=1)
def get_db():
    emb = get_embeddings()
    return FAISS.load_local(FAISS_DIR, emb, allow_dangerous_deserialization=True)

@lru_cache(maxsize=1)
def get_base_retriever():
    db = get_db()
    return db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": MMR_K,
            "fetch_k": MMR_FETCH_K,
            "lambda_mult": MMR_LAMBDA,
        },
    )

@lru_cache(maxsize=1)
def get_ranker():
    return Ranker(model_name=RERANK_MODEL)


# -----------------------
# 3) rerank_docs: 후보 → 재정렬 → top_n
# -----------------------
def rerank_docs(question: str):
    base_retriever = get_base_retriever()
    docs = base_retriever.invoke(question)
    if not docs:
        return []

    ranker = get_ranker()
    passages = [{"id": i, "text": d.page_content} for i, d in enumerate(docs)]
    ranked = ranker.rerank(RerankRequest(query=question, passages=passages))

    top_ids = [r["id"] for r in ranked[:RERANK_TOP_N]]
    return [docs[i] for i in top_ids]


# -----------------------
# 4) Prompt / LLM / Chain
# -----------------------
@lru_cache(maxsize=1)
def get_llm():
    return ChatOpenAI(model=LLM_MODEL, temperature=TEMPERATURE)

@lru_cache(maxsize=1)
def get_prompt():
    return ChatPromptTemplate.from_messages([
        ("system",
         "당신은 공공입찰/RFP 문서를 근거로 답변하는 어시스턴트입니다.\n"
         "반드시 제공된 CONTEXT(발췌문) 안에서만 답하세요. CONTEXT 밖의 지식/추론/추측은 금지합니다.\n"
         f"CONTEXT에 근거가 없으면 정확히 다음 문장만 출력하세요: {NO_EVIDENCE}\n"
         "(이 경우 ANSWER/SOURCES 출력 금지)\n\n"
         "출력 형식(반드시 준수):\n"
         "1) ANSWER: 한글로 핵심만\n"
         "2) SOURCES: 아래 형식으로 bullet(최소 1개)\n"
         "   - file_name=<...> | doc_id=<...> | chunk=<...> | source_type=<...>\n"
         "SOURCES에는 CONTEXT의 META에 있는 값만 사용하세요. 임의로 만들지 마세요.\n"
         "가능하면 숫자/기간/요건은 원문 표현을 최대한 유지하세요."),
        ("human",
         "질문: {question}\n\n"
         "CONTEXT:\n{context}")
    ])

def _build_inputs(question: str) -> Dict[str, Any]:
    docs = rerank_docs(question)
    return {"question": question, "docs": docs}

def _has_docs(x: Dict[str, Any]) -> bool:
    return len(x.get("docs") or []) > 0

def _to_llm_payload(x: Dict[str, Any]) -> Dict[str, str]:
    docs = x["docs"]
    return {
        "question": x["question"],
        "context": format_docs_json_meta(docs),
    }

@lru_cache(maxsize=1)
def get_rag_chain():
    llm = get_llm()
    prompt = get_prompt()

    llm_chain = RunnableLambda(_to_llm_payload) | prompt | llm | StrOutputParser()

    chain = (
        RunnableLambda(_build_inputs)
        | RunnableBranch(
            (RunnableLambda(_has_docs), llm_chain),
            RunnableLambda(lambda _: NO_EVIDENCE),
        )
    )
    return chain


# 외부에서 바로 import해서 쓰는 “고정 이름”
rag_chain = get_rag_chain()


# (선택) evaluate/디버그용으로 노출
db = get_db()
base_retriever = get_base_retriever()
