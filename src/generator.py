# src/generator.py
from functools import lru_cache
from typing import Any, Dict

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableBranch
from langchain_core.output_parsers import StrOutputParser

from src.loader import format_docs_json_meta
from src.retriever import rerank_docs
from src.loader import get_db  # (노출용)


# -----------------------
# 0) Generator 설정
# -----------------------
LLM_MODEL = "gpt-5-mini"
TEMPERATURE = 0

NO_EVIDENCE = "근거 문서에서 확인할 수 없습니다"


# -----------------------
# 1) Prompt / LLM
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
         "(이 경우 ANSWER/SOURCES 출력 금지)\n"
         "CONTEXT에 근거가 부분적으로라도 있으면, 그 단서 범위 내에서 가장 직접적으로 답을 구성하세요.\n"
         "답이 완전하지 않더라도 CONTEXT에 직접 언급된 사실(숫자/기간/기관명)이 있으면 그 사실만으로 답을 구성하세요\n"
         "(단, 단서가 전혀 없을 때만 NO_EVIDENCE를 출력)\n\n"
         "출력 형식(반드시 준수):\n"
         "1) ANSWER: 한글로 핵심만\n"
         "2) SOURCES: 아래 형식으로 bullet(최소 1개)\n"
         "   - file_name=<...> | doc_id=<...> | chunk=<...> | source_type=<...>\n"
         "SOURCES에는 CONTEXT의 META에 있는 값만 사용하세요. 임의로 만들지 마세요.\n"
         "가능하면 숫자/기간/요건은 원문 표현을 최대한 유지하세요."
        ),
        ("human",
         "질문: {question}\n\n"
         "CONTEXT:\n{context}"
        ),
    ])


# -----------------------
# 2) Chain 조립
# -----------------------
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
            (_has_docs, llm_chain),
            RunnableLambda(lambda _: NO_EVIDENCE),
)
    )
    return chain

def get_db_cached():
    return get_db()

# 외부에서 바로 import해서 쓰는 “고정 이름”

# 노트북/런타임에서 dotenv 로드 후 생성하는 것이 안전하므로 즉시 생성은 피함
# 필요하면 get_rag_chain() 또는 ask()를 사용하세요.
# rag_chain = get_rag_chain()

def ask(chain, question: str, use_langfuse: bool = False) -> str:
    """
    평가/노트북에서 공통으로 쓰는 얇은 래퍼.
    - chain이 None이면 내부에서 lazy 생성(get_rag_chain)
    - use_langfuse는 현재 generator가 미지원이므로 무시(호환성 유지용)
    """
    if chain is None:
        chain = get_rag_chain()
    out = chain.invoke(question)
    return out if isinstance(out, str) else str(out)