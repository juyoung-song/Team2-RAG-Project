# conda run -p /Users/apple/Team2-RAG-Project/.conda streamlit run src/streamlit.py

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"  # macOS OpenMP 충돌 방지 (반드시 최상단)

import pickle
import sys
from pathlib import Path


import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# src/ 안에 있어도 src.모듈을 import할 수 있도록 프로젝트 루트를 Python 경로에 추가
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generator import ask, build_chain
from src.retriever import build_hybrid_retriever

load_dotenv()

# ── 경로 설정 ────────────────────────────────────────────
DB_PATH  = str(PROJECT_ROOT / "data" / "vectorstore" / "faiss_advanced")
PKL_PATH = str(PROJECT_ROOT / "data" / "vectorstore" / "split_documents_advanced.pkl")


# ── @st.cache_resource ──────────────────────────────
# @st.cache_resource를 붙이면 함수 결과를 메모리에 저장해두고
# 이후 호출부터는 저장된 값을 재사용한다 (앱이 살아있는 동안 딱 1회만 실행).
@st.cache_resource
def load_chain():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)

    with open(PKL_PATH, "rb") as f:
        split_documents = pickle.load(f)

    retriever = build_hybrid_retriever(vectorstore, split_documents)
    chain = build_chain(retriever)
    return chain


# ── UI ──────────────────────────────────────────────────
st.title("📄 RAG 문서 질의응답")
st.caption("공공기관 제안요청서(RFP) 기반 질의응답 시스템")

# ── st.session_state.messages───────────────────────
# session_state는 새로고침이나 재실행 사이에서도 값이 유지되는 저장소
if "messages" not in st.session_state:
    st.session_state.messages = []

# 이전 대화 출력
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 질문 입력
if prompt := st.chat_input("질문을 입력하세요"):

    # 사용자 메시지 표시 및 저장
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # 답변 생성
    with st.chat_message("assistant"):
        with st.spinner("검색 중..."):
            chain = load_chain()        # 캐시된 chain 가져오기 (재로드 없음)
            answer = ask(chain, prompt) # retriever → LLM 실행
        st.write(answer)

    # 어시스턴트 답변도 기록에 추가 (다음 재실행 시 화면에 표시됨)
    st.session_state.messages.append({"role": "assistant", "content": answer})
