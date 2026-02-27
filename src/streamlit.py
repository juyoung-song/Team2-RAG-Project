import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"  # macOS OpenMP 충돌 방지 (반드시 최상단)

import pickle
import sys
import time
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
DB_PATH  = str(PROJECT_ROOT / "data" / "vectorstore" / "faiss_openai")
PKL_PATH = str(PROJECT_ROOT / "data" / "vectorstore" / "split_documents.pkl")

# ── 사이드바 UI & 설정 ───────────────────────────────────
with st.sidebar:
    st.title(" RAG 세부 설정")
    st.subheader("Hybrid 검색 가중치")
    st.caption("키워드 중심(Sparse)과 문맥 중심(Dense) 검색 비율을 조절합니다.")
    
    # 가중치 슬라이더
    sparse_weight = st.slider("Sparse (키워드) 가중치", 0.0, 1.0, 0.3, 0.1)
    dense_weight = round(1.0 - sparse_weight, 1)
    st.info(f"현재 비율 -> 키워드: {sparse_weight} / 문맥: {dense_weight}")

# ── 데이터 로드 (캐싱) ────────────────────────────────────
# 무거운 DB와 파일은 앱 실행 중 딱 1번만 메모리에 올립니다.
@st.cache_resource
def load_core_data():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)

    with open(PKL_PATH, "rb") as f:
        split_documents = pickle.load(f)
        
    return vectorstore, split_documents

# ── UI 메인 화면 ──────────────────────────────────────────
st.title("공공기관 RFP 분석 RAG 시스템")
st.caption("표와 이미지가 포함된 제안요청서를 빠르고 정확하게 분석해 드립니다.")

vectorstore, split_documents = load_core_data()

if "messages" not in st.session_state:
    st.session_state.messages = []

# 이전 대화 출력
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # 출처가 저장되어 있다면 함께 출력
        if "sources" in msg and msg["sources"]:
            with st.expander("참고한 문서 출처 보기"):
                st.markdown(msg["sources"])

# 질문 입력
if prompt := st.chat_input("지원 자격이나 평가 기준에 대해 질문해 보세요."):

    # 1. 사용자 메시지 화면 표시 및 저장
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # 2. 어시스턴트 답변 생성
    with st.chat_message("assistant"):
        with st.status("답변을 생성하는 중입니다...", expanded=True) as status:
            st.write(f"벡터 DB에서 관련 문서 검색 중... (가중치 적용)")
            
            # Retriever와 Chain 로드
            retriever = build_hybrid_retriever(vectorstore, split_documents) 
            chain = build_chain(retriever)
            
            st.write("LLM에 컨텍스트 전달 및 프롬프트 분석 중...")
            
            # [수정] ask 함수 호출 시 retriever를 함께 넘겨줍니다.
            answer, source_docs = ask(
                chain=chain, 
                question=prompt, 
                use_langfuse=True, # 필요시 True로 변경
            )
            
            status.update(label="답변 생성 완료!", state="complete", expanded=False)
        
        # 최종 답변 화면 출력
        st.markdown(answer)
        
        # [추가된 부분] 출처를 마크다운 텍스트로 예쁘게 조립합니다.
        source_markdown = ""
        if source_docs:
            for i, doc in enumerate(source_docs):
                # 메타데이터에서 페이지 번호 추출 (없으면 기본값 처리)
                page_num = doc.metadata.get("page", "알 수 없음")
                
                source_markdown += f"**[출처 {i+1}] 페이지: {page_num}**\n"
                # 표 데이터(마크다운)나 텍스트 원문을 그대로 출력
                source_markdown += f"{doc.page_content}\n\n---\n"
        else:
            source_markdown = "참고할 만한 문서 출처를 찾지 못했습니다."

        # 접기/펴기(Expander) UI 안에 출처 표시
        with st.expander("참고한 문서 출처 보기"):
            st.markdown(source_markdown)

    # 3. 어시스턴트 답변 및 출처를 기록에 추가 (새로고침 시 유지용)
    st.session_state.messages.append({
        "role": "assistant", 
        "content": answer,
        "sources": source_markdown
    })
