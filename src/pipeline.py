"""
Advanced RAG 전체 파이프라인

[각 모듈 역할]
- loader.py       : load_all_parsed()              → 텍스트+표+이미지 JSONL 통합 로드
- preprocessor.py : preprocess_parsed_dir(src,dst) → 헤더/푸터/제어문자 제거 (선택)
- embedding.py    : split_documents()              → 청킹 700/70
                    create_embeddings()             → OpenAI 임베딩 모델 준비
                    build_faiss_vectorstore()       → FAISS 벡터 DB 생성 및 저장
- retriever.py    : build_hybrid_retriever()       → Dense(MMR) + BM25 앙상블
- generator.py    : build_chain()                  → LangChain LCEL 체인 생성
                    ask()                          → 질문 → 답변 실행
"""

import os
import pickle
from pathlib import Path

from langchain_community.vectorstores import FAISS

from src.loader     import load_all_parsed
from src.embedding  import split_documents, create_embeddings, build_faiss_vectorstore
from src.retriever  import build_hybrid_retriever
from src.generator  import build_chain, ask


# ── 경로 설정 ─────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
DB_PATH     = ROOT / "data" / "vectorstore" / "faiss_advanced"
PKL_PATH    = ROOT / "data" / "vectorstore" / "split_documents_advanced.pkl"
PARSED_DIR  = ROOT / "data" / "parsed"
CSV_PATH    = ROOT / "data" / "raw" / "data_list.csv"


def build_pipeline(
    force_rebuild: bool = False,
    chunk_size: int = 700,
    chunk_overlap: int = 70,
    use_csv: bool = True,
):
    """
    전체 Advanced RAG 파이프라인을 한 번에 구성하고 chain을 반환합니다.

    Parameters
    ----------
    force_rebuild : 기존 DB가 있어도 강제로 새로 생성할지 여부
    chunk_size    : 청킹 크기 (기본 700)
    chunk_overlap : 청킹 오버랩 (기본 70)
    use_csv       : data_list.csv 메타데이터 병합 여부

    Returns
    -------
    chain : LangChain LCEL chain (invoke로 질문하면 답변 반환)
    """

    # ── 단계 1. 벡터 DB 로드 또는 신규 생성 ──────────────
    if not force_rebuild and DB_PATH.exists() and PKL_PATH.exists():
        print("[1/4] 기존 vectorstore 로드 중...")
        embeddings = create_embeddings()
        vectorstore = FAISS.load_local(
            str(DB_PATH),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        with open(PKL_PATH, "rb") as f:
            split_docs = pickle.load(f)
        print(f"      ✓ {len(split_docs)}개 청크 로드 완료")

    else:
        # 1-1. 파싱 결과 통합 로드 (텍스트 + 표 + 이미지)
        print("[1/4] 파싱 데이터 로드 중... (텍스트+표+이미지)")
        csv_path = str(CSV_PATH) if use_csv and CSV_PATH.exists() else None
        documents = load_all_parsed(parsed_dir=str(PARSED_DIR), csv_path=csv_path)

        # 1-2. 청킹
        print(f"[2/4] 청킹 중... (chunk_size={chunk_size}, overlap={chunk_overlap})")
        split_docs = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        print(f"      ✓ {len(split_docs)}개 청크 생성")

        # 1-3. 임베딩 + FAISS DB 생성
        print("[3/4] 임베딩 및 벡터 DB 생성 중...")
        embeddings  = create_embeddings()
        vectorstore = build_faiss_vectorstore(split_docs, embeddings)

        # 1-4. 저장
        DB_PATH.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(DB_PATH))
        with open(PKL_PATH, "wb") as f:
            pickle.dump(split_docs, f)
        print(f"      ✓ 저장 완료: {DB_PATH}")

    # ── 단계 2. Retriever 생성 (Dense + BM25 Hybrid) ──────
    print("[4/4] Hybrid Retriever + Chain 생성 중...")
    retriever = build_hybrid_retriever(vectorstore, split_docs)

    # ── 단계 3. Chain 생성 ────────────────────────────────
    chain = build_chain(retriever)
    print("      ✓ 파이프라인 준비 완료")

    return chain


if __name__ == "__main__":
    # 터미널에서 직접 실행 시 간단한 테스트
    chain = build_pipeline()
    question = "고려대학교 차세대 포털 구축사업의 사업 금액은 얼마인가요?"
    answer = ask(chain, question)
    print("\n[질문]", question)
    print("[답변]", answer)
