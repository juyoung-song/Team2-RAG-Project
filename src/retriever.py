import os
import json
import numpy as np
import faiss
from pathlib import Path
from openai import OpenAI
from collections import defaultdict

# 1. 경로 및 설정 (rag_service_final.py의 경로 준수)
BASE = Path("/mnt/c/Users/rnrud/Documents/project/Team2-RAG-Project")
INDEX_PATH = BASE / "data/out/faiss.index"
RECORDS_PATH = BASE / "data/out/corpus_records.json"

client = OpenAI() # 환경변수에 키가 설정되어 있어야 합니다.

def load_index_and_records():
    """FAISS 인덱스와 텍스트 레코드를 로드함"""
    if not INDEX_PATH.exists() or not RECORDS_PATH.exists():
        raise FileNotFoundError("인덱스나 레코드 파일이 없습니다. 전처리를 먼저 진행하세요.")
        
    index = faiss.read_index(str(INDEX_PATH))
    with open(RECORDS_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)
    return index, records

def get_embedding(text, model="text-embedding-3-small"):
    """텍스트를 벡터로 변환"""
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

def rerank_docs(query, chunks, top_n=5):
    """
    rag_service_final.py의 핵심 로직: 
    검색된 청크들을 문서(doc_id) 단위로 그룹화하여 점수가 높은 문서의 청크를 우선순위로 정렬
    """
    if not chunks:
        return []

    # 1. 문서별로 청크 점수 합산 (글로벌 라우팅 로직 반영)
    doc_scores = defaultdict(float)
    doc_chunks = defaultdict(list)
    
    for chunk in chunks:
        d_id = chunk.get("doc_id", "unknown")
        score = chunk.get("score", 0)
        doc_scores[d_id] += score
        doc_chunks[d_id].append(chunk)

    # 2. 총점이 높은 문서 순으로 정렬
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    
    # 3. 상위 문서에서 가장 관련성 높은 청크들을 순서대로 추출
    reranked = []
    for d_id, _ in sorted_docs:
        # 해당 문서 내의 청크들도 점수순 정렬
        current_doc_chunks = sorted(doc_chunks[d_id], key=lambda x: x.get("score", 0), reverse=True)
        reranked.extend(current_doc_chunks)
        if len(reranked) >= top_n:
            break
            
    return reranked[:top_n]

def retrieve_docs(query, index, records, k=160, top_n=5):
    """
    전체 프로세스: 임베딩 추출 -> 벡터 검색 -> 리랭킹
    """
    # 1. 쿼리 임베딩
    q_vector = np.array([get_embedding(query)]).astype('float32')
    faiss.normalize_L2(q_vector)
    
    # 2. 벡터 검색 (k=160 등 넓은 범위 검색)
    scores, indices = index.search(q_vector, k)
    
    # 3. 검색된 결과 매칭
    candidate_chunks = []
    for i, idx in enumerate(indices[0]):
        if idx < len(records):
            chunk = records[idx].copy()
            chunk["score"] = float(scores[0][i])
            candidate_chunks.append(chunk)
            
    # 4. 리랭킹 적용 (문서 단위 가중치 반영)
    return rerank_docs(query, candidate_chunks, top_n=top_n)