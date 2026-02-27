import json
import numpy as np
import faiss
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv
import os
load_dotenv()
client = OpenAI() 

# --- 설정 (전처리 스크립트와 동일하게 맞춤) ---
RECORDS_PATH = Path("./data/out/corpus_records.json")
INDEX_SAVE_PATH = Path("./data/out/faiss.index")
EMBEDDING_MODEL = "text-embedding-3-small" #


def get_embeddings(texts):
    """텍스트 리스트를 입력받아 벡터 리스트를 반환함"""
    # API 부하를 줄이기 위해 배치 단위 처리를 추천함
    response = client.embeddings.create(
        input=texts,
        model=EMBEDDING_MODEL
    )
    return [data.embedding for data in response.data]

def create_faiss_index():
    # 1. 전처리된 데이터 로드
    if not RECORDS_PATH.exists():
        print(f"❌ 에러: {RECORDS_PATH}가 없습니다. 전처리를 먼저 진행하세요.")
        return

    with open(RECORDS_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)
    
    print(f"📦 총 {len(records)}개의 조각을 인덱싱합니다.")

    # 2. 벡터 추출 (Embedding)
    all_vectors = []
    batch_size = 100 # API 효율을 위한 배치 크기
    
    for i in tqdm(range(0, len(records), batch_size), desc="임베딩 생성 중"):
        batch_texts = [r["text"] for r in records[i : i + batch_size]]
        vectors = get_embeddings(batch_texts)
        all_vectors.extend(vectors)

    # 3. FAISS 인덱스 생성 및 데이터 추가
    dimension = len(all_vectors[0]) # text-embedding-3-small은 보통 1536 차원
    # Inner Product(IP) 인덱스는 코사인 유사도와 유사한 계산 방식을 가짐
    index = faiss.IndexFlatIP(dimension) 
    
    # 벡터 정규화 (코사인 유사도를 위해 필요)
    vectors_np = np.array(all_vectors).astype('float32')
    faiss.normalize_L2(vectors_np)
    
    index.add(vectors_np)

    # 4. 저장
    INDEX_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_SAVE_PATH))
    
    print(f"✅ 인덱스 생성 완료!")
    print(f"📍 저장 위치: {INDEX_SAVE_PATH}")
    print(f"📏 인덱스 크기: {index.ntotal}개 벡터")

if __name__ == "__main__":
    create_faiss_index()