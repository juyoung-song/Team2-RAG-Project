import os
import json
import re
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup
from tqdm import tqdm
import easyocr

# --- 설정 (사용자님 환경 반영) ---
HTML_ROOT_DIR = "./data/html"
TXT_ROOT_DIR = "./data/txt"  # 텍스트 파일 경로
METADATA_CSV_PATH = "./data/data_list.csv" #
OUTPUT_PATH = "./data/out/corpus_records.json"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# OCR 설정 (선택 사항)
reader = easyocr.Reader(['ko', 'en'])

def load_metadata():
    """CSV를 읽어 100개 문서 전체의 메타데이터 맵 생성"""
    df = pd.read_csv(METADATA_CSV_PATH)
    meta_map = {}
    for _, row in df.iterrows():
        # 파일명에서 확장자 제거하여 매칭용 키 생성
        raw_fname = str(row['파일명']).strip()
        fname_no_ext = os.path.splitext(raw_fname)[0]
        meta_map[fname_no_ext] = {
            "사업명": str(row.get('사업명', '')),
            "발주기관": str(row.get('발주 기관', '')),
            "예산": str(row.get('사업 금액', '')),
            "요약": str(row.get('사업 요약', ''))
        }
    return meta_map

def html_table_to_markdown(table):
    """HTML 표를 마크다운으로 변환"""
    rows = []
    for tr in table.find_all('tr'):
        cells = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
        rows.append("| " + " | ".join(cells) + " |")
    if not rows: return ""
    header_sep = "| " + " | ".join(["---"] * len(rows[0].split('|')[1:-1])) + " |"
    if len(rows) > 1: rows.insert(1, header_sep)
    return "\n" + "\n".join(rows) + "\n"

def preprocess_hybrid_100():
    all_records = []
    meta_lookup = load_metadata() #
    
    print(f"🚀 총 {len(meta_lookup)}개 문서 통합 전처리를 시작합니다.")

    for fname_no_ext, doc_meta in tqdm(meta_lookup.items(), desc="문서 처리 중"):
        html_folder = Path(HTML_ROOT_DIR) / fname_no_ext
        txt_path = Path(TXT_ROOT_DIR) / f"{fname_no_ext}.txt"
        
        main_content = ""
        source_type = ""

        # 1순위: HTML 폴더가 있는 경우 (80개 예상)
        index_file = html_folder / "index.xhtml"
        if html_folder.exists() and index_file.exists():
            source_type = "xhtml_enriched"
            with open(index_file, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f, 'html.parser')
            
            # 표 보존 처리
            for table in soup.find_all('table'):
                md_table = html_table_to_markdown(table)
                table.replace_with(f"\n\n[표 데이터]\n{md_table}\n\n")
            
            main_content = soup.get_text(separator='\n')
            
            # 이미지(bindata) OCR (선택)
            bindata_dir = html_folder / "bindata"
            if bindata_dir.exists():
                for img_path in bindata_dir.iterdir():
                    if img_path.suffix.lower() in ['.jpg', '.png', '.jpeg']:
                        res = reader.readtext(str(img_path), detail=0)
                        if res: main_content += f"\n[그림:{img_path.name}] " + " ".join(res)

        # 2순위: HTML은 없지만 TXT 파일이 있는 경우 (20개 예상)
        elif txt_path.exists():
            source_type = "txt_fallback"
            with open(txt_path, 'r', encoding='utf-8') as f:
                main_content = f.read()
        
        else:
            print(f"⚠️ {fname_no_ext} 문서를 찾을 수 없습니다. (HTML/TXT 모두 없음)")
            continue

        # 3. 메타데이터와 결합하여 청킹
        # 사업명을 본문 최상단에 배치하여 라우팅 강화
        full_text = f"사업명: {doc_meta['사업명']}\n발주기관: {doc_meta['발주기관']}\n{main_content}"
        full_text = re.sub(r'\n+', '\n', full_text).strip()

        start = 0
        while start < len(full_text):
            all_records.append({
                "doc_id": fname_no_ext,
                "text": full_text[start : start + CHUNK_SIZE],
                "meta": {**doc_meta, "source": source_type}
            })
            start += (CHUNK_SIZE - CHUNK_OVERLAP)

    # 4. 저장
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 완료! 총 {len(all_records)}개의 조각이 생성되었습니다.")

if __name__ == "__main__":
    preprocess_hybrid_100()