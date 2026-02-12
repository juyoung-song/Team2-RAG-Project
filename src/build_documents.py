import os
import re
import json
import unicodedata
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd
from tqdm import tqdm

# --- 설정 ---
CSV_PATH = r"./data/data_list.csv"
RAW_DIR  = r"./data/raw"  # <-- 여기를 실제 raw 폴더 경로로 수정하세요
OUT_DIR  = r"./out"
CHUNK_SIZE = 900
OVERLAP = 150

os.makedirs(OUT_DIR, exist_ok=True)

# --- 텍스트 유틸 ---
def normalize_name(s: str) -> str:
    """
    파일명 매칭용 정규화:
    - 유니코드 정규화
    - 공백/특수문자 제거
    - 확장자 제거
    - 소문자
    """
    if s is None:
        return ""
    s = str(s).strip()
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    # 경로가 섞여있으면 파일명만
    s = os.path.basename(s)
    # 확장자 제거
    s = re.sub(r"\.(pdf|hwp|hwpx)$", "", s, flags=re.IGNORECASE)
    # 공백/특수문자 제거(한글/영문/숫자만 남김)
    s = re.sub(r"[^0-9a-z가-힣]+", "", s)
    return s

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("\x00", " ")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> List[str]:
    text = clean_text(text)
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks

# --- PDF 파싱(PyMuPDF) ---
def parse_pdf_text(pdf_path: str, min_chars_per_page: int = 30) -> Tuple[str, Dict[str, Any]]:
    """
    PDF 텍스트 레이어 파싱 (OCR 아님)
    반환: (전체 텍스트, stats)
    """
    import fitz  # pymupdf

    doc = fitz.open(pdf_path)
    pages = len(doc)

    texts = []
    scanned_like_pages = 0
    total_chars = 0

    for i in range(pages):
        t = doc[i].get_text("text") or ""
        t = clean_text(t)
        c = len(t)
        total_chars += c
        if c < min_chars_per_page:
            scanned_like_pages += 1
        if t:
            # 페이지 구분자를 넣어두면 디버깅/추적이 쉬움
            texts.append(f"[PAGE {i+1}]\n{t}")

    doc.close()

    full_text = "\n\n".join(texts).strip()
    stats = {
        "pages": pages,
        "total_chars": total_chars,
        "scanned_like_pages": scanned_like_pages,
        "scanned_like_ratio": (scanned_like_pages / pages) if pages else 0.0
    }
    return full_text, stats

# --- raw 파일 스캔 ---
def scan_raw_files(raw_dir: str) -> pd.DataFrame:
    exts = {".pdf", ".hwp", ".hwpx"}
    rows = []
    for p in Path(raw_dir).rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            rows.append({
                "raw_path": str(p),
                "raw_file": p.name,
                "raw_stem": p.stem,
                "raw_ext": p.suffix.lower(),
                "norm_key": normalize_name(p.name),
            })
    return pd.DataFrame(rows)

# --- 리스트 로드 ---
def load_list(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # 매칭 키 후보: 파일명 컬럼이 있다고 가정
    if "파일명" not in df.columns:
        raise ValueError("CSV에 '파일명' 컬럼이 없습니다. (data_list.csv 컬럼명을 확인해주세요)")
    df["list_file"] = df["파일명"].astype(str)
    df["norm_key"] = df["list_file"].apply(normalize_name)
    return df

# --- 매칭(정확키 우선 + 보조키) ---
def match_files(df_list: pd.DataFrame, df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    1) norm_key exact match
    2) 실패 시: raw_file에 list_file(정규화 전 일부) 포함 여부로 약매칭(간단 버전)
    """
    raw_map = {}
    # norm_key -> raw 후보들(동명이인 방지)
    for _, r in df_raw.iterrows():
        raw_map.setdefault(r["norm_key"], []).append(r["raw_path"])

    matched_path = []
    match_type = []
    ambiguity = []

    # 약매칭 보조 인덱스(정규화 전 파일명도 조금 활용)
    raw_files = df_raw[["raw_path", "raw_file"]].to_dict("records")

    for _, row in tqdm(df_list.iterrows(), total=len(df_list), desc="Match list -> raw"):
        key = row["norm_key"]
        candidates = raw_map.get(key, [])

        if len(candidates) == 1:
            matched_path.append(candidates[0])
            match_type.append("exact_norm")
            ambiguity.append("")
            continue
        elif len(candidates) > 1:
            # 동일 key 여러개면 애매함
            matched_path.append(candidates[0])
            match_type.append("ambiguous_norm")
            ambiguity.append("multiple_candidates")
            continue

        # 약매칭: list_file의 stem(확장자 제거)을 raw_file에 포함하는지
        lf = str(row["list_file"])
        lf_stem = re.sub(r"\.(pdf|hwp|hwpx)$", "", lf, flags=re.IGNORECASE)
        lf_stem_norm = normalize_name(lf_stem)

        # 너무 짧으면 의미 없음
        hit = None
        if len(lf_stem_norm) >= 6:
            for rf in raw_files:
                if lf_stem_norm and lf_stem_norm in normalize_name(rf["raw_file"]):
                    hit = rf["raw_path"]
                    break

        if hit:
            matched_path.append(hit)
            match_type.append("weak_contains")
            ambiguity.append("")
        else:
            matched_path.append("")
            match_type.append("unmatched")
            ambiguity.append("")

    out = df_list.copy()
    out["raw_path"] = matched_path
    out["match_type"] = match_type
    out["match_note"] = ambiguity
    return out

# --- documents 생성 ---
def build_documents(matched_df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    docs: List[Dict[str, Any]] = []

    report_rows = []

    for idx, row in tqdm(matched_df.iterrows(), total=len(matched_df), desc="Build documents"):
        # document_id 안정적으로 만들기
        gonggo = str(row.get("공고 번호", "")).strip()
        chasu  = str(row.get("공고 차수", "")).strip()
        fname  = str(row.get("파일명", "")).strip()
        document_id = f"{gonggo}_{chasu}_{normalize_name(fname)}".strip("_")

        base_meta = {
            "document_id": document_id,
            "row_index": int(idx),
            "공고 번호": gonggo,
            "공고 차수": row.get("공고 차수", None),
            "사업명": row.get("사업명", ""),
            "발주 기관": row.get("발주 기관", ""),
            "공개 일자": str(row.get("공개 일자", "")),
            "입찰 참여 시작일": str(row.get("입찰 참여 시작일", "")),
            "입찰 참여 마감일": str(row.get("입찰 참여 마감일", "")),
            "사업 금액": row.get("사업 금액", None),
            "파일형식": row.get("파일형식", ""),
            "파일명": fname,
            "raw_path": row.get("raw_path", ""),
            "match_type": row.get("match_type", ""),
        }

        # 1) CSV 텍스트 우선
        csv_text = clean_text(row.get("텍스트", ""))
        used_source = None
        full_text = ""
        pdf_stats = {}

        if csv_text:
            used_source = "csv_text"
            full_text = csv_text
        else:
            # 2) 없으면 raw 파일로 채우기 (PDF만 파싱, HWP는 일단 스킵)
            raw_path = str(row.get("raw_path", "")).strip()
            if raw_path and os.path.exists(raw_path):
                ext = Path(raw_path).suffix.lower()
                if ext == ".pdf":
                    used_source = "pdf_parse"
                    full_text, pdf_stats = parse_pdf_text(raw_path)
                elif ext in (".hwp", ".hwpx"):
                    used_source = "hwp_unparsed"
                    full_text = ""  # 다음 단계에서 변환/파싱 붙이기
                else:
                    used_source = "unknown"
                    full_text = ""
            else:
                used_source = "missing"

        # 요약도 따로 넣기(있으면)
        summary = clean_text(row.get("사업 요약", ""))
        if summary:
            docs.append({
                "content": summary,
                "metadata": {**base_meta, "type": "summary", "chunk": -1, "source_type": "csv_summary"}
            })

        # 본문 청킹
        chunks = chunk_text(full_text)
        for ci, c in enumerate(chunks):
            docs.append({
                "content": c,
                "metadata": {**base_meta, "type": "body", "chunk": ci, "source_type": used_source, **({"pdf_stats": pdf_stats} if pdf_stats else {})}
            })

        report_rows.append({
            "row_index": int(idx),
            "document_id": document_id,
            "match_type": base_meta["match_type"],
            "raw_path_exists": bool(base_meta["raw_path"]) and os.path.exists(str(base_meta["raw_path"])),
            "used_source": used_source,
            "body_char_len": len(full_text),
            "chunk_count": len(chunks),
            "pdf_pages": pdf_stats.get("pages", None),
            "pdf_scanned_like_ratio": pdf_stats.get("scanned_like_ratio", None),
            "파일명": fname,
            "공고 번호": gonggo,
            "사업명": base_meta["사업명"],
        })

    report_df = pd.DataFrame(report_rows)
    return docs, report_df

def save_jsonl(docs: List[Dict[str, Any]], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

# --- 실행 ---
def main():
    df_list = load_list(CSV_PATH)
    df_raw  = scan_raw_files(RAW_DIR)

    print(f"[INFO] list rows: {len(df_list)}")
    print(f"[INFO] raw files: {len(df_raw)}")

    matched = match_files(df_list, df_raw)

    # 매칭 결과 저장
    match_out = os.path.join(OUT_DIR, "match_result.csv")
    matched.to_csv(match_out, index=False, encoding="utf-8-sig")
    print(f"[OK] saved: {match_out}")

    docs, report = build_documents(matched)

    docs_out = os.path.join(OUT_DIR, "documents.jsonl")
    save_jsonl(docs, docs_out)
    print(f"[OK] saved: {docs_out}")

    report_out = os.path.join(OUT_DIR, "quality_report.csv")
    report.to_csv(report_out, index=False, encoding="utf-8-sig")
    print(f"[OK] saved: {report_out}")

    # 콘솔 요약
    print("\n=== SUMMARY ===")
    print(report["used_source"].value_counts(dropna=False))
    print("\n=== MATCH TYPE ===")
    print(matched["match_type"].value_counts(dropna=False))

    # 품질 이슈 상위 (본문이 너무 짧은 문서)
    weak = report.sort_values("body_char_len").head(15)
    print("\n=== SHORTEST BODY (top 15) ===")
    print(weak[["row_index","used_source","body_char_len","chunk_count","match_type","파일명","사업명"]].to_string(index=False))

if __name__ == "__main__":
    main()
