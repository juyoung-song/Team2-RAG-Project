# src/pdf_ingest.py
# -*- coding: utf-8 -*-
"""
PDF -> (PyMuPDF) -> Markdown-ish -> 청킹 -> OpenAI Embedding -> FAISS 저장
+ (옵션) HWP chunk JSONL과 병합하여 merged FAISS 생성

✅ 완성본 특징
- [ADDED] main() + CLI 인자 지원 (import 시 자동 실행 방지)
- [ADDED] 프로젝트 루트 기준 경로 고정(상대경로 함정 방지)
- [ADDED] OpenAI 키(.env) 로딩 및 미설정 시 명확한 에러
- [CHANGED] 청킹: 문자 슬라이스 -> 문단(\n\n) 경계 우선 합치기(검색 품질 개선)
- [CHANGED] 임베딩 배치 기본값: 300 -> 100 (API 제한 회피에 유리)
- [ADDED] 산출물 덮어쓰기 정책 (--force 없으면 기존 산출물 있으면 중단)
- [ADDED] (옵션) merged chunk jsonl 생성 + merged FAISS 생성
"""

import os
import re
import json
import shutil  # [CHANGED] 병합 부분에서 필요(원본 코드에 누락되어 있었음)
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

import fitz  # pymupdf
from tqdm import tqdm
import pandas as pd

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv


# =========================================================
# 텍스트 유틸
# =========================================================
def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("\x00", " ")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def clean_markdown(md: str) -> str:
    md = clean_text(md)
    md = re.sub(r"^\s*-\s*\d+\s*-\s*$", "", md, flags=re.MULTILINE)
    md = re.sub(r"^\s*\d+\s*/\s*\d+\s*$", "", md, flags=re.MULTILINE)
    md = re.sub(r"^\s*Page\s*\d+\s*$", "", md, flags=re.MULTILINE | re.I)
    md = re.sub(r"\n{3,}", "\n\n", md).strip()
    return md


def _split_paragraphs(text: str) -> List[str]:
    """[ADDED] 문단(\n\n) 우선 분해"""
    t = clean_markdown(text)
    if not t:
        return []
    paras = [p.strip() for p in t.split("\n\n")]
    return [p for p in paras if p]


def chunk_text_paragraph_first(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    [CHANGED] 문자 슬라이싱 대신 문단 경계 우선 청킹.
    - 문단이 chunk_size보다 큰 경우만 fallback 문자 슬라이스
    - overlap은 '이전 chunk의 마지막 overlap 문자'를 다음 chunk 앞에 붙이는 방식
    """
    paras = _split_paragraphs(text)
    if not paras:
        return []

    chunks: List[str] = []
    buf = ""

    def flush():
        nonlocal buf
        if buf.strip():
            chunks.append(buf.strip())
        buf = ""

    for p in paras:
        # 아주 긴 문단은 fallback
        if len(p) > chunk_size:
            flush()
            s = 0
            while s < len(p):
                e = min(len(p), s + chunk_size)
                chunks.append(p[s:e].strip())
                if e == len(p):
                    break
                s = max(0, e - overlap)
            continue

        candidate = (buf + "\n\n" + p).strip() if buf else p
        if len(candidate) <= chunk_size:
            buf = candidate
        else:
            prev = buf
            flush()
            if overlap > 0 and prev:
                tail = prev[-overlap:]
                buf = (tail + "\n\n" + p).strip()
            else:
                buf = p

    flush()
    return chunks


# =========================================================
# PDF -> Markdown-ish (페이지 단위)
# =========================================================
def pdf_to_markdownish(pdf_path: Path, min_chars_per_page: int = 30) -> Dict[str, Any]:
    """
    PDF에서 텍스트 레이어를 추출해 Markdown-ish로 구성.
    - 페이지별로 '## Page N' 헤더를 붙여 구조 신호를 줌
    - 너무 텍스트가 적은 페이지는 스캔 의심으로 카운트
    """
    doc = fitz.open(str(pdf_path))
    pages = len(doc)

    blocks = [f"# {pdf_path.stem}"]
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
            blocks.append(f"## Page {i+1}\n{t}")

    doc.close()

    md_text = clean_markdown("\n\n".join(blocks))
    stats = {
        "pages": pages,
        "total_chars": total_chars,
        "scanned_like_pages": scanned_like_pages,
        "scanned_like_ratio": (scanned_like_pages / pages) if pages else 0.0,
    }
    return {"md": md_text, "stats": stats}


# =========================================================
# 1단: PDF -> cleaned_pdf_docs.jsonl (문서 단위)
# =========================================================
def build_clean_pdf_jsonl(raw_dir: Path, out_jsonl: Path, out_dir: Path) -> Path:
    pdf_files = sorted([p for p in Path(raw_dir).rglob("*") if p.is_file() and p.suffix.lower() == ".pdf"])
    if not pdf_files:
        raise FileNotFoundError(f".pdf 파일이 없습니다: {raw_dir}")

    report_rows = []
    with open(out_jsonl, "w", encoding="utf-8") as jf:
        for p in tqdm(pdf_files, desc="(PDF-1) PDF -> CLEAN_DOCS JSONL"):
            try:
                out = pdf_to_markdownish(p)
                md_text = out["md"]
                stats = out["stats"]

                rec = {
                    "doc_id": p.stem,
                    "content": md_text,
                    "metadata": {
                        "source_type": "pdf_clean_md",
                        "file_name": p.name,
                        "raw_path": str(p.resolve()),
                        "generated_at": datetime.now().isoformat(timespec="seconds"),
                        **stats
                    }
                }
                jf.write(json.dumps(rec, ensure_ascii=False) + "\n")

                report_rows.append({
                    "file_name": p.name,
                    "status": "ok",
                    "pages": stats["pages"],
                    "total_chars": stats["total_chars"],
                    "scanned_like_ratio": stats["scanned_like_ratio"],
                })

            except Exception as e:
                report_rows.append({
                    "file_name": p.name,
                    "status": f"fail:{type(e).__name__}",
                    "error": str(e)[:500],
                    "pages": 0,
                    "total_chars": 0,
                    "scanned_like_ratio": None,
                })

    rep = pd.DataFrame(report_rows)
    rep_csv = out_dir / "pdf_clean_stage_report.csv"
    rep.to_csv(rep_csv, index=False, encoding="utf-8-sig")

    if (rep["status"] == "ok").sum() == 0:
        raise RuntimeError(f"PDF 정제 성공 문서가 없습니다. report={rep_csv}")

    return rep_csv


# =========================================================
# 2단: cleaned_pdf_docs.jsonl -> chunks_pdf_*.jsonl
# =========================================================
def build_pdf_chunk_jsonl(clean_jsonl: Path, chunk_jsonl: Path, chunk_size: int, overlap: int) -> Path:
    report_rows = []
    with open(chunk_jsonl, "w", encoding="utf-8") as out_f:
        with open(clean_jsonl, "r", encoding="utf-8") as in_f:
            for line in tqdm(in_f, desc=f"(PDF-2) CLEAN -> CHUNKS cs={chunk_size}, ov={overlap}"):
                obj = json.loads(line)
                doc_id = obj["doc_id"]
                content = obj["content"]
                md_meta = obj.get("metadata", {})

                # [CHANGED] 문단 우선 청킹
                chunks = chunk_text_paragraph_first(content, chunk_size, overlap)

                for ci, ch in enumerate(chunks):
                    rec = {
                        "content": ch,
                        "metadata": {
                            **md_meta,
                            "doc_id": doc_id,
                            "chunk": ci,
                            "chunk_size": chunk_size,
                            "overlap": overlap,
                            "source_type": "pdf_chunk_md",
                        }
                    }
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                report_rows.append({
                    "doc_id": doc_id,
                    "file_name": md_meta.get("file_name"),
                    "pages": md_meta.get("pages"),
                    "total_chars": md_meta.get("total_chars"),
                    "chunks": len(chunks),
                    "scanned_like_ratio": md_meta.get("scanned_like_ratio"),
                })

    rep = pd.DataFrame(report_rows)
    rep_csv = chunk_jsonl.with_suffix(".report.csv")
    rep.to_csv(rep_csv, index=False, encoding="utf-8-sig")
    return rep_csv


# =========================================================
# 3단: chunk JSONL -> 임베딩 -> FAISS 저장
# =========================================================
def build_faiss_from_chunk_jsonl(
    chunk_jsonl: Path,
    faiss_dir: Path,
    embed_model: str,
    batch_size: int = 100,  # [CHANGED] 기본 300 -> 100
) -> Path:
    emb = OpenAIEmbeddings(model=embed_model)

    def iter_docs():
        with open(chunk_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                yield Document(page_content=obj["content"], metadata=obj.get("metadata", {}))

    db = None
    batch = []
    for d in tqdm(iter_docs(), desc="(PDF-3) CHUNKS -> FAISS (embed)"):
        batch.append(d)
        if len(batch) >= batch_size:
            if db is None:
                db = FAISS.from_documents(batch, emb)
            else:
                db.add_documents(batch)
            batch = []

    if batch:
        if db is None:
            db = FAISS.from_documents(batch, emb)
        else:
            db.add_documents(batch)

    if db is None:
        raise RuntimeError("Chunk JSONL이 비어있습니다.")

    faiss_dir = Path(faiss_dir)
    faiss_dir.parent.mkdir(parents=True, exist_ok=True)
    db.save_local(str(faiss_dir))
    return faiss_dir


# =========================================================
# 병합 유틸 (HWP chunk + PDF chunk)
# =========================================================
def merge_chunk_jsonls(
    out_jsonl: Path,
    sources: List[Path],
) -> int:
    """
    [ADDED] 라인 단위 단순 병합(concat)
    - 반환: 총 라인 수
    """
    total = 0
    with open(out_jsonl, "w", encoding="utf-8") as out_f:
        for src in sources:
            if not Path(src).exists():
                raise FileNotFoundError(f"병합 소스가 없습니다: {src}")
            with open(src, "r", encoding="utf-8") as in_f:
                for line in in_f:
                    out_f.write(line)
                    total += 1
    return total


# =========================================================
# CLI / main
# =========================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PDF -> Clean -> Chunk -> FAISS (+ optional merge with HWP chunks)")

    p.add_argument("--raw-dir", type=str, default=None, help="PDF가 있는 data/raw 경로")
    p.add_argument("--out-dir", type=str, default=None, help="산출물 out 경로")

    p.add_argument("--chunk-size", type=int, default=900)
    p.add_argument("--overlap", type=int, default=150)
    p.add_argument("--embed-model", type=str, default="text-embedding-3-small")
    p.add_argument("--embed-batch", type=int, default=100)

    # 단계 선택
    p.add_argument("--run-clean", action="store_true")
    p.add_argument("--run-chunk", action="store_true")
    p.add_argument("--run-faiss", action="store_true")
    p.add_argument("--all", action="store_true", help="(1)(2)(3) 전체 실행")

    # 덮어쓰기
    p.add_argument("--force", action="store_true")

    # 병합 옵션
    p.add_argument("--merge-hwp-chunks", action="store_true", help="HWP chunk JSONL과 병합")
    p.add_argument("--hwp-chunk-jsonl", type=str, default=None, help="HWP chunk jsonl 경로(미지정 시 out에서 추정)")
    p.add_argument("--build-merged-faiss", action="store_true", help="merged chunk jsonl로 merged FAISS 생성")

    return p.parse_args()


def ensure_paths(args: argparse.Namespace) -> Dict[str, Path]:
    """[ADDED] 프로젝트 루트 기준 기본 경로 고정."""
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = Path(args.raw_dir) if args.raw_dir else (project_root / "data" / "raw")
    out_dir = Path(args.out_dir) if args.out_dir else (project_root / "out")
    out_dir.mkdir(parents=True, exist_ok=True)
    return {"project_root": project_root, "raw_dir": raw_dir, "out_dir": out_dir}


def refuse_overwrite(path: Path, force: bool) -> None:
    """[ADDED] force 없으면 기존 산출물 있으면 중단."""
    if path.exists() and not force:
        raise RuntimeError(f"산출물이 이미 존재합니다: {path}\n덮어쓰려면 --force 를 사용하세요.")


def main() -> None:
    load_dotenv()  # [ADDED]
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다. .env 또는 환경변수를 확인하세요.")

    args = parse_args()
    paths = ensure_paths(args)

    RAW_DIR = paths["raw_dir"]
    OUT_DIR = paths["out_dir"]

    CHUNK_SIZE = args.chunk_size
    OVERLAP = args.overlap

    CLEAN_PDF_JSONL = OUT_DIR / "cleaned_pdf_docs.jsonl"
    CHUNK_PDF_JSONL = OUT_DIR / f"chunks_pdf_cs{CHUNK_SIZE}_ov{OVERLAP}.jsonl"
    FAISS_PDF_DIR = OUT_DIR / f"faiss_pdf_cs{CHUNK_SIZE}_ov{OVERLAP}"

    # 실행 단계 결정: 아무 플래그 없으면 전체 실행
    no_flags = not (args.run_clean or args.run_chunk or args.run_faiss or args.all)
    run_clean = args.all or args.run_clean or no_flags
    run_chunk = args.all or args.run_chunk or no_flags
    run_faiss = args.all or args.run_faiss or no_flags

    # 덮어쓰기 방지
    if run_clean:
        refuse_overwrite(CLEAN_PDF_JSONL, args.force)
    if run_chunk:
        refuse_overwrite(CHUNK_PDF_JSONL, args.force)
    if run_faiss:
        if FAISS_PDF_DIR.exists() and not args.force:
            raise RuntimeError(f"FAISS_PDF_DIR이 이미 존재합니다: {FAISS_PDF_DIR}\n덮어쓰려면 --force 사용")

    # (1) clean
    pdf_clean_report = None
    if run_clean:
        pdf_clean_report = build_clean_pdf_jsonl(RAW_DIR, CLEAN_PDF_JSONL, OUT_DIR)
        print("✅ PDF 1단 리포트:", pdf_clean_report)
    else:
        if not CLEAN_PDF_JSONL.exists():
            raise RuntimeError(f"clean jsonl이 없습니다: {CLEAN_PDF_JSONL} (run-clean 필요)")

    # (2) chunk
    pdf_chunk_report = None
    if run_chunk:
        pdf_chunk_report = build_pdf_chunk_jsonl(CLEAN_PDF_JSONL, CHUNK_PDF_JSONL, CHUNK_SIZE, OVERLAP)
        print("✅ PDF 2단 리포트:", pdf_chunk_report)
    else:
        if not CHUNK_PDF_JSONL.exists():
            raise RuntimeError(f"chunk jsonl이 없습니다: {CHUNK_PDF_JSONL} (run-chunk 필요)")

    # (3) faiss
    saved_pdf_faiss = None
    if run_faiss:
        saved_pdf_faiss = build_faiss_from_chunk_jsonl(
            CHUNK_PDF_JSONL, FAISS_PDF_DIR, args.embed_model, batch_size=args.embed_batch
        )
        print("✅ PDF 3단 FAISS :", saved_pdf_faiss)

    # (옵션) 병합
    if args.merge_hwp_chunks or args.build_merged_faiss:
        # HWP chunk jsonl 추정/지정
        if args.hwp_chunk_jsonl:
            CHUNK_HWP_JSONL = Path(args.hwp_chunk_jsonl)
        else:
            CHUNK_HWP_JSONL = OUT_DIR / f"chunks_hwp_cs{CHUNK_SIZE}_ov{OVERLAP}.jsonl"

        MERGED_CHUNK_JSONL = OUT_DIR / f"chunks_merged_hwp_pdf_cs{CHUNK_SIZE}_ov{OVERLAP}.jsonl"
        FAISS_MERGED_DIR = OUT_DIR / f"faiss_merged_hwp_pdf_cs{CHUNK_SIZE}_ov{OVERLAP}"

        # merged jsonl 생성
        refuse_overwrite(MERGED_CHUNK_JSONL, args.force)
        total_lines = merge_chunk_jsonls(MERGED_CHUNK_JSONL, [CHUNK_HWP_JSONL, CHUNK_PDF_JSONL])
        print("✅ merged jsonl:", MERGED_CHUNK_JSONL, f"(lines={total_lines})")

        # merged faiss 생성
        if args.build_merged_faiss:
            if FAISS_MERGED_DIR.exists() and not args.force:
                raise RuntimeError(f"FAISS_MERGED_DIR이 이미 존재합니다: {FAISS_MERGED_DIR}\n덮어쓰려면 --force 사용")
            saved_merged_faiss = build_faiss_from_chunk_jsonl(
                MERGED_CHUNK_JSONL, FAISS_MERGED_DIR, args.embed_model, batch_size=args.embed_batch
            )
            print("✅ merged FAISS:", saved_merged_faiss)

    print("DONE.")


if __name__ == "__main__":
    main()
