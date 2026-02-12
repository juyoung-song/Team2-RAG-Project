# src/hwp_ingest.py
# -*- coding: utf-8 -*-
"""
HWP -> (hwp5html) -> HTML/XHTML -> Markdown 정제 -> 청킹 -> OpenAI Embedding -> FAISS 저장

✅ 완성본 특징
- [ADDED] main() + CLI 인자 지원 (import 시 자동 실행 방지)
- [ADDED] 프로젝트 루트 기준 경로 고정(상대경로 함정 방지)
- [ADDED] OpenAI 키(.env) 로딩 및 미설정 시 명확한 에러
- [ADDED] hwp5html 존재 여부 사전 체크
- [CHANGED] 청킹: 문자 슬라이스 -> 문단(\n\n) 경계 우선 합치기(검색 품질 개선)
- [CHANGED] 임베딩 배치 기본값: 300 -> 100 (API 제한 회피에 유리)
- [ADDED] tmp(work_dir) 자동 정리 토글 (--cleanup-tmp/--no-cleanup-tmp)
- [ADDED] 산출물 덮어쓰기 정책 (--force 없으면 기존 산출물 있으면 중단)
"""

import os
import re
import json
import shutil
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

from bs4 import BeautifulSoup
from markdownify import markdownify as md
from tqdm import tqdm
import pandas as pd

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv


# =========================================================
# 안전 유틸
# =========================================================
def safe_filename(name: str) -> str:
    return "".join(c for c in name if c not in r'\/:*?"<>|').strip()


def safe_rmtree(path: Path, tmp_root: Path) -> None:
    """tmp_root 하위만 삭제 허용(안전장치)."""
    path = Path(path).resolve()
    tmp_root = Path(tmp_root).resolve()
    if path == tmp_root:
        raise RuntimeError(f"Refuse to delete tmp_root itself: {path}")
    if tmp_root not in path.parents:
        raise RuntimeError(f"Refuse to delete non-temp path: {path}")
    if path.exists():
        shutil.rmtree(path)


def require_cmd(cmd_name: str) -> None:
    """[ADDED] 외부 커맨드 존재 여부 사전 체크."""
    if shutil.which(cmd_name) is None:
        raise RuntimeError(
            f"필수 커맨드 '{cmd_name}'를 찾을 수 없습니다.\n"
            f"- 해결: pip/conda로 설치 후 PATH 인식 확인\n"
            f"- Windows: 'where {cmd_name}' 로 확인"
        )


# =========================================================
# HWP -> XHTML/HTML via hwp5html
# =========================================================
def run_hwp5html(hwp_path: Path, out_dir: Path, tmp_root: Path) -> None:
    hwp_path = Path(hwp_path)
    out_dir = Path(out_dir)

    if not hwp_path.exists():
        raise FileNotFoundError(f"HWP not found: {hwp_path}")

    if out_dir.exists():
        safe_rmtree(out_dir, tmp_root)

    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["hwp5html", "--output", str(out_dir), str(hwp_path)]
    res = subprocess.run(cmd, capture_output=True, text=True)

    if res.returncode != 0:
        raise RuntimeError(
            f"hwp5html 실패 (exit={res.returncode})\n"
            f"file: {hwp_path}\n"
            f"stdout:\n{res.stdout}\n\nstderr:\n{res.stderr}"
        )


def pick_main_markup(out_dir: Path, topn: int = 5) -> Path:
    files = list(Path(out_dir).rglob("*.html")) + list(Path(out_dir).rglob("*.xhtml"))
    if not files:
        raise FileNotFoundError(f"HTML/XHTML 결과가 없습니다: {out_dir}")

    files.sort(key=lambda p: p.stat().st_size, reverse=True)
    candidates = files[:topn]

    best = None
    best_score = -1
    for p in candidates:
        html = p.read_text(encoding="utf-8", errors="ignore")
        score = len(re.sub(r"<[^>]+>", "", html))
        if score > best_score:
            best = p
            best_score = score

    return best if best else candidates[0]


# =========================================================
# XHTML/HTML -> Markdown
# =========================================================
def clean_markdown(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    text = re.sub(r"^\s*-\s*\d+\s*-\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\s*/\s*\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*Page\s*\d+\s*$", "", text, flags=re.MULTILINE | re.I)

    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def html_to_markdown(markup_path: Path) -> str:
    html = Path(markup_path).read_text(encoding="utf-8", errors="ignore")

    soup = None
    for parser in ("xml", "lxml-xml", "lxml", "html.parser"):
        try:
            soup = BeautifulSoup(html, parser)
            break
        except Exception:
            continue
    if soup is None:
        soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    target = soup.body if getattr(soup, "body", None) else soup
    md_text = md(str(target), heading_style="ATX")
    return clean_markdown(md_text)


# =========================================================
# RFP 정제 (사용자 코드 기반)
# =========================================================
def normalize_toc(md_text: str) -> str:
    if "목 차" not in md_text:
        return md_text

    idx = md_text.find("목 차")
    start = max(0, idx - 1500)
    end = min(len(md_text), idx + 2000)
    chunk = md_text[start:end]

    items = re.findall(r"(\d+)\.\s*([^\d\n]{2,}?)\s+(\d{1,3})", chunk)
    if not items:
        return md_text

    toc_lines = ["## 목차"]
    for n, title, page in items:
        title = re.sub(r"\s+", " ", title).strip()
        toc_lines.append(f"- {n}. {title} (p.{page})")

    toc_block = "\n".join(toc_lines) + "\n"

    m = re.search(r"목\s*차", chunk)
    if not m:
        return md_text

    last = None
    for it in re.finditer(r"\d+\.\s*[^\d\n]{2,}?\s+\d{1,3}", chunk):
        last = it
    if not last:
        return md_text

    cut_s = start + m.start()
    cut_e = start + last.end()
    return md_text[:cut_s] + toc_block + md_text[cut_e:]


def clean_rfp_markdown(md_text: str) -> str:
    text = md_text
    text = re.sub(r'^\s*xml version=.*?\?\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'!\[.*?\]\(.*?\)\s*', '', text)

    def drop_empty_tables(t: str) -> str:
        lines = t.splitlines()
        out = []
        i = 0
        while i < len(lines):
            if "|" in lines[i]:
                j = i
                block = []
                while j < len(lines) and ("|" in lines[j] or re.match(r'^\s*$', lines[j])):
                    block.append(lines[j])
                    j += 1

                block_text = "\n".join(block)
                if re.search(r"\|\s*---", block_text):
                    content = re.sub(r"\|\s*---.*", "", block_text)
                    content = re.sub(r"[|\s\-:]", "", content)
                    if len(content) < 10:
                        i = j
                        continue

                out.extend(block)
                i = j
            else:
                out.append(lines[i])
                i += 1
        return "\n".join(out)

    text = drop_empty_tables(text)
    text = normalize_toc(text)

    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\n\s+\n", "\n\n", text).strip()
    return text


# =========================================================
# 1단: HWP -> cleaned_doc JSONL
# =========================================================
def convert_one_hwp_to_clean_md(
    hwp_path: Path,
    tmp_root: Path,
    cleanup_tmp: bool = True,   # [ADDED]
) -> Tuple[str, Path]:
    """
    [CHANGED] tmp 정리 토글 지원
    """
    work_dir = Path(tmp_root) / safe_filename(hwp_path.stem)
    try:
        run_hwp5html(hwp_path, work_dir, tmp_root=tmp_root)
        main_markup = pick_main_markup(work_dir, topn=5)

        md_text = html_to_markdown(main_markup)
        cleaned = clean_rfp_markdown(md_text)
        return cleaned, main_markup
    finally:
        # [ADDED] 성공/실패와 무관하게 tmp 정리(토글 가능)
        if cleanup_tmp:
            try:
                safe_rmtree(work_dir, tmp_root)
            except Exception:
                # tmp 정리는 best-effort: 여기서 전체 파이프라인 죽이지 않음
                pass


def save_doc_artifacts(cleaned: str, hwp_path: Path, out_md_dir: Path, main_markup: Path) -> dict:
    out_md_dir.mkdir(parents=True, exist_ok=True)
    base = safe_filename(hwp_path.stem)
    md_path = out_md_dir / f"{base}.clean.md"
    meta_path = out_md_dir / f"{base}.meta.json"

    md_path.write_text(cleaned, encoding="utf-8")

    meta = {
        "source_hwp": str(hwp_path.resolve()),
        "main_markup": str(main_markup.resolve()),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "output_md": str(md_path.resolve()),
        "char_len": len(cleaned),
        "pipeline": "hwp5html -> pick_main_markup -> html_to_markdown(body) -> clean_rfp_markdown",
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def build_clean_docs_jsonl(
    raw_dir: Path,
    tmp_root: Path,
    out_md_dir: Path,
    clean_jsonl: Path,
    cleanup_tmp: bool = True,   # [ADDED]
) -> Path:
    hwp_files = sorted([p for p in Path(raw_dir).rglob("*") if p.is_file() and p.suffix.lower() == ".hwp"])
    if not hwp_files:
        raise FileNotFoundError(f".hwp 파일이 없습니다: {raw_dir}")

    report_rows = []
    with open(clean_jsonl, "w", encoding="utf-8") as jf:
        for p in tqdm(hwp_files, desc="(1) HWP -> CLEAN_DOCS JSONL"):
            try:
                cleaned, main_markup = convert_one_hwp_to_clean_md(
                    p, tmp_root, cleanup_tmp=cleanup_tmp
                )
                meta = save_doc_artifacts(cleaned, p, out_md_dir, main_markup)

                rec = {
                    "doc_id": safe_filename(p.stem),
                    "content": cleaned,
                    "metadata": {
                        "source_type": "hwp_clean_md",
                        "file_name": p.name,
                        "raw_path": str(p.resolve()),
                        "generated_at": meta["generated_at"],
                        "main_markup": meta["main_markup"],
                        "char_len": meta["char_len"],
                    }
                }
                jf.write(json.dumps(rec, ensure_ascii=False) + "\n")

                report_rows.append({
                    "file_name": p.name,
                    "status": "ok",
                    "char_len": meta["char_len"],
                })

            except Exception as e:
                report_rows.append({
                    "file_name": p.name,
                    "status": f"fail:{type(e).__name__}",
                    "error": str(e)[:500],
                    "char_len": 0,
                })

    rep = pd.DataFrame(report_rows)
    rep_csv = out_md_dir / "hwp_clean_stage_report.csv"
    rep.to_csv(rep_csv, index=False, encoding="utf-8-sig")

    if (rep["status"] == "ok").sum() == 0:
        raise RuntimeError(f"1단(정제) 성공 문서가 없습니다. report={rep_csv}")

    return rep_csv


# =========================================================
# 2단: CLEAN_DOCS JSONL -> CHUNK_DOCS JSONL (청킹)
# =========================================================
def _split_paragraphs(text: str) -> List[str]:
    """
    [ADDED] 문단(\n\n) 단위 우선 분해.
    - 빈 문단 제거
    """
    t = clean_markdown(text)
    if not t:
        return []
    paras = [p.strip() for p in t.split("\n\n")]
    return [p for p in paras if p]


def chunk_text_paragraph_first(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    [CHANGED] 기존 '문자 단위 슬라이싱' 대신
    - 문단 단위로 먼저 쪼개고
    - chunk_size를 넘지 않도록 문단을 누적
    - overlap은 '마지막 overlap 문자'를 다음 chunk 앞에 붙이는 방식

    장점: 문장/문단 경계를 덜 깨므로 검색 품질이 보통 개선됨.
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
        # 문단이 chunk_size보다 큰 경우: 안전하게 문자 슬라이스로 쪼갬(예외 처리)
        if len(p) > chunk_size:
            # 기존 buf를 먼저 비움
            flush()
            s = 0
            while s < len(p):
                e = min(len(p), s + chunk_size)
                chunks.append(p[s:e].strip())
                if e == len(p):
                    break
                s = max(0, e - overlap)
            continue

        # buf에 문단을 붙였을 때 초과하면 flush 후 새로 시작
        candidate = (buf + "\n\n" + p).strip() if buf else p
        if len(candidate) <= chunk_size:
            buf = candidate
        else:
            # flush 하고, overlap 적용해서 새 buf 시작
            prev = buf
            flush()
            if overlap > 0 and prev:
                tail = prev[-overlap:]
                buf = (tail + "\n\n" + p).strip()
            else:
                buf = p

    flush()
    return chunks


def build_chunk_docs_jsonl(
    clean_jsonl: Path,
    chunk_jsonl: Path,
    chunk_size: int,
    overlap: int
) -> Path:
    report_rows = []
    with open(chunk_jsonl, "w", encoding="utf-8") as out_f:
        with open(clean_jsonl, "r", encoding="utf-8") as in_f:
            for line in tqdm(in_f, desc=f"(2) CLEAN -> CHUNKS cs={chunk_size}, ov={overlap}"):
                obj = json.loads(line)
                doc_id = obj["doc_id"]
                content = obj["content"]
                md_meta = obj.get("metadata", {})

                # [CHANGED]
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
                            "source_type": "hwp_chunk_md",
                        }
                    }
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                report_rows.append({
                    "doc_id": doc_id,
                    "file_name": md_meta.get("file_name"),
                    "char_len": md_meta.get("char_len"),
                    "chunks": len(chunks),
                })

    rep = pd.DataFrame(report_rows)
    rep_csv = chunk_jsonl.with_suffix(".report.csv")
    rep.to_csv(rep_csv, index=False, encoding="utf-8-sig")
    return rep_csv


# =========================================================
# 3단: CHUNK_DOCS JSONL -> 임베딩 -> FAISS 저장
# =========================================================
def build_faiss_from_chunk_jsonl(
    chunk_jsonl: Path,
    faiss_dir: Path,
    embed_model: str,
    batch_size: int = 100,   # [CHANGED] 기본 300 -> 100
) -> Path:
    emb = OpenAIEmbeddings(model=embed_model)

    def iter_docs():
        with open(chunk_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                yield Document(page_content=obj["content"], metadata=obj.get("metadata", {}))

    db = None
    batch = []
    for d in tqdm(iter_docs(), desc="(3) CHUNKS -> FAISS (embed)"):
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
# CLI / main
# =========================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HWP -> Clean MD -> Chunk -> FAISS builder")

    # [ADDED] 경로를 CLI로 받되, 기본은 '프로젝트 루트 기준'
    p.add_argument("--raw-dir", type=str, default=None, help="HWP가 있는 data/raw 경로")
    p.add_argument("--out-dir", type=str, default=None, help="산출물 out 경로")
    p.add_argument("--tmp-root", type=str, default=None, help="임시 폴더 경로")

    p.add_argument("--chunk-size", type=int, default=900)
    p.add_argument("--overlap", type=int, default=150)
    p.add_argument("--embed-model", type=str, default="text-embedding-3-small")
    p.add_argument("--embed-batch", type=int, default=100)  # [CHANGED] 기본 100

    # 단계 선택
    p.add_argument("--run-clean", action="store_true", help="(1) clean stage 실행")
    p.add_argument("--run-chunk", action="store_true", help="(2) chunk stage 실행")
    p.add_argument("--run-faiss", action="store_true", help="(3) faiss stage 실행")

    # tmp 정리 토글
    g = p.add_mutually_exclusive_group()
    g.add_argument("--cleanup-tmp", dest="cleanup_tmp", action="store_true", help="tmp 폴더 자동 정리(기본)")
    g.add_argument("--no-cleanup-tmp", dest="cleanup_tmp", action="store_false", help="tmp 폴더 유지")
    p.set_defaults(cleanup_tmp=True)

    # 덮어쓰기 정책
    p.add_argument("--force", action="store_true", help="기존 산출물 있어도 덮어쓰기")

    # 아무 단계 플래그가 없으면 전체 실행
    p.add_argument("--all", action="store_true", help="(1)(2)(3) 전체 실행")

    return p.parse_args()


def ensure_paths(args: argparse.Namespace) -> Dict[str, Path]:
    """
    [ADDED] 프로젝트 루트 기준으로 기본 경로를 고정.
    - 프로젝트 루트 = src/의 상위 폴더
    """
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = Path(args.raw_dir) if args.raw_dir else (project_root / "data" / "raw")
    out_dir = Path(args.out_dir) if args.out_dir else (project_root / "out")
    tmp_root = Path(args.tmp_root) if args.tmp_root else (project_root / "tmp_hwp5html")

    out_md_dir = out_dir / "hwp_md"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_md_dir.mkdir(parents=True, exist_ok=True)
    tmp_root.mkdir(parents=True, exist_ok=True)

    return {
        "project_root": project_root,
        "raw_dir": raw_dir,
        "out_dir": out_dir,
        "out_md_dir": out_md_dir,
        "tmp_root": tmp_root,
    }


def refuse_overwrite(path: Path, force: bool) -> None:
    """[ADDED] force 없으면 기존 산출물 있으면 중단."""
    if path.exists() and not force:
        raise RuntimeError(
            f"산출물이 이미 존재합니다: {path}\n"
            f"덮어쓰려면 --force 를 사용하세요."
        )


def main() -> None:
    load_dotenv()  # [ADDED] .env 로딩

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다. .env 또는 환경변수를 확인하세요.")

    require_cmd("hwp5html")  # [ADDED] 외부 커맨드 체크

    args = parse_args()
    paths = ensure_paths(args)

    RAW_DIR = paths["raw_dir"]
    OUT_DIR = paths["out_dir"]
    OUT_MD_DIR = paths["out_md_dir"]
    TMP_ROOT = paths["tmp_root"]

    CHUNK_SIZE = args.chunk_size
    OVERLAP = args.overlap

    CLEAN_DOCS_JSONL = OUT_DIR / "cleaned_hwp_docs.jsonl"
    CHUNK_DOCS_JSONL = OUT_DIR / f"chunks_hwp_cs{CHUNK_SIZE}_ov{OVERLAP}.jsonl"
    FAISS_DIR = OUT_DIR / f"faiss_hwp_cs{CHUNK_SIZE}_ov{OVERLAP}"

    # 실행 단계 결정
    run_clean = args.all or args.run_clean or (not (args.run_clean or args.run_chunk or args.run_faiss))
    run_chunk = args.all or args.run_chunk or (not (args.run_clean or args.run_chunk or args.run_faiss))
    run_faiss = args.all or args.run_faiss or (not (args.run_clean or args.run_chunk or args.run_faiss))

    # 덮어쓰기 정책
    if run_clean:
        refuse_overwrite(CLEAN_DOCS_JSONL, args.force)
    if run_chunk:
        refuse_overwrite(CHUNK_DOCS_JSONL, args.force)
    if run_faiss:
        # FAISS는 디렉토리 산출물이므로 디렉토리 기준
        if FAISS_DIR.exists() and not args.force:
            raise RuntimeError(f"FAISS_DIR이 이미 존재합니다: {FAISS_DIR}\n덮어쓰려면 --force 사용")

    # (1) clean
    clean_report = None
    if run_clean:
        clean_report = build_clean_docs_jsonl(
            raw_dir=RAW_DIR,
            tmp_root=TMP_ROOT,
            out_md_dir=OUT_MD_DIR,
            clean_jsonl=CLEAN_DOCS_JSONL,
            cleanup_tmp=args.cleanup_tmp,  # [ADDED]
        )

    # (2) chunk
    chunk_report = None
    if run_chunk:
        if not CLEAN_DOCS_JSONL.exists():
            raise RuntimeError(f"clean_jsonl이 없습니다: {CLEAN_DOCS_JSONL} (먼저 (1) 실행 필요)")
        chunk_report = build_chunk_docs_jsonl(
            clean_jsonl=CLEAN_DOCS_JSONL,
            chunk_jsonl=CHUNK_DOCS_JSONL,
            chunk_size=CHUNK_SIZE,
            overlap=OVERLAP
        )

    # (3) faiss
    saved_faiss = None
    if run_faiss:
        if not CHUNK_DOCS_JSONL.exists():
            raise RuntimeError(f"chunk_jsonl이 없습니다: {CHUNK_DOCS_JSONL} (먼저 (2) 실행 필요)")
        saved_faiss = build_faiss_from_chunk_jsonl(
            chunk_jsonl=CHUNK_DOCS_JSONL,
            faiss_dir=FAISS_DIR,
            embed_model=args.embed_model,
            batch_size=args.embed_batch,   # [CHANGED]
        )

    # 결과 출력
    if clean_report:
        print("✅ 1단 리포트:", clean_report)
    if chunk_report:
        print("✅ 2단 리포트:", chunk_report)
    if saved_faiss:
        print("✅ 3단 FAISS  :", saved_faiss)
    print("DONE.")


if __name__ == "__main__":
    main()
