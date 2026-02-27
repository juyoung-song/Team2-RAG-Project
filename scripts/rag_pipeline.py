import os
import re
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

import faiss
from openai import OpenAI

# ============================================================
# Project Paths
# ============================================================
BASE = Path("/mnt/c/Users/rnrud/Documents/project/Team2-RAG-Project")

DATA_DIR = BASE / "data"
REPORT_DIR = DATA_DIR / "reports"
CORPUS_DIR = DATA_DIR / "corpus"
INDEX_DIR = DATA_DIR / "index"
OUT_DIR = DATA_DIR / "out"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
CORPUS_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Input text priority: HTML-derived clean text > clean_text > extracted_raw
TXT_DIR_HTML = DATA_DIR / "clean_text_html"
TXT_DIR_CLEAN = DATA_DIR / "clean_text"
TXT_DIR_RAW = DATA_DIR / "extracted_raw"

# Metadata priority: safe manifest > data_list
SAFE_MANIFEST = DATA_DIR / "metadata" / "metadata_safe_for_html.csv"
DATA_LIST = Path("/mnt/data/data_list.csv")  # user uploaded path; if missing, fall back below
DATA_LIST_LOCAL = BASE / "data_list.csv"     # optional local copy

# Outputs
CORPUS_JSONL = CORPUS_DIR / "corpus_blocks.jsonl"
RECORDS_JSON = INDEX_DIR / "corpus_records.json"
FAISS_INDEX = INDEX_DIR / "faiss.index"
RUN_REPORT = REPORT_DIR / "rag_pipeline_build_report.json"

# ============================================================
# Chunking / Retrieval params
# ============================================================
MAX_CHARS = 1600
OVERLAP_CHARS = 200
MIN_DOC_CHARS = 400

EMBED_BATCH = 64

TOP_K_GLOBAL_QA = 10
TOP_K_DOC_QA = 10

CONTEXT_MAX_CHARS = 9000

# ============================================================
# OpenAI
# ============================================================
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
EMB_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")


# ============================================================
# Helpers: normalize / IO
# ============================================================
def norm(s: Any) -> str:
    return ("" if s is None else str(s)).strip()


def sha1_short(s: str, n: int = 10) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()[:n]


def pick_text_dir() -> Path:
    if TXT_DIR_HTML.exists() and any(TXT_DIR_HTML.glob("*.txt")):
        return TXT_DIR_HTML
    if TXT_DIR_CLEAN.exists() and any(TXT_DIR_CLEAN.glob("*.txt")):
        return TXT_DIR_CLEAN
    return TXT_DIR_RAW


def read_txt(doc_id: str, txt_dir: Path) -> str:
    p = txt_dir / f"{doc_id}.txt"
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8", errors="ignore")


def basic_clean(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ============================================================
# Metadata load + doc_id build
# ============================================================
def load_metadata() -> pd.DataFrame:
    if SAFE_MANIFEST.exists():
        df = pd.read_csv(SAFE_MANIFEST)
        df = df.copy()
        df["__meta_source__"] = "metadata_safe_for_html.csv"
    else:
        if DATA_LIST.exists():
            df = pd.read_csv(DATA_LIST)
            df = df.copy()
            df["__meta_source__"] = "/mnt/data/data_list.csv"
        elif DATA_LIST_LOCAL.exists():
            df = pd.read_csv(DATA_LIST_LOCAL)
            df = df.copy()
            df["__meta_source__"] = "data_list.csv(local)"
        else:
            raise FileNotFoundError("No metadata found: metadata_safe_for_html.csv or data_list.csv")

    if "doc_id" not in df.columns:
        if "raw_path_final" in df.columns:
            df["doc_id"] = df["raw_path_final"].astype(str).map(lambda p: Path(p).stem.strip())
        elif "파일명" in df.columns:
            df["doc_id"] = df["파일명"].astype(str).map(lambda p: Path(p).stem.strip())
        elif "file" in df.columns:
            df["doc_id"] = df["file"].astype(str).map(lambda p: Path(p).stem.strip())
        else:
            raise ValueError(f"Cannot derive doc_id. columns={list(df.columns)}")

    df["doc_id"] = df["doc_id"].astype(str).map(lambda x: x.strip())
    df = df.dropna(subset=["doc_id"])
    df = df[df["doc_id"].astype(str).str.len() > 0]
    df = df.drop_duplicates(subset=["doc_id"]).reset_index(drop=True)

    return df


def build_meta_map(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    keep_cols = [c for c in df.columns if c not in {"__meta_source__"}]
    meta_map: Dict[str, Dict[str, Any]] = {}
    for _, r in df.iterrows():
        doc_id = norm(r.get("doc_id"))
        if not doc_id:
            continue
        meta = {}
        for c in keep_cols:
            if c == "doc_id":
                continue
            v = r.get(c)
            if pd.isna(v):
                continue
            meta[c] = v
        meta_map[doc_id] = meta
    return meta_map


# ============================================================
# Chunking
# ============================================================
def split_into_chunks(text: str, max_chars: int, overlap: int) -> List[str]:
    text = text.strip()
    if not text:
        return []

    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []
    buf = ""

    for p in paras:
        if len(buf) + len(p) + 2 <= max_chars:
            buf = (buf + "\n\n" + p).strip() if buf else p
        else:
            if buf:
                chunks.append(buf)
            while len(p) > max_chars:
                chunks.append(p[:max_chars])
                p = p[max_chars - overlap:] if overlap > 0 else p[max_chars:]
            buf = p

    if buf:
        chunks.append(buf)

    if overlap > 0 and len(chunks) > 1:
        out = []
        for i, c in enumerate(chunks):
            if i == 0:
                out.append(c)
            else:
                prev = out[-1]
                tail = prev[-overlap:] if len(prev) > overlap else prev
                out.append((tail + "\n" + c).strip())
        chunks = out

    return chunks


def make_block_id(doc_id: str, idx: int) -> str:
    return f"{doc_id} #{idx:04d}"


# ============================================================
# Corpus build (JSONL)
# ============================================================
def build_corpus(meta_df: pd.DataFrame, txt_dir: Path) -> Dict[str, Any]:
    meta_map = build_meta_map(meta_df)

    ok_docs = 0
    missing_txt = 0
    too_short = 0
    total_blocks = 0

    with CORPUS_JSONL.open("w", encoding="utf-8") as f:
        for doc_id in tqdm(meta_df["doc_id"].tolist(), desc="Building corpus"):
            raw = read_txt(doc_id, txt_dir)
            if not raw:
                missing_txt += 1
                continue

            text = basic_clean(raw)
            if len(text) < MIN_DOC_CHARS:
                too_short += 1
                continue

            chunks = split_into_chunks(text, max_chars=MAX_CHARS, overlap=OVERLAP_CHARS)
            if not chunks:
                too_short += 1
                continue

            meta = meta_map.get(doc_id, {})
            for i, ch in enumerate(chunks):
                rec = {
                    "doc_id": doc_id,
                    "block_id": make_block_id(doc_id, i),
                    "type": "block",
                    "text": ch,
                    "meta": meta,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total_blocks += 1

            ok_docs += 1

    report = {
        "txt_dir": str(txt_dir),
        "meta_source": meta_df.get("__meta_source__", pd.Series(["unknown"])).iloc[0] if len(meta_df) else "unknown",
        "docs_in_meta": int(len(meta_df)),
        "docs_ok_written": int(ok_docs),
        "missing_txt": int(missing_txt),
        "too_short": int(too_short),
        "total_blocks": int(total_blocks),
        "corpus_jsonl": str(CORPUS_JSONL),
    }
    return report


# ============================================================
# Embeddings + FAISS
# ============================================================
def embed_texts(texts: List[str]) -> np.ndarray:
    resp = client.embeddings.create(model=EMB_MODEL, input=texts)
    vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
    return vecs


def build_index_from_corpus() -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    texts: List[str] = []

    with CORPUS_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            records.append(rec)
            texts.append(rec["text"])

    if not records:
        raise ValueError("Corpus is empty. Check corpus_blocks.jsonl generation first.")

    all_vecs: List[np.ndarray] = []
    for i in tqdm(range(0, len(texts), EMBED_BATCH), desc="Embedding"):
        batch = texts[i:i + EMBED_BATCH]
        all_vecs.append(embed_texts(batch))
    X = np.vstack(all_vecs)

    faiss.normalize_L2(X)
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)

    faiss.write_index(index, str(FAISS_INDEX))
    RECORDS_JSON.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")

    report = {
        "records": int(len(records)),
        "dim": int(X.shape[1]),
        "faiss_index": str(FAISS_INDEX),
        "records_json": str(RECORDS_JSON),
    }
    return report


def load_index_and_records() -> Tuple[faiss.IndexFlatIP, List[Dict[str, Any]]]:
    if not FAISS_INDEX.exists() or not RECORDS_JSON.exists():
        raise FileNotFoundError("Index files missing. Run build step first.")
    index = faiss.read_index(str(FAISS_INDEX))
    records = json.loads(RECORDS_JSON.read_text(encoding="utf-8"))
    return index, records


# ============================================================
# Retrieval
# ============================================================
def record_block_id(rec: Dict[str, Any], fallback_idx: int) -> str:
    bid = rec.get("block_id") or rec.get("chunk_id") or rec.get("id")
    if bid:
        return norm(bid)
    doc_id = norm(rec.get("doc_id", "UNKNOWN"))
    return f"{doc_id}__idx{fallback_idx:06d}"


def retrieve_global(query: str, index, records, top_k: int) -> List[Dict[str, Any]]:
    qv = embed_texts([query])
    faiss.normalize_L2(qv)
    k = min(top_k, len(records))
    D, I = index.search(qv, k)

    hits = []
    for score, idx in zip(D[0], I[0]):
        rec = records[int(idx)]
        hits.append({
            "score": float(score),
            "doc_id": norm(rec.get("doc_id", "")),
            "block_id": record_block_id(rec, int(idx)),
            "text": rec.get("text", ""),
            "meta": rec.get("meta", {}) or {},
        })
    return hits


def retrieve_doc(query: str, doc_id: str, index, records, top_k: int) -> List[Dict[str, Any]]:
    doc_id = norm(doc_id)
    probe_k = min(len(records), max(top_k * 12, 50))

    qv = embed_texts([query])
    faiss.normalize_L2(qv)
    D, I = index.search(qv, probe_k)

    hits = []
    for score, idx in zip(D[0], I[0]):
        rec = records[int(idx)]
        if norm(rec.get("doc_id", "")) != doc_id:
            continue
        hits.append({
            "score": float(score),
            "doc_id": doc_id,
            "block_id": record_block_id(rec, int(idx)),
            "text": rec.get("text", ""),
            "meta": rec.get("meta", {}) or {},
        })
        if len(hits) >= top_k:
            break
    return hits


def meta_line(meta: Dict[str, Any]) -> str:
    keys = ["사업명", "발주 기관", "사업 금액", "입찰 참여 마감일", "공고 번호", "공고 차수", "파일명", "file"]
    out = []
    for k in keys:
        if k in meta:
            v = norm(meta.get(k))
            if v:
                out.append(f"{k}={v}")
    return " | ".join(out)


def format_context(hits: List[Dict[str, Any]], max_chars: int) -> str:
    parts = []
    used = 0
    for h in hits:
        m = meta_line(h.get("meta", {}))
        block = f"[{h['block_id']}] score={h['score']:.3f}\nMETA: {m}\n{h['text']}\n"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n---\n".join(parts)


# ============================================================
# RAG QA / Brief
# ============================================================
def rag_global_qa(question: str, index, records, top_k: int = TOP_K_GLOBAL_QA) -> Dict[str, Any]:
    hits = retrieve_global(question, index, records, top_k=max(top_k, 30))
    ctx = format_context(hits[:top_k], max_chars=CONTEXT_MAX_CHARS)

    sys = (
        "너는 '입찰메이트' RAG 어시스턴트다. "
        "반드시 CONTEXT에 근거해서만 답하고, 없으면 '문서 근거를 찾지 못했다'고 답하라. "
        "근거는 block_id를 괄호로 표시하라."
    )

    resp = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": sys},
            {"role": "user", "content": f"질문: {question}\n\nCONTEXT:\n{ctx}"},
        ],
    )

    return {
        "question": question,
        "answer": resp.output_text,
        "hits": hits[:top_k],
    }


def rag_doc_brief(doc_id: str, index, records) -> Dict[str, Any]:
    doc_id = norm(doc_id)
    hits = retrieve_doc("이 문서의 예산/일정/제출/요구사항 중심으로 핵심 파악", doc_id, index, records, top_k=14)
    ctx = format_context(hits, max_chars=CONTEXT_MAX_CHARS)

    sys = (
        "너는 공공입찰 컨설턴트 지원 AI다. "
        "반드시 CONTEXT만 근거로 요약하고, 없는 정보는 추측하지 말고 생략하라. "
        "다음 섹션 제목을 그대로 사용하라.\n\n"
        "기관\n"
        "사업명\n"
        "예산\n"
        "일정/마감\n"
        "제출 방식\n"
        "주요 요구사항\n"
        "리스크/주의사항\n\n"
        "각 항목에 근거 block_id를 괄호로 붙여라."
    )

    resp = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": sys},
            {"role": "user", "content": f"doc_id={doc_id}\n\nCONTEXT:\n{ctx}"},
        ],
    )

    return {
        "doc_id": doc_id,
        "brief": resp.output_text,
        "used_blocks": [h["block_id"] for h in hits],
    }


def rag_doc_qa(doc_id: str, question: str, index, records, top_k: int = TOP_K_DOC_QA) -> Dict[str, Any]:
    doc_id = norm(doc_id)
    hits = retrieve_doc(question, doc_id, index, records, top_k=top_k)
    ctx = format_context(hits, max_chars=CONTEXT_MAX_CHARS)

    sys = (
        "너는 '입찰메이트' RFP 문서 QA 어시스턴트다. "
        "반드시 제공된 CONTEXT에 근거해서만 답하고, 없으면 '문서 근거를 찾지 못했다'고 답하라. "
        "답변은 한국어로, 컨설턴트가 바로 쓰기 좋게 기관/예산/일정/제출/요구사항을 분리해라. "
        "근거는 block_id를 괄호로 표시하라."
    )

    resp = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": sys},
            {"role": "user", "content": f"질문: {question}\n\nCONTEXT:\n{ctx}"},
        ],
    )

    return {
        "doc_id": doc_id,
        "question": question,
        "answer": resp.output_text,
        "hits": hits,
    }


# ============================================================
# Main
# ============================================================
def main():
    meta_df = load_metadata()
    txt_dir = pick_text_dir()

    corpus_report = build_corpus(meta_df, txt_dir)
    index_report = build_index_from_corpus()

    index, records = load_index_and_records()

    doc_id_test = norm(records[0].get("doc_id", "")) if records else ""
    if not doc_id_test:
        raise ValueError("No doc_id found in records for smoke test.")

    brief = rag_doc_brief(doc_id_test, index, records)
    doc_qa = rag_doc_qa(doc_id_test, "예산, 마감일, 제출 방식, 주요 요구사항을 알려줘", index, records, top_k=10)
    global_qa = rag_global_qa("나라장터 전자제출 관련 유의사항을 요약해줘", index, records, top_k=10)

    OUT_DIR / "smoke"  # no-op for readability
    (OUT_DIR / "smoke_doc_brief.json").write_text(json.dumps(brief, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "smoke_doc_qa.json").write_text(json.dumps(doc_qa, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "smoke_global_qa.json").write_text(json.dumps(global_qa, ensure_ascii=False, indent=2), encoding="utf-8")

    full_report = {
        "corpus": corpus_report,
        "index": index_report,
        "smoke_doc_id": doc_id_test,
        "outputs": {
            "corpus_jsonl": str(CORPUS_JSONL),
            "faiss_index": str(FAISS_INDEX),
            "records_json": str(RECORDS_JSON),
            "smoke_doc_brief": str(OUT_DIR / "smoke_doc_brief.json"),
            "smoke_doc_qa": str(OUT_DIR / "smoke_doc_qa.json"),
            "smoke_global_qa": str(OUT_DIR / "smoke_global_qa.json"),
        }
    }
    RUN_REPORT.write_text(json.dumps(full_report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Build done.")
    print("Text dir:", txt_dir)
    print("Corpus:", CORPUS_JSONL)
    print("Index:", FAISS_INDEX)
    print("Records:", RECORDS_JSON)
    print("Smoke outputs:")
    print(" -", OUT_DIR / "smoke_doc_brief.json")
    print(" -", OUT_DIR / "smoke_doc_qa.json")
    print(" -", OUT_DIR / "smoke_global_qa.json")
    print("Report:", RUN_REPORT)


if __name__ == "__main__":
    main()
