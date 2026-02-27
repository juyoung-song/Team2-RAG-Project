# scripts/rag_service_final.py
import os
import json
import re
import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from datetime import datetime

import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI

def normalize_cite_ids(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"\((\d{4})\)", r"(#\1)", text)
    def repl(m):
        s = m.group(1)
        if s.startswith("20"):
            return s
        return f"#{s}"
    text = re.sub(r"(?<!#)\b(\d{4})\b", repl, text)
    return text

# =========================
# Paths / Config
# =========================
BASE = Path("/mnt/c/Users/rnrud/Documents/project/Team2-RAG-Project")
INDEX_DIR = BASE / "data/index"
OUT_DIR = BASE / "data/out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FAISS_PATH = INDEX_DIR / "faiss.index"
RECORDS_PATH = INDEX_DIR / "corpus_records.json"

TOP_K_DOC_QA = 12
TOP_K_GLOBAL_ROUTE = 160
TOP_K_GLOBAL_QA = 25
PROBE_MULTIPLIER = 15
CONTEXT_MAX_CHARS = 9000

AUTO_MIN_MARGIN = 0.08
AUTO_MIN_TOP_SCORE = 0.18
AUTO_MIN_DOC_HITS = 1

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
EMB_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

# =========================
# Load index + records
# =========================
def load_index_and_records():
    if not FAISS_PATH.exists():
        raise FileNotFoundError(f"Missing faiss index: {FAISS_PATH}")
    if not RECORDS_PATH.exists():
        raise FileNotFoundError(f"Missing records json: {RECORDS_PATH}")
    index = faiss.read_index(str(FAISS_PATH))
    records = json.loads(RECORDS_PATH.read_text(encoding="utf-8"))
    return index, records

# =========================
# Embedding helper
# =========================
def embed(texts: List[str]) -> np.ndarray:
    resp = client.embeddings.create(model=EMB_MODEL, input=texts)
    vec = np.array([d.embedding for d in resp.data], dtype=np.float32)
    faiss.normalize_L2(vec)
    return vec

# =========================
# ID normalization & Retrieval
# =========================
def norm_doc_id(x: str) -> str:
    return (x or "").strip()

def record_block_id(rec: Dict[str, Any], fallback_idx: int) -> str:
    cid = rec.get("chunk_id") or rec.get("block_id") or rec.get("chunkId") or rec.get("id")
    if cid:
        return str(cid).strip()
    doc_id = str(rec.get("doc_id", "UNKNOWN")).strip()
    return f"{doc_id}__idx{fallback_idx:06d}"

def _pack_hit(score: float, idx: int, rec: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "score": float(score),
        "doc_id": norm_doc_id(rec.get("doc_id", "")),
        "block_id": record_block_id(rec, idx),
        "text": rec.get("text", "") or "",
        "meta": rec.get("meta", {}) or {}
    }

def retrieve_global(query: str, index, records, k: int = TOP_K_GLOBAL_ROUTE) -> List[Dict[str, Any]]:
    qv = embed([query])
    k2 = min(k, len(records))
    D, I = index.search(qv, k2)
    return [_pack_hit(score, int(idx), records[int(idx)]) for score, idx in zip(D[0], I[0])]

# rag_service_final.py 내 수정
def retrieve_doc(query: str, doc_id: str, index, records, k: int = TOP_K_DOC_QA) -> List[Dict[str, Any]]:
    doc_id = norm_doc_id(doc_id)
    probe_k = min(len(records), 150) # 후보군을 넉넉히 확보
    qv = embed([query])
    D, I = index.search(qv, probe_k)
    
    keywords = [kw for kw in query.split() if len(kw) >= 2]
    hits = []
    for score, idx in zip(D[0], I[0]):
        rec = records[int(idx)]
        if norm_doc_id(rec.get("doc_id", "")) != doc_id:
            continue
            
        # [하이브리드 가중치] 본문에 키워드 포함 시 보너스
        text = rec.get("text", "")
        boost = sum(0.05 for kw in keywords if kw in text)
        
        hits.append(_pack_hit(float(score) + boost, int(idx), rec))

    hits.sort(key=lambda x: x["score"], reverse=True)
    return hits[:k]
# =========================
# Formatting helpers
# =========================
def try_parse_float(x):
    try:
        if x is None: return None
        if isinstance(x, (int, float, np.number)): return float(x)
        s = re.sub(r"[^\d\.\-]", "", str(x).strip())
        return float(s) if s else None
    except Exception:
        return None

def format_krw_compact(value) -> str:
    v = try_parse_float(value)
    if v is None or math.isnan(v): return ""
    v_int = int(round(v))
    if v_int >= 100_000_000: return f"{v_int / 100_000_000:.2f}억 원"
    if v_int >= 10_000_000: return f"{v_int // 10_000:,}만 원"
    return f"{v_int:,}원"

def format_dt_like(x) -> str:
    if not x: return ""
    s = str(x).strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}(\s+\d{2}:\d{2}:\d{2})?$", s): return s
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y.%m.%d", "%Y/%m/%d"):
        try:
            dt = datetime.strptime(s[:19] if len(s) >= 19 else s, fmt)
            return dt.strftime("%Y-%m-%d %H:%M:%S") if "%H" in fmt else dt.strftime("%Y-%m-%d")
        except Exception:
            pass
    return s

def normalize_meta_for_context(meta: dict) -> dict:
    m = dict(meta) if isinstance(meta, dict) else {}
    for k in ["사업 금액", "사업금액", "예산", "사업예산"]:
        if k in m:
            if fmt := format_krw_compact(m.get(k)): m[k] = fmt
            break
    for k in ["공개 일자", "입찰 참여 시작일", "입찰 참여 마감일", "제출 마감일", "마감일", "제안서 제출 마감일"]:
        if k in m: m[k] = format_dt_like(m.get(k))
    return m

def meta_line(meta: Dict[str, Any]) -> str:
    meta = normalize_meta_for_context(meta)
    keys = ["사업명", "발주 기관", "사업 금액", "입찰 참여 마감일", "공고 번호", "공고 차수", "파일명"]
    return " | ".join(f"{k}={v}" for k in keys if (v := str(meta.get(k, "")).strip()))

def build_meta_context_block(doc_id: str, records: list) -> str:
    for r in records:
        if r.get("doc_id") == doc_id:
            m = normalize_meta_for_context(r.get("meta", {}))
            if m: return "[META]\n" + meta_line(m)
            break
    return "[META]\n(메타데이터 없음)"

# [수정 반영] Context Truncation 개선
def format_context(hits: List[Dict[str, Any]], doc_id: Optional[str] = None, records: Optional[list] = None) -> str:
    lines = []
    current_length = 0
    if doc_id and records:
        meta_block = build_meta_context_block(doc_id, records)
        lines.append(meta_block)
        current_length += len(meta_block)

    for h in hits:
        text = (h.get("text") or "").strip()
        if not text: continue
        bid = (h.get("block_id") or "").strip()
        mline = meta_line(h.get("meta", {}))
        
        block_text = f"[{bid}]\n{mline}\n{text}\n\n"
        if current_length + len(block_text) > CONTEXT_MAX_CHARS:
            lines.append("\n[TRUNCATED: 최대 컨텍스트 길이에 도달하여 이후 내용은 생략됩니다.]")
            break
            
        lines.append(block_text.strip())
        current_length += len(block_text)

    return "\n\n".join(lines)

# =========================
# Auto routing (safe)
# =========================
# rag_service_final.py 내 수정
def route_best_doc_from_hits(hits: List[Dict[str, Any]], query: str):
    if not hits:
        return None, {"ok": False, "reasons": ["검색 결과 없음"]}

    doc_scores = {}
    # 질문에서 핵심 키워드 추출 (예: '봉화군', '모잠비크', '나노', '의료')
    keywords = [kw for kw in re.sub(r'[^\w\s]', '', query).split() if len(kw) >= 2]

    for h in hits:
        d = h["doc_id"]
        biz_name = str(h.get("meta", {}).get("사업명", ""))
        
        # [하이브리드 가중치] 질문의 키워드가 사업명에 있으면 +0.5점 보너스
        boost = 0.0
        for kw in keywords:
            if kw in biz_name:
                boost += 0.5
        
        doc_scores[d] = doc_scores.get(d, 0.0) + h["score"] + boost

    # 점수 순으로 정렬
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    best_doc, top_score = sorted_docs[0]
    
    # 안전장치 작동
    reasons = []
    if top_score < AUTO_MIN_TOP_SCORE:
        reasons.append(f"최고 점수({top_score:.2f})가 기준 미달")
    
    if reasons:
        return None, {"ok": False, "reasons": reasons}
        
    return best_doc, {"ok": True, "top_score": top_score}
# =========================
# LLM calls (API 문법 수정 반영)
# =========================
def llm_document_brief(doc_id: str, index, records) -> Dict[str, Any]:
    hits = retrieve_doc("이 문서의 예산/일정/제출/요구사항/리스크를 컨설턴트 관점에서 핵심만 뽑아", doc_id, index, records, k=14)
    ctx = format_context(hits, doc_id=doc_id, records=records)
    sys = (
        "너는 '입찰메이트' 공공입찰 컨설턴트 지원 AI다. 반드시 주어진 CONTEXT만 근거로 요약하고, "
        "없는 정보는 추측하지 말고 '문서 근거를 찾지 못했다'라고 적어라. 아래 섹션 제목을 그대로 사용하라.\n\n"
        "기관\n사업명\n예산\n일정/마감\n제출 방식\n주요 요구사항\n리스크/주의사항\n\n"
        "각 문장 끝에 근거 block_id를 괄호로 붙여라. (META도 근거로 허용)"
    )
    
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": f"doc_id={doc_id}\n\nCONTEXT:\n{ctx}"},
        ],
        # temperature=0.0
    )
    return {"doc_id": doc_id, "brief": resp.choices[0].message.content, "used_blocks": [h["block_id"] for h in hits]}

def llm_document_qa(doc_id: str, question: str, index, records, k: int = TOP_K_DOC_QA) -> Dict[str, Any]:
    hits = retrieve_doc(question, doc_id, index, records, k=k)
    ctx = format_context(hits, doc_id=doc_id, records=records)
    sys = (
        "너는 '입찰메이트' RFP 문서 QA 어시스턴트다. 반드시 제공된 CONTEXT에서만 근거를 사용한다. "
        "근거가 없으면 반드시 '문서 근거를 찾지 못했다'라고 답하고 종료한다. 절대 추측하거나 다른 문서 내용을 섞지 마라. "
        "답변은 최대한 간결하게 정답 위주로 기술하고, 출처는 [문서ID] 형식으로 끝에 한 번만 표기하라"
    )
    
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": f"질문: {question}\n\nCONTEXT:\n{ctx}"},
        ],
        # temperature=0.0
    )
    return {"doc_id": doc_id, "question": question, "answer": resp.choices[0].message.content, "hits": hits}

def llm_global_qa(question: str, index, records, k: int = TOP_K_GLOBAL_QA) -> Dict[str, Any]:
    hits = retrieve_global(question, index, records, k=max(k, 50))
    ctx = format_context(hits[:k])
    sys = (
        "너는 여러 RFP 코퍼스를 검색하는 RAG 어시스턴트다. 반드시 제공된 CONTEXT에서만 근거를 사용한다. "
        "근거가 없으면 '문서 근거를 찾지 못했다'라고 답한다. 근거는 반드시 (block_id)로 표기한다."
    )
    
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": f"질문: {question}\n\nCONTEXT:\n{ctx}"},
        ],
        # temperature=0.0
    )
    return {"question": question, "answer": resp.choices[0].message.content, "hits": hits[:k]}

# =========================
# Auto routing (safe)
# =========================
# (rag_service_final.py 파일 내 auto_answer 함수 전체를 아래 코드로 교체해 주세요)

def auto_answer(question: str, index, records, debug: bool = False, k_doc: int = 12, k_global: int = 160, **kwargs) -> Dict[str, Any]:
    # 1. 전역 검색
    wide_hits = retrieve_global(question, index, records, k=k_global)
    
    # 2. 문서 라우팅 (question을 두 번째 인자로 반드시 전달!)
    best_doc, route_info = route_best_doc_from_hits(wide_hits, question)
    
    if not best_doc:
        return {
            "mode": "auto", 
            "doc_id": None, 
            "question": question, 
            "answer": "문서 근거를 찾지 못했다.", 
            "route_info": route_info if debug else None
        }
    
    # 3. 상세 QA 진행
    qa = llm_document_qa(best_doc, question, index, records, k=k_doc)
    return {
        "mode": "auto", 
        "doc_id": best_doc, 
        "question": question, 
        "answer": qa["answer"], 
        "route_info": route_info if debug else None, 
        "hits": qa["hits"] if debug else None
    }
# Utilities / Demo main
# =========================
def save_json(path: Path, obj: Any):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def recommend_rfps(profile_text: str, index, records, top_n: int = 8, chunk_k: int = 120):
    hits = retrieve_global(profile_text, index, records, k=chunk_k)
    by_doc: Dict[str, List[Dict[str, Any]]] = {}
    for h in hits:
        if d := norm_doc_id(h["doc_id"]): by_doc.setdefault(d, []).append(h)

    ranked = []
    for doc_id, hs in by_doc.items():
        hs_sorted = sorted(hs, key=lambda x: x["score"], reverse=True)
        topm = hs_sorted[:5]
        score = 0.6 * topm[0]["score"] + 0.4 * (sum(x["score"] for x in topm) / len(topm))
        ranked.append({
            "doc_id": doc_id, "score": float(score),
            "evidence_blocks": [{"block_id": x["block_id"], "score": float(x["score"]), "text_preview": x["text"][:220].replace("\n", " ")} for x in topm[:3]],
            "meta": (topm[0].get("meta", {}) or {})
        })
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked[:top_n]

if __name__ == "__main__":
    index, records = load_index_and_records()

    profile = (
        "우리는 공공기관 정보시스템 구축 경험이 있고, 웹/백엔드 기반 시스템 고도화, 운영/유지보수, 데이터 분석/대시보드 구축을 수행할 수 있다. "
        "나라장터 전자제출 경험이 있으며, 예산 1억~10억 규모를 선호한다. 가능하면 서울/수도권 수행을 선호한다."
    )

    recs = recommend_rfps(profile, index, records, top_n=8, chunk_k=120)
    save_json(OUT_DIR / "recommendations.json", {"profile": profile, "recommendations": recs})

    target_doc = recs[0]["doc_id"] if recs else norm_doc_id(records[0].get("doc_id", ""))
    brief = llm_document_brief(target_doc, index, records)
    save_json(OUT_DIR / "doc_brief.json", brief)

    doc_qa = llm_document_qa(target_doc, "예산, 마감일, 제출 방식, 주요 요구사항을 알려줘", index, records, k=TOP_K_DOC_QA)
    save_json(OUT_DIR / "doc_qa.json", doc_qa)

    global_qa = llm_global_qa("나라장터 전자제출 관련해서 자주 나오는 제출 유의사항을 요약해줘", index, records, k=10)
    save_json(OUT_DIR / "global_qa.json", global_qa)

    demo_run = {
        "target_doc_id": target_doc, "top_recommendation": recs[0] if recs else None,
        "brief": brief, "doc_qa": doc_qa, "global_qa": global_qa,
    }
    save_json(OUT_DIR / "demo_run.json", demo_run)
    print("Saved:", OUT_DIR / "demo_run.json")
