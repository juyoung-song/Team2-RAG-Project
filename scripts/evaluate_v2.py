#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluate_v2.py

Project-specific evaluator adapter for Team2-RAG-Project.

This script evaluates your RAG system outputs against a gold set in test_questions.json
using lightweight heuristic matching, with an optional embedding-based semantic judge.

Expected input file format (JSON list):
[
  {"question": "...", "answer": "..."},
  {"doc_id": "문서ID", "question": "...", "answer": "..."},
  {"mode": "global"|"doc", "doc_id": "...", "question": "...", "answer": "..."}
]

Default behavior:
- If doc_id is provided (or mode == "doc"): uses llm_document_qa(doc_id, question, ...)
- Otherwise: uses llm_global_qa(question, ...)
"""

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import os, sys

NO_EVIDENCE_PATTERNS = [
    "문서 근거를 찾지 못했다",
    "근거를 찾을 수 없다",
    "정보가 없다",
    "알 수 없다",
    "언급되어 있지 않다"
]
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
    

import numpy as np


def load_rag_module():
    """Import scripts/rag_service_final.py as a module."""
    try:
        import scripts.rag_service_final as rsf  # type: ignore
        return rsf
    except Exception as e:
        raise RuntimeError(
            "Could not import scripts.rag_service_final. Run from repo root:\n"
            "  cd /mnt/c/Users/rnrud/Documents/project/Team2-RAG-Project\n"
            "  python scripts/evaluate_v2.py\n"
            f"Original import error: {e}"
        ) from e


def predict_answer(rsf, index, records, question: str, doc_id=None, mode="auto", k_doc=8, k_global=10) -> str:
    mode = (mode or "auto").strip().lower()

    # auto: 라우팅 포함 (문서 추천/선택 후 문서 기반 QA)
    if mode == "auto":
        if hasattr(rsf, "auto_answer"):
            out = rsf.auto_answer(question, index, records, k_doc=k_doc, k_global=k_global)

            # auto_answer 반환 형태 유연하게 처리
            if isinstance(out, dict):
                if "answer" in out:
                    return str(out["answer"])
                if "pred" in out:
                    return str(out["pred"])
                if "text" in out:
                    return str(out["text"])
                return json.dumps(out, ensure_ascii=False)

            return str(out)

        # auto_answer가 없다면 최소한 global로 fallback (하지만 이건 성능이 낮을 수 있음)
        return rsf.llm_global_qa(question, index, records, k=k_global)

    # doc: doc_id가 있어야 문서 한 건으로 QA
    if mode == "doc":
        if not doc_id:
            return "문서 근거를 찾지 못했다. (doc 모드인데 doc_id가 제공되지 않음)"
        if hasattr(rsf, "llm_document_qa"):
            return rsf.llm_document_qa(doc_id, question, index, records, k=k_doc)
        # 문서 QA 함수명이 다르면 여기 맞춰서 변경
        return rsf.llm_doc_qa(doc_id, question, index, records, k=k_doc)

    # global: 전체 코퍼스 대상으로 QA
    if mode == "global":
        return rsf.llm_global_qa(question, index, records, k=k_global)

    return f"문서 근거를 찾지 못했다. (unknown mode={mode})"


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).lower()
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def has_no_evidence(pred: str) -> bool:
    p = normalize_text(pred)
    return any(re.search(pat, p) for pat in NO_EVIDENCE_PATTERNS)


def extract_numbers(s: str) -> List[str]:
    if not s:
        return []
    nums = re.findall(r"\d[\d,\.]*", s)
    cleaned: List[str] = []
    for x in nums:
        x2 = x.replace(",", "")
        if x2:
            cleaned.append(x2)
    return cleaned


_KO_WORD = re.compile(r"[가-힣A-Za-z0-9]+")


def extract_keywords(s: str) -> List[str]:
    if not s:
        return []
    toks = _KO_WORD.findall(s)
    toks = [t.lower() for t in toks if len(t) >= 2]
    stop = {"합니다", "하여", "하는", "있음", "없음", "및", "등", "제출", "사업", "관련", "필요"}
    return [t for t in toks if t not in stop]


def f1_overlap(a: List[str], b: List[str]) -> float:
    if not a or not b:
        return 0.0
    a_set, b_set = set(a), set(b)
    tp = len(a_set & b_set)
    if tp == 0:
        return 0.0
    prec = tp / max(1, len(b_set))
    rec = tp / max(1, len(a_set))
    return 2 * prec * rec / max(1e-9, (prec + rec))


def numbers_coverage(gold: str, pred: str) -> float:
    g = set(extract_numbers(gold))
    p = set(extract_numbers(pred))
    if not g:
        return 0.0
    return len(g & p) / len(g)


def substring_match(gold: str, pred: str) -> bool:
    g = normalize_text(gold)
    p = normalize_text(pred)
    if not g or not p:
        return False
    return (g in p) or (p in g)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class EmbedJudge:
    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model
        self.cache: Dict[str, np.ndarray] = {}
        from openai import OpenAI  # type: ignore
        self.client = OpenAI()

    def embed(self, text: str) -> np.ndarray:
        key = normalize_text(text)
        if key in self.cache:
            return self.cache[key]
        resp = self.client.embeddings.create(model=self.model, input=text)
        vec = np.array(resp.data[0].embedding, dtype=np.float32)
        self.cache[key] = vec
        return vec

    def score(self, gold: str, pred: str) -> float:
        return cosine(self.embed(gold), self.embed(pred))


def judge_one(
    gold: str,
    pred: str,
    use_embed: bool,
    embed_judge: Optional[EmbedJudge],
    thresholds: Dict[str, float],
) -> Tuple[bool, Dict[str, Any]]:
    info: Dict[str, Any] = {}

    if has_no_evidence(pred):
        info["rule"] = "no_evidence"
        return False, info

    if substring_match(gold, pred):
        info["rule"] = "substring"
        return True, info

    num_cov = numbers_coverage(gold, pred)
    info["num_coverage"] = num_cov
    if num_cov >= thresholds["num_coverage"]:
        info["rule"] = "numbers"
        return True, info

    kw_f1 = f1_overlap(extract_keywords(gold), extract_keywords(pred))
    info["kw_f1"] = kw_f1
    if kw_f1 >= thresholds["kw_f1"]:
        info["rule"] = "kw_f1"
        return True, info

    if use_embed:
        if embed_judge is None:
            raise RuntimeError("Embedding judge requested but not initialized")
        emb = embed_judge.score(gold, pred)
        info["embed_cosine"] = emb
        if emb >= thresholds["embed_cosine"]:
            info["rule"] = "embed"
            return True, info

    info["rule"] = "fail"
    return False, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="test_questions.json", help="Path to test_questions.json")
    ap.add_argument("--out", default="data/out/eval_report.json", help="Where to save evaluation report JSON")
    ap.add_argument("--mode", default="auto", choices=["auto", "global", "doc"], help="Force evaluation mode")
    ap.add_argument("--k_doc", type=int, default=8)
    ap.add_argument("--k_global", type=int, default=10)
    ap.add_argument("--embed_judge", action="store_true", help="Enable embedding semantic judge (costs API)")
    ap.add_argument("--embed_model", default="text-embedding-3-small")
    ap.add_argument("--embed_threshold", type=float, default=0.75)
    ap.add_argument("--kw_f1_threshold", type=float, default=0.65)
    ap.add_argument("--num_cov_threshold", type=float, default=0.60)
    ap.add_argument("--limit", type=int, default=0, help="Evaluate only first N examples (0 = all)")
    ap.add_argument("--debug_n", type=int, default=5, help="Print first N detailed cases")
    args = ap.parse_args()

    questions_path = Path(args.questions)
    if not questions_path.exists():
        raise FileNotFoundError(
            f"Missing {questions_path}. Create it next to this script or pass --questions. "
            "Expected format: [{'question':..., 'answer':...}, ...]"
        )

    rsf = load_rag_module()
    index, records = rsf.load_index_and_records()
    
    items = json.loads(questions_path.read_text(encoding="utf-8"))
    if not isinstance(items, list) or not items:
        raise ValueError("test_questions.json must be a non-empty JSON list")

    if args.limit and args.limit > 0:
        items = items[: args.limit]

    thresholds = {
        "num_coverage": float(args.num_cov_threshold),
        "kw_f1": float(args.kw_f1_threshold),
        "embed_cosine": float(args.embed_threshold),
    }

    embed = EmbedJudge(model=args.embed_model) if args.embed_judge else None

    results = []
    ok = 0
    t0 = time.time()

    for i, ex in enumerate(items, 1):
        q = (ex.get("question") or "").strip()
        g = (ex.get("answer") or "").strip()
        doc_id = ex.get("doc_id")
        ex_mode = ex.get("mode") or args.mode
        
        # ⭐ 터미널 멈춤 방지용: 진행 상황 출력 코드 추가!
        print(f"\n⏳ [{i}/{len(items)}] 평가 진행 중... (질문: {q})")

        if not q or not g:
            results.append({
                "i": i,
                "question": q,
                "gold": g,
                "pred": "",
                "correct": False,
                "error": "missing question/answer in test_questions.json item",
            })
            continue

        try:
            pred = predict_answer(
                rsf, index, records, q,
                doc_id=doc_id,
                mode=ex_mode,
                k_doc=args.k_doc,
                k_global=args.k_global,
            )
            correct, info = judge_one(g, pred, args.embed_judge, embed, thresholds)
            ok += int(correct)
            results.append({
                "i": i,
                "mode": ex_mode,
                "doc_id": doc_id,
                "question": q,
                "gold": g,
                "pred": pred,
                "correct": bool(correct),
                "judge_info": info,
            })

            if i <= args.debug_n:
                print(f"\n[{i}] correct={correct} rule={info.get('rule')}")
                if doc_id:
                    print("doc_id:", doc_id)
                print("Q:", q)
                print("G:", g[:300])
                print("P:", pred[:300])

        except Exception as e:
            results.append({
                "i": i,
                "mode": ex_mode,
                "doc_id": doc_id,
                "question": q,
                "gold": g,
                "pred": "",
                "correct": False,
                "error": str(e),
            })

    dt = time.time() - t0
    report = {
        "questions_path": str(questions_path),
        "n": len(results),
        "ok": ok,
        "accuracy": ok / max(1, len(results)),
        "elapsed_sec": dt,
        "thresholds": thresholds,
        "embed_judge": bool(args.embed_judge),
        "embed_model": args.embed_model if args.embed_judge else None,
        "k_doc": args.k_doc,
        "k_global": args.k_global,
        "results": results,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n--- Summary ---")
    print("n:", len(results))
    print("ok:", ok)
    print("accuracy:", round(report["accuracy"], 4))
    print("saved:", out_path)


if __name__ == "__main__":
    main()
