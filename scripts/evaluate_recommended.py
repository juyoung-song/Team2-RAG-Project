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
import sys
import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np



def load_rag_module():
    """
    Load scripts/rag_service_final.py even if ./scripts is not a Python package.

    This avoids the common "No module named 'scripts'" error when scripts/ lacks __init__.py.
    """
    root = Path(__file__).resolve().parents[1]  # repo root (../)
    # Ensure repo root is on sys.path so relative imports inside rag_service_final (if any) can work
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    mod_path = root / "scripts" / "rag_service_final.py"
    if not mod_path.exists():
        raise FileNotFoundError(f"Missing rag module at: {mod_path}")

    # Prefer normal import if scripts is a package; otherwise load by file location.
    try:
        import importlib
        return importlib.import_module("scripts.rag_service_final")  # type: ignore
    except Exception:
        import importlib.util
        spec = importlib.util.spec_from_file_location("rag_service_final", mod_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load module spec for: {mod_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore
        return mod

    except Exception as e:
        raise RuntimeError(
            "Could not import scripts.rag_service_final. Run from repo root:\n"
            "  cd /mnt/c/Users/rnrud/Documents/project/Team2-RAG-Project\n"
            "  python scripts/evaluate_v2.py\n"
            f"Original import error: {e}"
        ) from e


def predict_answer(
    rsf,
    index,
    records,
    question: str,
    doc_id: Optional[str] = None,
    mode: str = "auto",
    k_doc: int = 8,
    k_global: int = 10,
) -> str:
    mode = (mode or "auto").strip().lower()
    use_doc = False
    if mode == "doc":
        use_doc = True
    elif mode == "global":
        use_doc = False
    elif mode == "auto":
        use_doc = bool(doc_id)
    else:
        use_doc = bool(doc_id)

    if use_doc and not doc_id:
        raise ValueError("mode='doc' requires doc_id")

    if use_doc:
        out = rsf.llm_document_qa(doc_id, question, index, records, k=k_doc)
        return (out.get("answer", "") or "").strip()

    out = rsf.llm_global_qa(question, index, records, k=k_global)
    return (out.get("answer", "") or "").strip()


NO_EVIDENCE_PATTERNS = [
    r"문서\s*근거를\s*찾지\s*못했다",
    r"근거\s*문서에서\s*확인할\s*수\s*없",
    r"제공된\s*context.*없",
]


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
    ap.add_argument("--kw_f1_threshold", type=float, default=0.40)
    ap.add_argument("--num_cov_threshold", type=float, default=0.40)
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

        # Auto-resolve doc_id from common fields in the question set.
        # Many datasets store the filename in "document" (e.g., "... .hwp"). We map that to doc_id = stem.
        if not doc_id:
            doc_field = ex.get("document") or ex.get("file") or ex.get("filename")
            if isinstance(doc_field, str) and doc_field.strip():
                doc_id = Path(doc_field.strip()).stem

        ex_mode = ex.get("mode") or args.mode

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
