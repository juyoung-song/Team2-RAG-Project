#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hybrid corpus builder:
- If HTML exists: parse -> section-aware blocks (headings + paragraphs + tables)
- Else: fallback to TXT (clean_text first, then extracted_raw) -> heuristic blocks
- Output: data/corpus/corpus_blocks.jsonl (doc_id, block_id, chunk_id, type, text, meta)

Run:
  cd /mnt/c/Users/rnrud/Documents/project/Team2-RAG-Project
  python scripts/build_corpus_hybrid.py \
    --raw_dir data/raw \
    --html_dir data/html \
    --clean_txt_dir data/clean_text \
    --extracted_txt_dir data/extracted_raw \
    --out data/corpus/corpus_blocks.jsonl \
    --target_ext .hwp
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from bs4 import BeautifulSoup

# ---- utils ----

def norm_stem(s: str) -> str:
    s = s.strip().lower()

    # 반복적으로 확장자 제거 (.hwp.html 같은 케이스 + 폴더명에 .hwp가 붙은 케이스)
    while True:
        new = re.sub(r"\.(hwp|hwpx|html|htm|xhtml|xml|txt)$", "", s, flags=re.I)
        if new == s:
            break
        s = new

    s = s.replace("\u3000", " ")
    s = re.sub(r"\s+", " ", s).strip()

    # 비교용 키: 공백/특수문자 제거
    return re.sub(r"[^\w가-힣]+", "", s)


def safe_read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return p.read_text(encoding="cp949", errors="ignore")

def clean_line(s: str) -> str:
    s = s.replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def chunk_by_char(text: str, target: int = 1400, overlap: int = 150) -> List[str]:
    """Simple char chunker as last resort (used only inside TXT fallback)."""
    text = text.strip()
    if len(text) <= target:
        return [text] if text else []
    chunks = []
    i = 0
    while i < len(text):
        j = min(i + target, len(text))
        # try cut at nearest newline boundary
        cut = text.rfind("\n", i, j)
        if cut < i + int(target * 0.6):
            cut = j
        chunks.append(text[i:cut].strip())
        if cut >= len(text):
            break
        i = max(cut - overlap, i + 1)
    return [c for c in chunks if c]

# ---- HTML parsing (structure-aware) ----

HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}

def html_to_blocks(html: str, max_block_chars: int = 1800) -> List[Tuple[str, str]]:
    """
    Return list of (block_type, block_text).
    Strategy:
    - Track current heading path
    - Extract text from paragraphs/list items
    - Convert tables into TSV-like text with row separation
    - Merge small adjacent items under same heading up to max_block_chars
    """
    # Many HWP HTML exports are XHTML/XML-like; lxml parser handles both.
    soup = BeautifulSoup(html, "lxml")

    body = soup.body if soup.body else soup
    items: List[Tuple[str, str]] = []

    def table_to_text(tbl) -> str:
        rows = []
        for tr in tbl.find_all("tr"):
            cols = []
            for td in tr.find_all(["td", "th"]):
                t = td.get_text(" ", strip=True)
                t = re.sub(r"\s+", " ", t)
                cols.append(t)
            if cols:
                rows.append(" | ".join(cols))
        return "\n".join(rows).strip()

    heading_stack: List[str] = []

    # iterate through elements in order
    for el in body.descendants:
        if not hasattr(el, "name"):
            continue
        name = (el.name or "").lower()

        if name in HEADING_TAGS:
            h = el.get_text(" ", strip=True)
            h = re.sub(r"\s+", " ", h).strip()
            if h:
                # reset deeper levels roughly
                level = int(name[1])
                heading_stack = heading_stack[: level - 1]
                heading_stack.append(h)

        elif name == "table":
            txt = table_to_text(el)
            if txt:
                prefix = " > ".join(heading_stack).strip()
                if prefix:
                    txt = f"[섹션] {prefix}\n[표]\n{txt}"
                else:
                    txt = f"[표]\n{txt}"
                items.append(("table", txt))

        elif name in {"p", "li"}:
            t = el.get_text(" ", strip=True)
            t = re.sub(r"\s+", " ", t).strip()
            if t:
                prefix = " > ".join(heading_stack).strip()
                if prefix:
                    t = f"[섹션] {prefix}\n{t}"
                items.append(("paragraph", t))

    # Merge adjacent items (same type) up to max_block_chars
    merged: List[Tuple[str, str]] = []
    buf_type = None
    buf = ""

    def flush():
        nonlocal buf_type, buf
        if buf and buf_type:
            merged.append((buf_type, buf.strip()))
        buf_type, buf = None, ""

    for typ, txt in items:
        txt = txt.strip()
        if not txt:
            continue
        if buf_type is None:
            buf_type = typ
            buf = txt
            continue
        if buf_type == typ and (len(buf) + 2 + len(txt)) <= max_block_chars:
            buf = buf + "\n\n" + txt
        else:
            flush()
            buf_type = typ
            buf = txt
    flush()

    # if still too big, sub-chunk
    final: List[Tuple[str, str]] = []
    for typ, txt in merged:
        if len(txt) <= max_block_chars:
            final.append((typ, txt))
        else:
            # split large tables/paras by char safely
            for c in chunk_by_char(txt, target=max_block_chars, overlap=150):
                final.append((typ, c))
    return final

# ---- TXT fallback (heuristic) ----

SECTION_PAT = re.compile(r"^\s*(Ⅰ|Ⅱ|Ⅲ|Ⅳ|Ⅴ|VI|VII|VIII|IX|X|[0-9]{1,2}\.|[0-9]{1,2}\)|[가-힣]\.)\s+.*$")

def txt_to_blocks(txt: str, max_block_chars: int = 1800) -> List[Tuple[str, str]]:
    """
    Heuristic blocks from plain text:
    - split by blank lines
    - detect section-like lines as anchors
    - merge small paragraphs under recent section header
    """
    txt = clean_line(txt)
    parts = [p.strip() for p in txt.split("\n\n") if p.strip()]
    blocks: List[Tuple[str, str]] = []

    current_section = ""
    buf = ""

    def flush():
        nonlocal buf
        if buf.strip():
            blocks.append(("block", buf.strip()))
        buf = ""

    for p in parts:
        lines = [l.strip() for l in p.split("\n") if l.strip()]
        if not lines:
            continue

        # if first line looks like section header, update current_section
        if SECTION_PAT.match(lines[0]) and len(lines[0]) <= 80:
            current_section = lines[0]
            # start new buffer with section label
            flush()
            buf = f"[섹션] {current_section}\n" + "\n".join(lines[1:]).strip()
        else:
            # normal paragraph
            add = "\n".join(lines)
            if current_section and not add.startswith("[섹션]"):
                add = f"[섹션] {current_section}\n{add}"
            if len(buf) + 2 + len(add) <= max_block_chars:
                buf = (buf + "\n\n" + add).strip() if buf else add
            else:
                flush()
                buf = add

    flush()

    # ensure not too big
    final: List[Tuple[str, str]] = []
    for typ, b in blocks:
        if len(b) <= max_block_chars:
            final.append((typ, b))
        else:
            for c in chunk_by_char(b, target=max_block_chars, overlap=150):
                final.append((typ, c))
    return final

# ---- mapping & corpus writing ----

def build_index_map(dir_path: Path, exts: Tuple[str, ...]) -> Dict[str, Path]:
    """
    Map doc key -> representative html file path.

    Supports:
    - flat files: data/html/*.html, *.xhtml ...
    - folder export: data/html/<doc_folder>/index.xhtml (or index.html)
    """
    m: Dict[str, Path] = {}

    if not dir_path.exists():
        return m

    for p in dir_path.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            # flat html file
            m[norm_stem(p.name)] = p
            continue

        if p.is_dir():
            # folder export: use index.xhtml / index.html
            idx = None
            for cand in ["index.xhtml", "index.html", "index.htm", "index.xml"]:
                cp = p / cand
                if cp.exists() and cp.is_file():
                    idx = cp
                    break

            # some exporters may not use index.*; fallback to first matching file inside
            if idx is None:
                for cp in p.rglob("*"):
                    if cp.is_file() and cp.suffix.lower() in exts:
                        idx = cp
                        break

            if idx is not None:
                # key should be folder name, not index file name
                m[norm_stem(p.name)] = idx

    return m

    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="data/raw")
    ap.add_argument("--html_dir", default="data/html")
    ap.add_argument("--clean_txt_dir", default="data/clean_text")
    ap.add_argument("--extracted_txt_dir", default="data/extracted_raw")
    ap.add_argument("--out", default="data/corpus/corpus_blocks.jsonl")
    ap.add_argument("--target_ext", default=".hwp", help="Raw doc extension for doc list, usually .hwp")
    ap.add_argument("--max_block_chars", type=int, default=1800)
    args = ap.parse_args()

    repo = Path(".")
    raw_dir = repo / args.raw_dir
    html_dir = repo / args.html_dir
    clean_dir = repo / args.clean_txt_dir
    ext_dir = repo / args.extracted_txt_dir
    out_path = repo / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # list docs from raw dir (authoritative doc set)
    raw_docs = sorted([p for p in raw_dir.glob(f"*{args.target_ext}") if p.is_file()])
    if not raw_docs:
        raise SystemExit(f"No raw docs found in {raw_dir} with ext {args.target_ext}")

    html_map = build_index_map(html_dir, (".html", ".htm", ".xhtml", ".xml"))
    clean_map = build_index_map(clean_dir, (".txt",))
    ext_map = build_index_map(ext_dir, (".txt",))

    n_html = 0
    n_txt = 0
    n_miss = 0

    with out_path.open("w", encoding="utf-8") as wf:
        for raw in raw_docs:
            doc_id = raw.stem  # keep existing convention (filename without .hwp)
            key = norm_stem(raw.name)

            # choose source: HTML > clean_txt > extracted_raw
            src_type = None
            src_path: Optional[Path] = None

            if key in html_map:
                src_type = "html"
                src_path = html_map[key]
            elif key in clean_map:
                src_type = "txt_clean"
                src_path = clean_map[key]
            elif key in ext_map:
                src_type = "txt_raw"
                src_path = ext_map[key]

            if not src_path:
                n_miss += 1
                continue

            raw_text = safe_read_text(src_path)

            if src_type == "html":
                blocks = html_to_blocks(raw_text, max_block_chars=args.max_block_chars)
                n_html += 1
            else:
                blocks = txt_to_blocks(raw_text, max_block_chars=args.max_block_chars)
                n_txt += 1

            # write blocks with stable ids
            for i, (typ, text) in enumerate(blocks):
                block_id = f"{doc_id} #{i:04d}"
                rec = {
                    "doc_id": doc_id,
                    "block_id": block_id,
                    "chunk_id": f"{i:04d}",  # per-doc chunk id
                    "type": typ if src_type == "html" else "block",
                    "text": clean_line(text),
                    "meta": {
                        "source_type": src_type,
                        "source_path": str(src_path),
                        "raw_path": str(raw),
                    },
                }
                wf.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Saved corpus: {out_path}")
    print(f"Docs: raw={len(raw_docs)} | html_used={n_html} | txt_used={n_txt} | missing={n_miss}")
    if n_miss > 0:
        print("WARNING: some raw docs had no html/txt fallback. Fill by running html conversion or hwp5txt.")

if __name__ == "__main__":
    main()
