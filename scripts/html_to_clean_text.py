from curses import raw
import re
import csv
from pathlib import Path
from bs4 import BeautifulSoup

BASE = Path("/mnt/c/Users/rnrud/Documents/project/Team2-RAG-Project")

HTML_ROOT = BASE / "data/html"
OUT_DIR = BASE / "data/clean_text_html"
REPORT_DIR = BASE / "data/reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = REPORT_DIR / "html_to_text_report.csv"


def pick_main_html(doc_dir: Path) -> Path | None:
    """
    문서 폴더 내에서 main html/xhtml을 선택.
    우선순위: index.xhtml > index.html > 가장 큰 html/xhtml 파일
    """
    candidates = []
    for name in ["index.xhtml", "index.html", "Index.xhtml", "Index.html"]:
        p = doc_dir / name
        if p.exists():
            return p

    # 없으면 html/xhtml 파일 중 가장 큰 걸 선택
    for ext in ("*.xhtml", "*.html", "*.htm"):
        candidates.extend(doc_dir.glob(ext))

    if not candidates:
        return None

    candidates = sorted(candidates, key=lambda p: p.stat().st_size, reverse=True)
    return candidates[0]


def normalize_text(s: str) -> str:
    s = s.replace("\u00a0", " ")      # NBSP
    s = s.replace("\u200b", "")       # zero-width space
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def table_to_text(table_tag) -> str:
    """
    HTML <table>을 텍스트로 평탄화.
    - 행 단위로 " | " 구분
    - 빈 셀은 제거
    """
    rows = []
    for tr in table_tag.find_all("tr"):
        cells = []
        for cell in tr.find_all(["th", "td"]):
            txt = cell.get_text(" ", strip=True)
            txt = re.sub(r"\s+", " ", txt).strip()
            if txt:
                cells.append(txt)
        if cells:
            rows.append(" | ".join(cells))
    if not rows:
        return ""
    return "[TABLE]\n" + "\n".join(rows) + "\n[/TABLE]"


def extract_text_from_html(html_path: Path) -> tuple[str, int]:
    raw = html_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(raw, "xml")

    # 제거: script/style/nav 등
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # table은 먼저 텍스트로 치환(테이블 구조 보존)
    table_count = 0
    for t in soup.find_all("table"):
        t_text = table_to_text(t)
        if t_text:
            table_count += 1
            t.replace_with(soup.new_string("\n" + t_text + "\n"))
        else:
            t.decompose()

    # 이미지 alt(있으면)만 남기기 (필요하면 캡션 추출용)
    for img in soup.find_all("img"):
        alt = (img.get("alt") or "").strip()
        if alt:
            img.replace_with(soup.new_string(f"\n[FIGURE_ALT] {alt}\n"))
        else:
            img.decompose()

    # 줄바꿈 의미 있는 태그를 \n으로 치환
    for br in soup.find_all("br"):
        br.replace_with("\n")

    for blk in soup.find_all(["p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6"]):
        # 문단/블록 끝에 줄바꿈 강제
        if blk.string is None:
            blk.append("\n")

    text = soup.get_text("\n", strip=True)
    text = normalize_text(text)
    return text, table_count


def header_footer_prune(text: str) -> str:
    """
    보수적인 헤더/푸터 반복 제거(너무 공격적으로 제거하지 않음).
    - 동일한 짧은 라인이 과도 반복되는 경우만 제거
    """
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]

    if len(lines) < 50:
        return text

    freq = {}
    for ln in lines:
        if 3 <= len(ln) <= 40:
            freq[ln] = freq.get(ln, 0) + 1

    # 반복이 심한 라인 후보
    bad = {ln for ln, c in freq.items() if c >= 8}

    cleaned = []
    for ln in lines:
        if ln in bad:
            continue
        cleaned.append(ln)

    out = "\n".join(cleaned)
    out = normalize_text(out)
    return out


def main():
    doc_dirs = [p for p in HTML_ROOT.iterdir() if p.is_dir()]
    if not doc_dirs:
        raise SystemExit(f"No doc dirs found under: {HTML_ROOT}")

    rows = []

    for doc_dir in sorted(doc_dirs):
        doc_id = doc_dir.name
        html_path = pick_main_html(doc_dir)

        if html_path is None:
            rows.append({
                "doc_id": doc_id,
                "status": "FAIL_NO_HTML",
                "html_path": "",
                "out_txt": "",
                "char_len": 0,
                "table_count": 0,
                "note": "no html/xhtml found"
            })
            continue

        try:
            text, table_count = extract_text_from_html(html_path)
            text = header_footer_prune(text)

            out_path = OUT_DIR / f"{doc_id}.txt"
            out_path.write_text(text, encoding="utf-8")

            rows.append({
                "doc_id": doc_id,
                "status": "OK",
                "html_path": str(html_path),
                "out_txt": str(out_path),
                "char_len": len(text),
                "table_count": table_count,
                "note": ""
            })
        except Exception as e:
            rows.append({
                "doc_id": doc_id,
                "status": "FAIL_EXCEPTION",
                "html_path": str(html_path),
                "out_txt": "",
                "char_len": 0,
                "table_count": 0,
                "note": str(e)[:2000]
            })

    # report 저장
    with REPORT_PATH.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    ok = sum(1 for r in rows if r["status"] == "OK")
    fail = len(rows) - ok
    print("Done.")
    print("OK:", ok, "| FAIL:", fail)
    print("OUT_DIR:", OUT_DIR)
    print("REPORT:", REPORT_PATH)


if __name__ == "__main__":
    main()
