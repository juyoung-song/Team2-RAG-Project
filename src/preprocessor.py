import json
import re
import unicodedata
from collections import Counter
from pathlib import Path


def preprocess_parsed_dir(src_dir, dst_dir):
    """data/parsed -> data/preprocessed"""
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    page_no_pattern = re.compile(r"^-\s*\d+\s*-$")

    for fp in sorted(src_dir.glob("*.jsonl")):
        rows = [json.loads(line) for line in fp.open("r", encoding="utf-8")]
        pages = [r.get("text", "") for r in rows]

        # 페이지 상/하단 반복 줄(헤더/푸터 후보) 탐지
        top, bottom = Counter(), Counter()
        for text in pages:
            lines = [x.strip() for x in text.splitlines() if x.strip()]
            top.update(lines[:2])
            bottom.update(lines[-2:])

        n = max(1, len(pages))
        repeated = {k for k, v in top.items() if v / n >= 0.5}
        repeated |= {k for k, v in bottom.items() if v / n >= 0.5}

        out_path = dst_dir / fp.name
        with out_path.open("w", encoding="utf-8") as out:
            for r in rows:
                s = r.get("text", "")

                # 1) 제어문자 제거 (\n, \t는 유지)
                s = "".join(
                    ch for ch in s if unicodedata.category(ch)[0] != "C" or ch in "\n\t"
                )
                # 2) 헤더/푸터 + 페이지번호 패턴 제거
                s = "\n".join(
                    ln
                    for ln in s.splitlines()
                    if ln.strip() not in repeated and not page_no_pattern.match(ln.strip())
                )
                # 3) 공백/줄바꿈 정리
                s = re.sub(r"\r\n?", "\n", s)
                s = re.sub(r"[ \t]+", " ", s)
                s = re.sub(r"\n{2,}", "\n", s).strip()

                r["text"] = s
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
