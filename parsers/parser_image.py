import base64
from pathlib import Path

import fitz
from langchain_core.documents import Document
from openai import OpenAI

PDF_DIR = Path(__file__).parent.parent / "data" / "pdfs"
IMAGE_OUT_DIR = Path(__file__).parent.parent / "data" / "parsed" / "image"
MIN_SIZE = 100  # 로고·장식 이미지 제외 기준 (px)

VISION_PROMPT = """
당신은 RAG(검색 증강 생성) 시스템의 문서 데이터 추출 전문가입니다. 
이 이미지는 제안요청서(RFP) 관련 문서입니다.
데이터베이스에서 의미 기반 검색(Semantic Search)이 최적화되도록 아래 규칙을 엄격히 지켜 정보만 추출하세요.

[규칙]
1. 시각적 묘사 절대 금지: "좌측 상단", "중앙 다이어그램", "빨간색 삼각형", "슬라이드 제목" 등 위치나 형태를 묘사하는 단어는 노이즈이므로 절대 쓰지 마세요.
2. 의미와 논리 관계만 추출: 그림이 나타내는 계층 구조, 포함 관계, 인과 관계 등을 논리적인 글머리 기호나 자연스러운 문장으로 풀어쓰세요. (예: "교육구국이라는 최상위 이념 아래 자유, 정의, 진리의 가치가 있다")
3. 서론/결론 생략: "아래는 요약입니다", "이 이미지는~" 등의 불필요한 인사말이나 출처 표기 없이 곧바로 본문 핵심 내용만 마크다운으로 출력하세요.
4. 검색 최적화: 사용자가 '고려대학교 교육 이념', '인재상', '교육 목표' 등의 키워드로 질문했을 때 이 텍스트가 정확히 검색될 수 있도록 명사형 키워드 중심으로 간결하게 정리하세요.
"""

def parse_images(pdf_dir: Path = PDF_DIR) -> list[Document]:
    client = OpenAI()
    docs = []

    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        out_dir = IMAGE_OUT_DIR / pdf_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)

        # 이미 jsonl이 있는 PDF는 건너뜀 (resume 가능)
        jsonl_path = IMAGE_OUT_DIR / (pdf_path.stem + ".jsonl")
        if jsonl_path.exists():
            print(f"  [skip] {pdf_path.name} (이미 완료)")
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    r = __import__("json").loads(line)
                    docs.append(Document(page_content=r["text"], metadata={
                        "source_file": r["source_file"],
                        "page": r["page"],
                        "type": "image",
                    }))
            continue

        print(f"  파싱 중: {pdf_path.name}")
        pdf = fitz.open(str(pdf_path))
        pdf_docs = []

        for page_num in range(len(pdf)):
            for idx, img_info in enumerate(pdf[page_num].get_images(full=True)):
                xref, width, height = img_info[0], img_info[2], img_info[3]
                if width <= MIN_SIZE or height <= MIN_SIZE:
                    continue

                base_img = pdf.extract_image(xref)
                img_bytes, ext = base_img["image"], base_img["ext"]

                # 이미지 파일 저장
                (out_dir / f"page{page_num+1}_img{idx+1}.{ext}").write_bytes(img_bytes)

                # Vision 요약 (SSL 등 일시 오류 시 최대 3회 재시도)
                mime = "image/png" if ext == "png" else "image/jpeg"
                b64 = base64.b64encode(img_bytes).decode()

                response = None
                for attempt in range(3):
                    try:
                        response = client.chat.completions.create(
                            model="gpt-5-mini",
                            messages=[{"role": "user", "content": [
                                {"type": "text", "text": VISION_PROMPT},
                                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                            ]}],
                            max_completion_tokens=4000,
                        )
                        break  # 성공 시 재시도 루프 탈출
                    except Exception as e:
                        print(f"    [재시도 {attempt+1}/3] 오류: {e}")
                        if attempt == 2:
                            print(f"    → 이미지 skip (page{page_num+1}_img{idx+1})")

                if response is None:
                    continue  # 3회 모두 실패 시 해당 이미지 건너뜀


                pdf_docs.append(Document(
                    page_content=response.choices[0].message.content.strip(),
                    metadata={
                        "source_file": pdf_path.name,
                        "page": page_num,
                        "type": "image",
                    },
                ))

        pdf.close()

        # PDF 1개 완료 즉시 jsonl 저장 (중단해도 보존됨)
        import json
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for d in pdf_docs:
                f.write(json.dumps({
                    "source_file": d.metadata["source_file"],
                    "page": d.metadata["page"],
                    "type": "image",
                    "text": d.page_content,
                }, ensure_ascii=False) + "\n")

        docs.extend(pdf_docs)
        print(f"  → {pdf_path.name}: 이미지 {len(pdf_docs)}개 저장 완료")

    return docs

