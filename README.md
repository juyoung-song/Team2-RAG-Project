# 🔍 Team2 — RFP 문서 기반 RAG 질의응답 시스템

> 공공기관 제안요청서(RFP) PDF 100개를 텍스트·표·이미지까지 파싱하여  
> 정확한 답변을 제공하는 **Advanced RAG 시스템**

---

## 🎯 프로젝트 개요

| 항목 | 내용 |
|---|---|
| 데이터 | 공공기관 RFP PDF 100개 (약 7,500 페이지) |
| 파싱 | 텍스트(PyMuPDF) · 표(pdfplumber) · 이미지(GPT Vision) |
| 검색 | Hybrid Retriever (Dense 70% + BM25 30%) |
| 생성 | OpenAI gpt-5-mini + LangChain LCEL |
| 평가 | LLM as a Judge (gpt-4o-mini) · 30문항 |
| UI | Streamlit |

---

## 📑 최종 보고서

> [Team2_RAG_보고서.pdf](https://github.com/user-attachments/files/25600246/Team2_RAG_.pdf)


---

## 👥 Collaborators & Logs

| 팀원 | 역할 | 협업 일지 |
| :---: | :--- | :---: |
| <a href="https://github.com/juyoung-song"><img src="https://github.com/juyoung-song.png" width="80" title="송주영"></a> |. | <a href="https://www.notion.so/Daily-2fc0ddb4d3a880818a84edc3a7799050?source=copy_link"><img src="https://img.shields.io/badge/Notion-000000?style=flat-square&logo=notion&logoColor=white"></a> |
| <a href="https://github.com/choimuyeong"><img src="https://github.com/choimuyeong.png" width="80" title="최무영"></a> | . | <a href="https://www.notion.so/Daily-2fc0ddb4d3a88093a9b0d241ec496adb?source=copy_link"><img src="https://img.shields.io/badge/Notion-000000?style=flat-square&logo=notion&logoColor=white"></a> |
| <a href="https://github.com/Yumin-Hwang046"><img src="https://github.com/Yumin-Hwang046.png" width="80" title="황유민"></a> | . | <a href="https://www.notion.so/Daily-2fc0ddb4d3a880888eb1c0232db60f0f?source=copy_link"><img src="https://img.shields.io/badge/Notion-000000?style=flat-square&logo=notion&logoColor=white"></a> |
| <a href="https://github.com/rnrudwns123-design"><img src="https://github.com/rnrudwns123-design.png" width="80" title="구경준"></a> | . | <a href="https://www.notion.so/Daily-2fc0ddb4d3a880f19759c6baf4217b11?source=copy_link"><img src="https://img.shields.io/badge/Notion-000000?style=flat-square&logo=notion&logoColor=white"></a> |
| <a href="https://github.com/sfun1993-bit"><img src="https://github.com/sfun1993-bit.png" width="80" title="서정언"></a> | . | <a href="https://www.notion.so/Daily-2fc0ddb4d3a880d2b1fbc148b9c236d0?source=copy_link"><img src="https://img.shields.io/badge/Notion-000000?style=flat-square&logo=notion&logoColor=white"></a> |

---

## 🛠️ Tools and Technologies

<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white">
  <img src="https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black">
  <img src="https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white">
  <img src="https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white">
  <img src="https://img.shields.io/badge/FAISS-0467DF?style=for-the-badge&logo=meta&logoColor=white">
</div>

---

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 가상환경 생성
conda create -p .conda python=3.10 -y
conda activate .conda/

# 패키지 설치
pip install -r requirements.txt
```

### 2. API 키 설정

```bash
cp .env.example .env
# .env 파일을 열고 OPENAI_API_KEY 입력
```

### 3. Streamlit 웹 UI 실행

```bash
conda run -p .conda streamlit run src/streamlit.py
```

> **주의**: 첫 실행 시 `data/vectorstore/faiss_advanced/` 가 없으면 자동으로 생성합니다 (시간 소요).

---

## 🔄 파이프라인 실행 순서

처음부터 전체를 구성하려면 아래 순서로 실행하세요.

```python
# 1. 파싱 (이미 완료된 PDF는 자동 skip)
from parsers.parser_image import parse_images
parse_images()   # 이미지 파싱 (시간 소요: 수 시간)

# 2. 전체 파이프라인 한 번에 구성
from src.pipeline import build_pipeline
from src.generator import ask

chain = build_pipeline()               # DB가 없으면 자동 생성, 있으면 로드
answer = ask(chain, "질문을 입력하세요")
print(answer)
```

---

## 🏗️ 프로젝트 구조

```
Team2-RAG-Project/
├── data/
│   ├── pdfs/               # 원본 PDF 100개
│   ├── parsed/
│   │   ├── *.jsonl         # 텍스트 파싱 결과
│   │   ├── table/          # 표 파싱 결과 (마크다운)
│   │   ├── image/          # 이미지 파싱 결과 (Vision 요약)
│   │   └── merged/         # 3종 통합 결과
│   ├── preprocessed/       # 전처리 완료 텍스트
│   ├── vectorstore/
│   │   ├── faiss_advanced/ # Advanced RAG용 FAISS DB (텍스트+표+이미지)
│   │   └── faiss_openai/   # Naive RAG용 FAISS DB (텍스트만)
│   └── raw/data_list.csv   # PDF 메타데이터 (기관명, 사업명 등)
├── parsers/
│   ├── parser_text.py      # 텍스트 파싱 (PyMuPDF)
│   ├── parser_table.py     # 표 파싱 → 마크다운 (pdfplumber)
│   └── parser_image.py     # 이미지 파싱 → 요약 (GPT Vision)
├── src/
│   ├── pipeline.py         # ★ 전체 파이프라인 진입점
│   ├── loader.py           # 파싱 결과 통합 로드
│   ├── preprocessor.py     # 텍스트 전처리
│   ├── embedding.py        # 청킹 + FAISS DB 생성
│   ├── retriever.py        # Hybrid Retriever (Dense + BM25)
│   ├── generator.py        # Chain 생성 + ask()
│   └── streamlit.py        # 웹 UI
├── retrievers/             # Retriever 실험용 변형들
├── evaluation/
│   ├── test_questions.json # 평가 데이터셋 (30문항)
│   ├── evaluate_final.py   # 전체 평가 실행
│   └── LLM_as_a_judge.py  # GPT 채점기
├── notebooks/
│   ├── EDA/                # 데이터 탐색
│   ├── experiments/        # 파서·청킹·Retriever 실험
│   └── RAGsystem/          # NaiveRAG / AdvancedRAG 구현 노트북
└── requirements.txt
```

---

## 📊 실험 결과 요약

### 청킹 파라미터 (그리드서치)

| chunk_size | overlap | k | MRR |
|:---:|:---:|:---:|:---:|
| **700** | **70** | **12** | **0.1342** ✅ |
| 900 | 90 | 10 | 0.1311 |
| 700 | 70 | 10 | 0.1323 |

### NaiveRAG 평가 결과

| 평가 항목 | 결과 |
|---|---|
| 전체 점수 | **8 / 30** |
| 채점 방식 | LLM as a Judge (gpt-4o-mini) |

> AdvancedRAG (표+이미지 포함) 평가는 이미지 파싱 완료 후 업데이트 예정

---

## ⚙️ 주요 설정값

| 항목 | 값 |
|---|---|
| Embedding 모델 | `text-embedding-3-small` |
| LLM | `gpt-5-mini` |
| 평가 LLM | `gpt-4o-mini` |
| chunk_size | 700 |
| chunk_overlap | 70 |
| Retriever k | 12 (Dense 70% + BM25 30%) |
