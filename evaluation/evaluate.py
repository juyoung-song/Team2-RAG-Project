import re
import json
import math
from pathlib import Path
from functools import lru_cache

from src.pipeline import rag_chain


# =========================
# Config
# =========================
USE_EMBED_JUDGE = True
EMBED_MODEL = "text-embedding-3-small"

# 임베딩 유사도 임계값(경험값)
SIM_THRESHOLD_STRICT = 0.83   # 웬만하면 정답
SIM_THRESHOLD_LOOSE  = 0.78   # 엔티티도 어느 정도 맞으면 정답

DATA_PATH = Path(__file__).resolve().parent / "test_questions.json"

NO_EVIDENCE_KEY = "근거 문서에서 확인할 수 없습니다"

# 한국어 불용어(최소 세트)
STOPWORDS = {
    "및", "또는", "그리고", "에서", "으로", "하는", "합니다", "있습니다", "없습니다",
    "대한", "관련", "경우", "위해", "사업", "시스템", "요구", "구축", "고도화",
    "지원", "수행", "기반"
}


# =========================
# Helpers: parsing / normalization
# =========================
def extract_answer_text(prediction_text: str) -> str:
    """
    모델 출력이 아래 형태일 때:
      ANSWER: ...
      SOURCES:
      - ...
    채점에 사용할 ANSWER 본문만 뽑아낸다.
    """
    s = (prediction_text or "").strip()

    # SOURCES 이하 제거
    s = re.split(r"\n\s*SOURCES\s*:", s, maxsplit=1)[0]

    # "ANSWER:" prefix 제거
    s = re.sub(r"^\s*ANSWER\s*:\s*", "", s)

    return s.strip()


def norm(s: str) -> str:
    """공백/쉼표 제거 등 최소 정규화"""
    if s is None:
        return ""
    s = str(s)
    s = re.sub(r"\s+", "", s)
    s = s.replace(",", "")
    return s

def _fix_broken_korean_spaces(s: str) -> str:
    """
    OCR/파싱 결과에서 '서 울', '세 종' 처럼 한글 글자 사이에 공백이 끼는 경우가 많음.
    한글-공백-한글 패턴을 반복적으로 붙여서 '서울', '세종'으로 복원.
    """
    if not s:
        return ""
    s = str(s)
    # 반복 치환 (여러 번 끊긴 경우 대비)
    for _ in range(3):
        s2 = re.sub(r"([가-힣])\s+([가-힣])", r"\1\2", s)
        if s2 == s:
            break
        s = s2
    return s


def _strip_leading_bullets(s: str) -> str:
    """
    '5. 코드오류', '- 서울', '※ ...' 같은 접두 기호/번호 제거
    """
    if not s:
        return ""
    s = s.strip()
    s = re.sub(r"^\s*[\-\*\•\·\※]+\s*", "", s)
    s = re.sub(r"^\s*\d+\s*[\.\)\:]\s*", "", s)  # 5. / 5) / 5: 등
    return s.strip()


def _shorten_for_judge(s: str, max_chars: int = 140) -> str:
    """
    임베딩/엔티티 비교는 길면 오히려 유사도가 떨어질 수 있어서
    핵심만 남기기(앞부분 1~2문장 수준).
    """
    s = _fix_broken_korean_spaces(s)
    s = _strip_leading_bullets(s)
    s = _clean_text(s)

    # SOURCES/부록성 문구 제거(혹시 남아있을 때)
    s = re.split(r"\bSOURCES\b\s*:", s, maxsplit=1)[0].strip()

    # 문장 경계 기준으로 앞부분만
    parts = re.split(r"[\.!?。]\s+|(?:다\.)\s+", s)
    if parts and len(parts[0]) >= 15:
        s = parts[0]
    else:
        s = s[:max_chars]

    return s[:max_chars].strip()


def _clean_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = re.sub(r"[\s\u200b]+", " ", s).strip()
    s = re.sub(r"[·•\-\(\)\[\]{}<>\"'“”‘’]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _ko_terms(s: str) -> list[str]:
    """
    한국어/영문/숫자 토큰 추출(간단).
    - 깨진 한글 띄어쓰기(서 울) 복원
    - 앞번호/불릿 제거
    - 길이 2 이상
    - 불용어 제거
    """
    s = _fix_broken_korean_spaces(s)
    s = _strip_leading_bullets(s)
    s = _clean_text(s)

    toks = re.findall(r"[가-힣A-Za-z0-9]+", s)
    out = []
    for t in toks:
        t2 = t.lower()
        if len(t2) < 2:
            continue
        if t2 in STOPWORDS:
            continue
        out.append(t2)
    return out


def _entity_f1(gold: str, pred: str) -> float:
    """키워드/엔티티 기반 F1"""
    g = set(_ko_terms(gold))
    p = set(_ko_terms(pred))
    if not g or not p:
        return 0.0
    inter = len(g & p)
    prec = inter / len(p)
    rec = inter / len(g)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def _num_match_score(gold: str, pred: str) -> float:
    """
    숫자 토큰 매칭을 점수로.
    - gold 숫자 중 pred가 맞춘 비율(cover ratio)
    """
    gnums = set(re.findall(r"\d+", norm(gold)))
    pnums = set(re.findall(r"\d+", norm(pred)))
    if not gnums:
        return 0.0
    inter = len(gnums & pnums)
    return inter / len(gnums)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


@lru_cache(maxsize=4096)
def _embed(text: str) -> list[float]:
    """
    임베딩 결과 캐시(평가 중 반복 호출 절감)
    """
    from openai import OpenAI
    client = OpenAI()

    text = _clean_text(text)
    if not text:
        return []

    r = client.embeddings.create(model=EMBED_MODEL, input=text)
    return r.data[0].embedding


def _semantic_sim(gold: str, pred: str) -> float:
    """임베딩 코사인 유사도"""
    try:
        a = _embed(gold)
        b = _embed(pred)
        if not a or not b or len(a) != len(b):
            return 0.0
        return _cosine(a, b)
    except Exception:
        # 임베딩 호출 실패해도 평가가 중단되지 않게
        return 0.0


# =========================
# Judge
# =========================
def is_correct(gold: str, pred_core: str) -> bool:
    """
    개선판 채점:
    0) 근거 없음/빈 답변이면 X
    1) 포함(짧게 정규화 후)
    2) 숫자 커버율
    3) 엔티티/키워드 F1
    4) 임베딩 의미 유사(짧게 축약한 텍스트로 비교)
    """
    gold = gold or ""
    pred_core = pred_core or ""

    # (0) "근거 없음"은 무조건 X
    pred_n0 = norm(pred_core)
    if not pred_n0:
        return False
    if norm(NO_EVIDENCE_KEY) in pred_n0:
        return False

    # 비교용 텍스트(짧게 + 정규화 강화)
    gold_j = _shorten_for_judge(gold)
    pred_j = _shorten_for_judge(pred_core)

    gold_n = norm(gold_j)
    pred_n = norm(pred_j)

    # (1) 포함 매칭
    if (gold_n and gold_n in pred_n) or (pred_n and pred_n in gold_n):
        return True

    # (2) 숫자 커버율 (gold 숫자의 60% 이상 맞추면 True)
    num_score = _num_match_score(gold_j, pred_j)
    if num_score >= 0.6:
        return True

    # (3) 엔티티/키워드 F1
    ent_f1 = _entity_f1(gold_j, pred_j)
    if ent_f1 >= 0.65:
        return True

    # (4) 임베딩 의미 유사
    if USE_EMBED_JUDGE:
        sim = _semantic_sim(gold_j, pred_j)
        if sim >= SIM_THRESHOLD_STRICT:
            return True
        if sim >= SIM_THRESHOLD_LOOSE and ent_f1 >= 0.35:
            return True

    return False


# =========================
# Evaluation runner
# =========================
def run_evaluation():
    # 1) 데이터 로드
    try:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print("에러: test_questions.json 파일이 없습니다.")
        return

    # 2) 질문 및 정답 추출
    questions = [item["question"] for item in dataset]
    answer_key = [item["answer"] for item in dataset]

    correct_count = 0

    print("🧐 RAG 시스템 평가를 시작합니다...\n")

    DEBUG_N = 30  # 필요 시 0으로

    # 3) 답변 생성 및 채점
    for i, (q, gold) in enumerate(zip(questions, answer_key), 1):
        prediction = rag_chain.invoke(q)
        prediction_text = prediction if isinstance(prediction, str) else str(prediction)

        # ANSWER 본문만 추출해서 채점
        pred_core = extract_answer_text(prediction_text)

        if i <= DEBUG_N:
            print(f"\n================= DEBUG Q{i:02d} =================")
            print("Q:", q)
            print("GOLD:", gold)
            print("\n--- PRED_CORE(300) ---")
            print(pred_core[:300])

        #     # 디버그용 점수 확인(원하면 유지)
        #     ent_f1 = _entity_f1(gold, pred_core)
        #     num_score = _num_match_score(gold, pred_core)
        #     if USE_EMBED_JUDGE:
        #         sim = _semantic_sim(gold, pred_core)
        #         print(f"\n--- SCORES ---")
        #         print(f"num_cover={num_score:.3f} | ent_f1={ent_f1:.3f} | embed_sim={sim:.3f}")
        #     else:
        #         print(f"\n--- SCORES ---")
        #         print(f"num_cover={num_score:.3f} | ent_f1={ent_f1:.3f}")

        #     print("===============================================\n")

        ok = is_correct(gold, pred_core)

        if ok:
            correct_count += 1
            status = "O"
        else:
            status = "X"

        print(f"Q{i:02d}: [{status}] | 질문: {q[:30]}...")

    # 4) 최종 결과 출력
    print("\n" + "=" * 30)
    print(f"평가 완료: {correct_count} / {len(questions)} 맞춤")
    print("=" * 30)


if __name__ == "__main__":
    run_evaluation()
