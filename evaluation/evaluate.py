import re
import json
from pathlib import Path
from src.pipeline import rag_chain

DATA_PATH = Path(__file__).resolve().parent / "test_questions.json"

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


def nums(s: str) -> set[str]:
    """문자열에서 숫자 토큰만 추출"""
    return set(re.findall(r"\d+", norm(s)))


def is_correct(gold: str, pred_core: str) -> bool:
    """
    추천 방식 1) 양방향 포함
    추천 방식 2) 숫자 토큰 겹침(완화)
    """
    gold_n = norm(gold)
    pred_n = norm(pred_core)

    # 1) 양방향 포함(문장 길이 차이/포맷 차이 대응)
    if (gold_n and gold_n in pred_n) or (pred_n and pred_n in gold_n):
        return True

    # 2) 숫자 토큰 기반 완화 (gold에 숫자가 있으면, pred에도 같은 숫자가 1개라도 있으면 정답)
    gnums = nums(gold)
    pnums = nums(pred_core)
    if gnums and (gnums & pnums):
        return True

    return False


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
            print("===============================================\n")

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
