import json
import re

from src.generator import ask
from evaluation.LLM_as_a_judge import evaluate_answer


def _normalize(text):
    text = "" if text is None else str(text)
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def soft_score(pred, gold):
    pred_n = _normalize(pred).replace(" ", "")
    gold_n = _normalize(gold).replace(" ", "")

    if not pred_n or not gold_n:
        return 0.0

    if gold_n in pred_n or pred_n in gold_n:
        return 1.0

    p_set, g_set = set(pred_n), set(gold_n)
    inter = len(p_set & g_set)
    union = len(p_set | g_set)
    return inter / union if union else 0.0


def _extract_pred_core(prediction_text):
    """
    모델 출력에서 채점/디버그에 사용할 핵심 답변만 추출.
    - 'SOURCES:' 이후는 제거
    - 맨 앞 'ANSWER:' 접두어 제거
    """
    text = "" if prediction_text is None else str(prediction_text).strip()
    text = re.split(r"\n\s*SOURCES\s*:", text, maxsplit=1)[0]
    text = re.sub(r"^\s*ANSWER\s*:\s*", "", text)
    return text.strip()


def run_evaluation(
    chain,
    dataset_path,
    threshold=0.35,
    use_langfuse=False,
    pred_preview_chars=300,
    retriever=None,
):
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    results = []
    correct_count = 0
    print("RAG 시스템 평가 시작...\n")

    for i, item in enumerate(dataset, 1):
        question = item["question"]
        gold = item["answer"]

        try:
            prediction = ask(chain, question, use_langfuse=use_langfuse)
        except Exception as e:
            prediction = f"[ERROR] {e}"

        if not isinstance(prediction, str):
            prediction = str(prediction)

        pred_core = _extract_pred_core(prediction)
        
        # 1. soft_score는 보조 지표로 계산만 해둡니다.
        overlap_score = soft_score(pred_core, gold)
        
        # 2. LLM judge 점수를 계산합니다. (context_text 대신 gold를 넘겨줍니다!)
        llm_judge_score = None
        try:
            # 우리가 수정한 프롬프트가 '정답'과 비교하도록 되어 있으므로 gold를 넣습니다.
            llm_judge_score = evaluate_answer(question, pred_core, gold)
        except Exception as e:
            print(f"[LLM 평가 오류] {e}")
            llm_judge_score = None

        # 3. O/X 판단 로직을 LLM 점수 기준으로 변경합니다.
        # LLM 점수가 성공적으로 나왔다면 그 점수를 기준(예: 0.5 이상이면 정답)으로 삼고, 
        # 오류가 나서 점수가 없다면 기존 overlap_score를 사용합니다.
        if llm_judge_score is not None:
            score = llm_judge_score
            correct = score >= 0.5  # 0.5점(부분 점수) 이상이면 'O'로 처리 (원하신다면 1.0으로 더 엄격하게 바꿀 수 있습니다)
        else:
            score = overlap_score
            correct = score >= threshold

        if correct:
            correct_count += 1

        results.append(
            {
                "id": item.get("id", i),
                "question": question,
                "gold": gold,
                "prediction": prediction,
                "pred_core": pred_core,
                "score": score,               # 최종 사용된 점수
                "llm_judge_score": llm_judge_score, # LLM 원본 점수
            }
        )

        print(f"================= DEBUG Q{i:02d} =================")
        print(f"Q: {question}")
        print(f"GOLD: {gold}")
        print("")
        print(f"--- PRED_CORE({pred_preview_chars}) ---")
        print(pred_core[:pred_preview_chars])
        print("")
        # LLM 점수도 콘솔에서 바로 확인할 수 있도록 출력에 추가합니다.
        print(f"LLM Judge Score: {llm_judge_score}") 
        print(f"Q{i:02d}: [{'O' if correct else 'X'}]")

    print(f"\n평가 완료: {correct_count} / {len(results)}")
    return results