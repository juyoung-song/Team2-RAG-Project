import json
import re

from src.generator import ask


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
        score = soft_score(pred_core, gold)
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
                "score": score,
                "correct": correct,
            }
        )

        print(f"================= DEBUG Q{i:02d} =================")
        print(f"Q: {question}")
        print(f"GOLD: {gold}")
        print("")
        print(f"--- PRED_CORE({pred_preview_chars}) ---")
        print(pred_core[:pred_preview_chars])
        print("")
        print(f"Q{i:02d}: [{'O' if correct else 'X'}]")

    print(f"\n평가 완료: {correct_count} / {len(results)}")
    return results
