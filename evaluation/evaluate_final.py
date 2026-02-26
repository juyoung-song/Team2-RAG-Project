import json
import re

from src.generator import ask
from evaluation.LLM_as_a_judge import evaluate_answer


from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

#Step 1: 텍스트 전처리 및 후처리 유틸리티 함수
"""텍스트 평가 전에 공백, 대소문자 등을 정리하는 보조 함수"""

def _normalize(text):
    text = "" if text is None else str(text)
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text

#Step 2: LLM을 채점관으로 활용하는 함수 (soft_score)
"""키워드 매칭이 아닌, LLM(gpt-5-mini)에게 정답과 답안을 주고 채점하게 하는 함수입니다."""
def soft_score(pred, gold):
    if not pred or not gold:
        return 0.0

    pred_n = str(pred).replace(" ", "").lower()
    gold_n = str(gold).replace(" ", "").lower()
    if gold_n in pred_n:
        return 1.0

    try:
        sys_prompt = (
            "너는 RAG 시스템의 답변을 평가하는 지능적이고 실무적인 채점관이야. "
            "정답(Gold)과 학생의 답안(Pred)을 비교해서 다음 기준에 따라 'PASS' 또는 'FAIL'만 출력해.\n\n"
            "[채점 기준]\n"
            "1. 의미적 동의어 및 구체화 인정: 정답의 추상적인 개념(예: UI/UX 개선)을 학생이 구체적인 사례(예: 사용자 편의성을 위한 인터페이스 개선)로 풀어서 설명했다면 완벽한 PASS로 인정해.\n"
            "2. 부분 정답의 포용: 정답에 여러 조건(예: UI/UX 개선 및 모바일 강화)이 있을 때, 학생이 그중 핵심 절반 이상을 맞추고 RAG 문서의 관련 부가 설명을 덧붙였다면 문맥이 일치하므로 PASS.\n"
            "3. 숫자/수치 엄격 확인: 단, 금액, 코어 수(예: 16코어 vs 8코어), 퍼센트 등 '정확한 수치'가 다르면 무조건 FAIL.\n"
            "4. 모순 금지: 학생 답안이 정답과 완전히 반대되거나 엉뚱한 사업을 설명하면 FAIL.\n\n"
            "결과는 반드시 'PASS' 또는 'FAIL' 단어 하나만 출력해."
        )
        user_prompt = f"[정답]\n{gold}\n\n[학생 답안]\n{pred}\n\n채점 결과 (PASS 또는 FAIL):"

        res = client.chat.completions.create(
            model="gpt-5-mini", # (또는 현재 사용 중인 모델)
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ],
        )
        
        result_text = res.choices[0].message.content.strip().upper()
        return 1.0 if "PASS" in result_text else 0.0

    except Exception as e:
        print(f"채점 중 에러 발생: {e}")
        return 0.0
    
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

#Step 3: 메인 평가 파이프라인 (run_evaluation)
"""평가 데이터셋을 불러와서 한 질문씩 RAG 시스템에 답변하게 하고, 그 결과를 채점합니다."""
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
        score = soft_score(pred_core, gold)
        correct = score >= threshold
        if correct:
            correct_count += 1

        # LLM judge 점수: retriever가 전달된 경우에만 계산
        llm_judge_score = None
        if retriever is not None:
            try:
                context_docs = retriever.invoke(question)
                context_text = "\n\n".join(
                    getattr(d, "page_content", "") for d in context_docs
                )
                llm_judge_score = evaluate_answer(question, pred_core, context_text)
            except Exception:
                llm_judge_score = None

        results.append(
            {
                "id": item.get("id", i),
                "question": question,
                "gold": gold,
                "prediction": prediction,
                "pred_core": pred_core,
                "score": score,
                "correct": correct,
                "llm_judge_score": llm_judge_score,
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