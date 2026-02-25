import json
import re

from LLM_as_a_judge_try import evaluate_answer


def _normalize(text):
    text = "" if text is None else str(text)
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


import re

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

# ==========================================
# 폴더 구조 없이 바로 실행하는 올인원 테스트 블록
# ==========================================
if __name__ == "__main__":
    import os, json, faiss, numpy as np
    from openai import OpenAI
    from dotenv import load_dotenv

    load_dotenv()
    client = OpenAI()

    # 1. 파일 경로 설정 (현재 파일들이 있는 위치로 맞춰주세요)
    # 기본적으로 우리가 작업했던 경로로 세팅해 두었습니다.
    BASE_DIR = "/mnt/c/Users/rnrud/Documents/project/Team2-RAG-Project"
    DATASET_PATH = f"{BASE_DIR}/data/eval/test_questions.json" # 시험지 경로 (만약 data 폴더에 있다면 "data/test_questions.json" 등으로 수정)
    INDEX_PATH = f"{BASE_DIR}/data/out/faiss.index"
    RECORDS_PATH = f"{BASE_DIR}/data/out/corpus_records.json"

    print("데이터를 불러오는 중입니다...")
    index = faiss.read_index(INDEX_PATH)
    with open(RECORDS_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)

    # 2. 임시 검색기 (Retriever)
    class TempRetriever:
        def invoke(self, query):
            q_vec = np.array([client.embeddings.create(input=[query], model="text-embedding-3-small").data[0].embedding]).astype('float32')
            faiss.normalize_L2(q_vec)
            scores, indices = index.search(q_vec, 5) # 상위 5개 문서 검색
            
            class Doc:
                def __init__(self, content): self.page_content = content
            
            return [Doc(records[idx]['text']) for idx in indices[0] if idx < len(records)]

    # 3. 임시 생성기 (Generator - ask 함수 덮어쓰기)
    def temp_ask(chain, query, use_langfuse=False):
        retriever = TempRetriever()
        docs = retriever.invoke(query)
        context = "\n\n".join([d.page_content for d in docs])
        
        res = client.chat.completions.create(
            model="gpt-5-mini", # (또는 현재 작동하는 모델)
            messages=[
                {
                    "role": "system", 
                    "content": (
                        "너는 주어진 문서(Context)의 내용만 앵무새처럼 요약해서 답변하는 AI야. "
                        "절대로 너의 사전 지식을 섞지 마. 문서에 없으면 '근거 없음'이라고 해.\n\n"
                        "[답변 출력 규칙]\n"
                        "1. 단답형이 아닌 여러 줄의 설명이 필요한 답변일 경우, 무조건 첫 줄에 답변 전체를 관통하는 짧은 1문장 요약을 작성해.\n"
                        "2. 그 다음 줄부터 구체적인 상세 내용을 불릿 포인트(-)를 사용하여 작성해.\n"
                        "3. 출력 형식 예시:\n"
                        "**[핵심 요약]** 1문장 요약\n"
                        "**[상세 답변]**\n- 상세 내용 1\n- 상세 내용 2"
                    )
                },
                {
                    "role": "user", 
                    "content": f"Context:\n{context}\n\nQ: {query}\n\n위 규칙에 따라 작성해:"
                }
            ]
        )
        return res.choices[0].message.content
    # 평가 스크립트가 임시 생성기를 사용하도록 연결
    global ask
    ask = temp_ask

    # 4. 본격적인 평가 시작
    print("\n🚀 임시 실행 모드로 평가를 시작합니다!")
    final_results = run_evaluation(
        chain=None,
        dataset_path=DATASET_PATH,
        retriever=TempRetriever()
    )

    # 5. 결과 저장
    out_path = f"{BASE_DIR}/data/out/temp_evaluation_result.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 평가 완료! 결과가 다음 위치에 저장되었습니다: {out_path}")