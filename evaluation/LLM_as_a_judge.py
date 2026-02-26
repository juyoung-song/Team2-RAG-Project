from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

judge_prompt = """
당신은 RAG 시스템이 생성한 답변의 정확성을 평가하는 엄격하고 공정한 심사위원입니다.
주어진 [질문], [정답(Context)], [예측 답변]을 비교하여 점수를 매겨주세요.

[평가 기준]
- 1: 예측 답변이 정답의 핵심 내용을 모두 포함하며, 모순되는 정보가 없습니다. 예측 답변이 정답보다 더 상세한 정보(예: 금액 계산, 구체적인 파이프라인 설명 등)를 포함하더라도, 그것이 질문의 문맥상 올바르고 정답과 충돌하지 않는다면 1점으로 인정해야 합니다.
- 0.5: 예측 답변이 정답의 일부 내용만 포함하거나 덜 핵심적인 사소한 오류가 있습니다.
- 0: 예측 답변이 정답과 전혀 다르거나, 정답과 정면으로 모순되는 정보를 포함합니다.

[질문]
{query}

[정답(Context)]
{context}

[예측 답변]
{answer}

위 평가 기준에 따라 판단한 후, 절대로 이유나 부연 설명을 덧붙이지 말고 오직 '1', '0.5', '0' 중 하나의 숫자만 출력하세요.
"""

judge_template = PromptTemplate.from_template(judge_prompt)


def evaluate_answer(query, answer, context):
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    judge = llm.invoke(judge_template.format(query=query, answer=answer, context=context))
    return float(judge.content.strip())