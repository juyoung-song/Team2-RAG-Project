from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model_name="gpt-5-mini", temperature=0)

judge_prompt = """
다음은 RAG 모델이 생성한 답변과, 참조 문서(Context)입니다.
답변이 문서에 근거했는지 0~1 점수로 평가하세요.

- 1: 문서에서 근거를 명확히 가지고 있음
- 0: 문서에 없는 내용을 생성함 (hallucination)
- 0.5: 부분적으로만 근거가 있음

[질문]
{query}

[답변]
{answer}

[문맥]
{context}

점수만 출력하세요.
"""

from langchain_core.prompts import PromptTemplate
judge_template = PromptTemplate.from_template(judge_prompt)

def evaluate_answer(query, answer, context):
    judge = llm.invoke(judge_template.format(query=query, answer=answer, context=context))
    return float(judge.content.strip())