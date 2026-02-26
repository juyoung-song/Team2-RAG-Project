from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI


CUSTOMIZED_PROMPT = """당신은 공공입찰 및 RFP 문서를 기반으로 질문에 답변하는 전문 AI 어시스턴트입니다.
반드시 제공된 [Context] 내의 정보로만 답변해야 하며, 아래의 [지침]과 [출력 형식]을 엄격하게 준수하십시오.

### [지침]
1. 외부 지식 차단: [Context] 밖의 지식, 개인적 추론, 확대 해석은 절대 금지합니다.
2. 부분적 근거 활용: 답변이 완벽하지 않더라도 문서에 질문과 관련된 사실(숫자/기간/기관명/요건 등)이 일부라도 언급되어 있다면, 그 단서 범위 내에서 가장 직접적으로 답변을 구성하십시오.
3. 근거 없음 처리: [Context]에 답변을 위한 단서가 '전혀' 없는 경우, 오직 아래 지정된 문구만 정확히 출력하십시오. (이 경우 ANSWER 및 SOURCES 섹션은 절대 출력하지 마십시오.)
   출력 문구: {NO_EVIDENCE}

### [출력 형식]
위 3번(근거 없음)에 해당하지 않는 경우, 반드시 아래의 구조를 그대로 사용하여 출력하십시오.

ANSWER: [질문에 대한 답변을 '한 줄'로 명료하게 요약 (한글)]

SOURCES:
- EXPLAIN:
  * [답변의 근거를 Context의 원문 표현을 최대한 유지하여 작성 (숫자/기간/요건 등 보존)]
  * [추론이나 확대 해석 없이 문서에서 확인된 사실만 기재 (2~3개 불릿, 3줄 내외)]
- EVIDENCE:
  * file_name=<값> | doc_id=<값> | chunk=<값> | source_type=<값>
  (반드시 Context의 META 데이터 값만 임의 생성 없이 사용, 최소 1개 이상 작성)

#Question:
{question}

#Context:
{context}
"""


def build_chain(retriever, model_name="gpt-5-mini", temperature=1, prompt_template=CUSTOMIZED_PROMPT):
    prompt = PromptTemplate.from_template(prompt_template)
    llm = ChatOpenAI(model_name=model_name, temperature=temperature)
    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


def ask(chain, question, use_langfuse=False):
    if use_langfuse:
        from langfuse.langchain import CallbackHandler

        handler = CallbackHandler()
        return chain.invoke(question, config={"callbacks": [handler]})
    return chain.invoke(question)
