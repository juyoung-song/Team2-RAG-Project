from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI


DEFAULT_PROMPT = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Answer in Korean.

#Question:
{question}
#Context:
{context}

#Answer:"""


def build_chain(retriever, model_name="gpt-5-mini", temperature=1, prompt_template=DEFAULT_PROMPT):
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
        from langfuse.callback import CallbackHandler

        handler = CallbackHandler()
        return chain.invoke(question, config={"callbacks": [handler]})
    return chain.invoke(question)
