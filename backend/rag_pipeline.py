from backend.guardrails import guardrail_check
from backend.retriever import hybrid_retrieve, format_docs
from backend.memory import get_history
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a smart AI assistant.
Use only provided context.
"""),
    ("human", """
History:
{history}

Context:
{context}

Question:
{question}
""")
])

chain = prompt | model | StrOutputParser()


def get_answer(question):
    if not guardrail_check(question):
        return "Blocked due to unsafe query"

    docs = hybrid_retrieve(question)
    context = format_docs(docs)

    history = get_history()

    return chain.invoke({
        "question": question,
        "context": context,
        "history": history
    })