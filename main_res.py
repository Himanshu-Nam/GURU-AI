import streamlit as st
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from dotenv import load_dotenv
import time
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma


load_dotenv()
#embedding model
embedding_model = OpenAIEmbeddings()

#prompt
template = ChatPromptTemplate.from_messages(
    [
        ('system',"""You are an intelligent helpful Ai assistance
        "USE ONLY the provided context to answer the questions.
        if the answer is not present in the context,
        say: "I could not find the answer in the document."
        """),
        ('human',""" 
        Context:{context}
        Question:{question}""")
    ]
)
#load vector store
vectordb = Chroma(
    
    persist_directory="chroma_db",
    embedding_function = embedding_model
)
model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,
    max_completion_tokens=300
)

#retriver
retriever = vectordb.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)
print("Rag system created")

print("press Q to exit")

while True:
    qurey=input("You: ")
    if qurey=="Q":
        break
    docs = retriever.invoke(qurey)
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )
    final_prompt = template.invoke({
        "context":context,
        "question":qurey
    })
    res = model.invoke(final_prompt)

    print(f"\nAI: {res.content}")