from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
load_dotenv()

docs = [
    Document(
        page_content="Python is a powerful and easy-to-learn programming language used worldwide.",
        metadata={
            "topic": "Python",
            "level": "Beginner"
        }
    ),

    Document(
        page_content="Artificial Intelligence (AI) helps machines think and make decisions like humans.",
        metadata={
            "topic": "AI",
            "category": "Technology"
        }
    ),

    Document(
        page_content="Python is widely used in AI development because of its simple syntax and strong libraries.",
        metadata={
            "topic": "Python + AI",
            "difficulty": "Easy"
        }
    )
]
embedding_model = OpenAIEmbeddings()
vector_db = Chroma.from_documents(
    documents = docs,
    embedding = embedding_model,
    persist_directory="chroma-db"
)
res = vector_db.similarity_search("What is AI",k=2)
for r in res:
    print(r) 

retriver = vector_db.as_retriever()

res1 = retriver.invoke("Explain python")

for r2 in res1:
    print(r2.page_content)