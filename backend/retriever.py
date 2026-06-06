from rank_bm25 import BM25Okapi
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

vectordb = None
bm25 = None
docs = None

def init_retriever(split_docs):
    global vectordb, bm25, docs

    docs = split_docs

    embedding = OpenAIEmbeddings()

    vectordb = Chroma.from_documents(
        documents=split_docs,
        embedding=embedding,
        persist_directory="./chroma_db"
    )

    tokenized = [d.page_content.lower().split() for d in split_docs]
    bm25 = BM25Okapi(tokenized)


def hybrid_retrieve(query):
    vector_retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10}
    )

    vector_docs = vector_retriever.invoke(query)

    bm25_scores = bm25.get_scores(query.lower().split())
    top_idx = sorted(range(len(bm25_scores)),
                     key=lambda i: bm25_scores[i],
                     reverse=True)[:4]

    bm25_docs = [docs[i] for i in top_idx]

    # merge
    seen = set()
    final = []

    for d in vector_docs + bm25_docs:
        if d.page_content not in seen:
            final.append(d)
            seen.add(d.page_content)

    return final[:6]