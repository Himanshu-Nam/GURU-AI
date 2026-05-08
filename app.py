import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import tempfile
import os
import time

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="GURU AI",
    page_icon="🤖",
    layout="wide"
)

# Modern custom UI
st.markdown("""
<style>

.main {
    background-color: #0f172a;
    color: white;
}

.stApp {
    background: linear-gradient(135deg, #0f172a, #111827);
}

.title {
    font-size: 48px;
    font-weight: bold;
    text-align: center;
    margin-top: 10px;
    color: white;
    animation: fadeIn 1.5s ease-in;
}

.subtitle {
    text-align: center;
    color: #cbd5e1;
    margin-bottom: 30px;
}

.chat-box {
    padding: 15px;
    border-radius: 15px;
    margin-bottom: 12px;
    animation: slideUp 0.4s ease;
}

.user-box {
    background-color: #2563eb;
    color: white;
}

.ai-box {
    background-color: #1e293b;
    color: white;
}

.upload-box {
    padding: 20px;
    border-radius: 15px;
    background-color: #1e293b;
    margin-bottom: 20px;
}

.stTextInput input {
    border-radius: 12px;
}

.stButton button {
    width: 100%;
    border-radius: 12px;
    height: 45px;
    background-color: #2563eb;
    color: white;
    font-weight: bold;
    border: none;
}

.stButton button:hover {
    background-color: #1d4ed8;
}

@keyframes fadeIn {
    from {
        opacity: 0;
    }
    to {
        opacity: 1;
    }
}

@keyframes slideUp {
    from {
        transform: translateY(20px);
        opacity: 0;
    }
    to {
        transform: translateY(0px);
        opacity: 1;
    }
}

</style>
""", unsafe_allow_html=True)

# App title
st.markdown('<div class="title">GURU AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Smart PDF Question Answering Assistant</div>',
    unsafe_allow_html=True
)

# Store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Embedding model
embedding_model = OpenAIEmbeddings()

# LLM model
model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,
    max_completion_tokens=300
)

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an intelligent AI assistant.

Use ONLY the provided context to answer the user question.

If the answer is not available in the context,
say:
"I could not find the answer in the document."
"""
    ),
    (
        "human",
        """
Context:
{context}

Question:
{question}
"""
    )
])

# Sidebar
with st.sidebar:

    st.markdown("## Upload PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf"
    )

    st.markdown("---")
    st.markdown("### Features")
    st.markdown("""
    - PDF Question Answering
    - Chat History
    - LangChain Pipeline
    - Runnable Architecture
    """)

# Process uploaded PDF
if uploaded_file:

    with st.spinner("Processing PDF..."):

        # Save uploaded PDF temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            pdf_path = tmp_file.name

        # Load PDF
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        # Split documents into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        split_docs = splitter.split_documents(docs)

        # Create vector database
        vectordb = Chroma.from_documents(
            documents=split_docs,
            embedding=embedding_model
        )

        # Create retriever
        retriever = vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 4,
                "fetch_k": 10,
                "lambda_mult": 0.5
            }
        )

        # Function to format retrieved docs
        def format_docs(docs):
            return "\n\n".join(
                doc.page_content for doc in docs
            )

        # Runnable pipeline
        rag_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | model
            | StrOutputParser()
        )

    st.success("PDF processed successfully")

    # Display chat history
    for msg in st.session_state.messages:

        if msg["role"] == "user":
            st.markdown(
                f"""
                <div class="chat-box user-box">
                <b>You:</b><br>{msg["content"]}
                </div>
                """,
                unsafe_allow_html=True
            )

        else:
            st.markdown(
                f"""
                <div class="chat-box ai-box">
                <b>GURU AI:</b><br>{msg["content"]}
                </div>
                """,
                unsafe_allow_html=True
            )

    # User question input
    question = st.chat_input("Ask anything from your PDF...")

    # Generate answer
    if question:

        # Store user message
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        # Show user message instantly
        st.markdown(
            f"""
            <div class="chat-box user-box">
            <b>You:</b><br>{question}
            </div>
            """,
            unsafe_allow_html=True
        )

        # AI typing animation
        with st.spinner("GURU AI is thinking..."):

            response = rag_chain.invoke(question)

            time.sleep(1)

        # Store AI response
        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })

        # Display AI response
        st.markdown(
            f"""
            <div class="chat-box ai-box">
            <b>GURU AI:</b><br>{response}
            </div>
            """,
            unsafe_allow_html=True
        )

else:

    st.info("Upload a PDF file to start chatting with GURU AI")