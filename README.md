# GURU AI

GURU AI is a simple PDF Question Answering application built using Streamlit, LangChain, OpenAI, and ChromaDB.

Users can upload a PDF and ask questions based on the document content.

---

# Features

- Upload PDF
- Ask questions from PDF
- Modern Streamlit UI
- Chroma Vector Database
- OpenAI Embeddings
- Chat History
- Runnable LangChain Pipeline

---

# Tech Stack

- Python
- Streamlit
- LangChain
- OpenAI
- ChromaDB

---

# Installation

## Create virtual environment

```bash
python -m venv venv
```

## Activate virtual environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

---

# Install dependencies

```bash
pip install -r requirements.txt
```

---

# Add API Key

Create a `.env` file and add:

```env
OPENAI_API_KEY=your_api_key
```

---

# Run the project

```bash
streamlit run app.py
```

---

# How it works

1. Upload PDF
2. PDF gets split into chunks
3. Embeddings are created
4. Data is stored in ChromaDB
5. Relevant chunks are retrieved
6. AI generates answer from document context

---

# Vector Database

Every uploaded PDF creates its own vector database folder automatically.

Example:

```bash
vector_db/python
vector_db/deeplearning
```

---

# Author

Himanshu Namdeo