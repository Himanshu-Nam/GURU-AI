from fastapi import FastAPI
from pydantic import BaseModel
from backend.rag_pipeline import get_answer

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

@app.post("/chat")
def chat(req: QueryRequest):
    response = get_answer(req.question)
    return {"answer": response}