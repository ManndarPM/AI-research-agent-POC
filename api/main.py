from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict

app = FastAPI(title="AI Research Assistant API")

# Request model
class AskRequest(BaseModel):
    query: str

# Response model
class Citation(BaseModel):
    title: str
    url: str

class AskResponse(BaseModel):
    answer: str
    citations: List[Citation]

@app.get("/")
def root():
    return {"message": "AI Research Assistant API running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """
    For now this is a placeholder.
    Later we'll plug in search + LLM here.
    """
    fake_answer = f"You asked: '{request.query}'. Here is a placeholder answer."
    fake_citations = [
        {"title": "Example Source", "url": "https://example.com"},
        {"title": "Another Source", "url": "https://wikipedia.org"},
    ]
    return {"answer": fake_answer, "citations": fake_citations}
