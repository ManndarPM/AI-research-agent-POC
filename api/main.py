from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from duckduckgo_search import DDGS

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
    Step 3: Perform real web search with DuckDuckGo.
    Later we’ll add summarization (LLM).
    """
    query = request.query
    results = []

    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=3):
            results.append({"title": r["title"], "url": r["href"]})

    # For now, the answer is just a placeholder
    answer = f"Top {len(results)} search results for '{query}':"

    return {"answer": answer, "citations": results}
