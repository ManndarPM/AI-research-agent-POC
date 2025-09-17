from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from ddgs import DDGS
from dotenv import load_dotenv
import os
import google.generativeai as genai

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


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

    # Step 2: Build context text
    snippets = [f"{r['title']} — {r['url']}" for r in results]
    context_text = "\n".join(snippets)

    # Step 3: Summarize with Gemini
    if context_text.strip():
        prompt = f"""
        You are a research assistant.
        Question: {query}
        Sources:
        {context_text}

        Write a concise answer (3-4 sentences) using the sources above.
        Don't invent information. If unsure, say so.
        """
    else:
        prompt = f"""
        You are a research assistant.
        Question: {query}

        Provide a concise answer (3-4 sentences).
        If unsure, say so.
        """

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)

        # Sometimes Gemini returns chunks or None
        if response and hasattr(response, "text") and response.text:
            answer = response.text
        else:
            answer = "No answer returned by Gemini."
        
    except Exception as e:
        answer = f"Error calling Gemini API: {e}"
    
    return {"answer": answer, "citations": results}
        
