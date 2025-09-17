from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List
from duckduckgo_search import DDGS
from dotenv import load_dotenv
import google.generativeai as genai
import os

from api.services.doc_service import index_document, retrieve_similar

# Setup
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
os.makedirs("data/uploads", exist_ok=True)

app = FastAPI(title="AI Research Assistant API")

# Models
class AskRequest(BaseModel):
    query: str

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


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    filename = file.filename
    dest_path = os.path.join("data", "uploads", filename)
    with open(dest_path, "wb") as f:
        f.write(await file.read())
    added = index_document(dest_path, filename)
    return {"doc_id": filename, "indexed_chunks": len(added)}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    query = request.query

    # 1. Try uploaded docs
    doc_results = retrieve_similar(query, k=3)
    if doc_results:
        snippets = []
        for r in doc_results:
            preview = r["text"][:800].replace("\n", " ")
            snippets.append(f"Document [{r['doc_id']}]: {preview}")
        context_text = "\n".join(snippets)
        results = [{"title": f"Uploaded: {r['doc_id']}", "url": f"/data/uploads/{r['doc_id']}"} for r in doc_results]
    else:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=3):
                results.append({"title": r["title"], "url": r["href"]})
        snippets = [f"{r['title']} — {r['url']}" for r in results]
        context_text = "\n".join(snippets)

    # 2. Prompt
    if context_text.strip():
        prompt = f"""
        You are a research assistant.
        Question: {query}
        Sources:
        {context_text}

        Write a concise answer (3-4 sentences) using the sources above.
        Don't invent info. If unsure, say so.
        """
    else:
        prompt = f"""
        You are a research assistant.
        Question: {query}

        Provide a concise answer (3-4 sentences). If unsure, say so.
        """

    # 3. Gemini call
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        if response and hasattr(response, "text") and response.text:
            answer = response.text
        else:
            answer = "No answer returned by Gemini."
    except Exception as e:
        answer = f"Error calling Gemini API: {e}"

    return {"answer": answer, "citations": results}
