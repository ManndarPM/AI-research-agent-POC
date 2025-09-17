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
    doc_results = retrieve_similar(query, k=10)
    if doc_results:
            # Step 2: Ask Gemini to rerank them
            snippets = []
            for i, c in enumerate(doc_results):
                clean_text = c["text"][:500].replace("\n", " ")
                snippets.append(f"[{i}] {clean_text}")

            rerank_prompt = f"""
            You are helping rank document passages.
            Query: {query}

            Passages:
            {chr(10).join(snippets)}

            Task: Pick the 3 passages that are most relevant to the query.
            Reply ONLY with their indices (comma-separated).
            """

            try:
                model = genai.GenerativeModel("gemini-1.5-flash")
                rerank_response = model.generate_content(rerank_prompt)
                if rerank_response and hasattr(rerank_response, "text") and rerank_response.text:
                    indices = [int(x) for x in rerank_response.text.split(",") if x.strip().isdigit()]
                    indices = [i for i in indices if 0 <= i < len(doc_results)]
                else:
                    indices = list(range(min(3, len(doc_results))))
            except Exception:
                indices = list(range(min(3, len(doc_results))))

            # Step 3: Build context from reranked chunks
            chosen = [doc_results[i] for i in indices[:3]]
            context_text = "\n".join([c["text"][:800].replace("\n", " ") for c in chosen])
            results = [{"title": f"Uploaded: {c['doc_id']}", "url": f"/data/uploads/{c['doc_id']}"} for c in chosen]
    else:
        # Fallback to web search
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

        Write a concise answer (3-6 sentences) using ONLY the sources above.
        If the sources do not contain the answer, say so.
        """
    else:
        prompt = f"Answer the question: {query}"

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        answer = response.text if response and hasattr(response, "text") else "No answer returned by Gemini."
    except Exception as e:
        answer = f"Error calling Gemini API: {e}"

    return {"answer": answer, "citations": results}