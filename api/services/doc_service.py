# api/services/doc_service.py
import os
import pickle
from typing import List, Dict
import fitz  # PyMuPDF
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = "data"
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
INDEX_PATH = os.path.join(DATA_DIR, "index.pkl")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Try to load sentence-transformers; fall back to TF-IDF
USE_ST = False
try:
    from sentence_transformers import SentenceTransformer
    ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    USE_ST = True
except Exception:
    ST_MODEL = None
    from sklearn.feature_extraction.text import TfidfVectorizer

# INDEX structure (keeps in-memory + persisted to disk)
# INDEX = {"chunks": [ {"doc_id":..., "text":...}, ... ],
#          "embeddings": numpy.ndarray or None,
#          "tfidf": sklearn sparse matrix or None,
#          "tfidf_vectorizer": TfidfVectorizer or None }
if os.path.exists(INDEX_PATH):
    with open(INDEX_PATH, "rb") as f:
        INDEX = pickle.load(f)
else:
    INDEX = {"chunks": [], "embeddings": None, "tfidf": None, "tfidf_vectorizer": None}

def save_index():
    with open(INDEX_PATH, "wb") as f:
        pickle.dump(INDEX, f)

def extract_text_from_pdf(path: str) -> str:
    doc = fitz.open(path)
    pages = []
    for page in doc:
        pages.append(page.get_text())
    return "\n".join(pages)

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    text = text.replace("\r\n", "\n")
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks

def index_document(file_path: str, doc_id: str) -> List[Dict]:
    # parse file (PDF or txt)
    if file_path.lower().endswith(".pdf"):
        text = extract_text_from_pdf(file_path)
    else:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

    chunks = chunk_text(text)
    added = []
    for c in chunks:
        INDEX["chunks"].append({"doc_id": doc_id, "text": c})
        added.append({"doc_id": doc_id, "text": c})

    # compute embeddings or tfidf for the whole corpus
    texts = [c["text"] for c in INDEX["chunks"]]
    if USE_ST:
        embeddings = ST_MODEL.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        INDEX["embeddings"] = embeddings  # numpy array shape (N, D)
        INDEX["tfidf"] = None
        INDEX["tfidf_vectorizer"] = None
    else:
        vectorizer = TfidfVectorizer().fit(texts)
        INDEX["tfidf"] = vectorizer.transform(texts)  # sparse matrix
        INDEX["tfidf_vectorizer"] = vectorizer
        INDEX["embeddings"] = None

    save_index()
    return added

def retrieve_similar(query: str, k: int = 10) -> List[Dict]:
    if len(INDEX["chunks"]) == 0:
        return []

    if USE_ST and INDEX["embeddings"] is not None:
        qv = ST_MODEL.encode([query], convert_to_numpy=True)[0]
        emb = INDEX["embeddings"]
        # cosine similarity
        sims = np.dot(emb, qv) / (np.linalg.norm(emb, axis=1) * (np.linalg.norm(qv) + 1e-10) + 1e-10)
        top_idx = np.argsort(-sims)[:k]
        return [{"doc_id": INDEX["chunks"][i]["doc_id"], "text": INDEX["chunks"][i]["text"], "score": float(sims[i])} for i in top_idx]

    # TF-IDF fallback
    if INDEX["tfidf"] is not None and INDEX["tfidf_vectorizer"] is not None:
        qv = INDEX["tfidf_vectorizer"].transform([query])
        sims = cosine_similarity(qv, INDEX["tfidf"]).flatten()
        top_idx = sims.argsort()[::-1][:k]
        return [
                    {"doc_id": INDEX["chunks"][i]["doc_id"], "text": INDEX["chunks"][i]["text"], "score": float(sims[i])}
                    for i in top_idx
                ]
    return []
