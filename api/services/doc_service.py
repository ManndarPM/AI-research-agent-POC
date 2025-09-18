# api/services/doc_service.py
import os
import pickle
from typing import List, Dict
import fitz  # PyMuPDF
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import faiss 

DATA_DIR = "data"
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
INDEX_PATH = os.path.join(DATA_DIR, "faiss.index")
META_PATH = os.path.join(DATA_DIR, "metadata.pkl")

os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load embedding model
ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# Global FAISS index + metadata
embedding_dim = ST_MODEL.get_sentence_embedding_dimension()
if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        METADATA = pickle.load(f)
else:
    index = faiss.IndexFlatL2(embedding_dim)  # L2 distance
    METADATA = []  # each entry: {"doc_id": ..., "text": ...}

def save_index():
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(METADATA, f)

def extract_text_from_pdf(path: str) -> str:
    doc = fitz.open(path)
    pages = [page.get_text() for page in doc]
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

    
    # Chunk + embed
    chunks = chunk_text(text)
    embeddings = ST_MODEL.encode(chunks, convert_to_numpy=True)
    
    # Add to FAISS
    index.add(embeddings)
    for c in chunks:
        METADATA.append({"doc_id": doc_id, "text": c})

    save_index()

    return [{"doc_id": doc_id, "text": c} for c in chunks]

def retrieve_similar(query: str, k: int = 10) -> List[Dict]:
    if len(METADATA) == 0:
        return []

    qv = ST_MODEL.encode([query], convert_to_numpy=True)
    D, I = index.search(qv, k)

    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < len(METADATA):
            results.append({
                "doc_id": METADATA[idx]["doc_id"],
                "text": METADATA[idx]["text"],
                "score": float(score)  # smaller = better in L2
            })
    return results
