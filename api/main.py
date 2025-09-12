from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="AI Research Assistant API (Demo)")

@app.get("/")
def root():
    return {"message": "AI Research Assistant API running"}

@app.get("/health")
def health():
    return {"status": "ok"}
