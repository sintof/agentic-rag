from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .config import settings
from .document_parser import is_supported_file
from .graph.graph import run as run_graph
from .ingest import ingest_upload

app = FastAPI(title="Agentic RAG Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
async def ingest(file: UploadFile):
    if not file.filename or not is_supported_file(file.filename):
        raise HTTPException(status_code=400, detail="Unsupported file type. Use PDF, DOCX, PPTX, or TXT.")

    content = await file.read()
    try:
        result = ingest_upload(settings.file_storage_path, file.filename, content)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return result


class ChatIn(BaseModel):
    question: str


@app.post("/chat")
def chat(body: ChatIn):
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    result = run_graph(body.question)
    return {
        "answer": result["generation"],
        "steps": result["steps"],
        "sources": result["sources"],
        "web_search_used": result["web_search_used"],
        "retries": result["retries"],
    }
