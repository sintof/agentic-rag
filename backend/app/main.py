from pathlib import Path

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


async def _read_limited(file: UploadFile, limit: int) -> bytes:
    """Reads in chunks and aborts as soon as the real byte count exceeds the limit —
    doesn't trust the client-supplied Content-Length header, and never buffers more
    than one chunk past the limit regardless of how large the upload claims to be."""
    chunks: list[bytes] = []
    total = 0
    chunk_size = 1024 * 1024
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise HTTPException(
                status_code=413,
                detail=f"File too large — max {limit // (1024 * 1024)}MB.",
            )
        chunks.append(chunk)
    return b"".join(chunks)


@app.post("/ingest")
async def ingest(file: UploadFile):
    # Sanitize before any use: Path(...).name strips directory components, so a
    # filename like "../../app/main.py" can't be used to write outside uploads/.
    safe_filename = Path(file.filename or "").name
    if not safe_filename or not is_supported_file(safe_filename):
        raise HTTPException(status_code=400, detail="Unsupported file type. Use PDF, DOCX, PPTX, or TXT.")

    content = await _read_limited(file, settings.max_upload_bytes)
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        result = ingest_upload(settings.file_storage_path, safe_filename, content)
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
