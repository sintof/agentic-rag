from pathlib import Path

from langchain_core.documents import Document
from qdrant_client.http.models import FieldCondition, Filter, FilterSelector, MatchValue

from .document_parser import chunk_text, extract_text, get_file_type, is_supported_file
from .vectorstore import get_vectorstore


def _delete_existing(vectorstore, filename: str) -> None:
    """Re-ingesting the same filename would otherwise duplicate its chunks as new
    points on every upload — delete the old ones first so a re-upload replaces
    rather than accumulates."""
    vectorstore.client.delete(
        collection_name=vectorstore.collection_name,
        points_selector=FilterSelector(
            filter=Filter(must=[FieldCondition(key="metadata.source", match=MatchValue(value=filename))])
        ),
    )


def ingest_file(file_path: str, original_filename: str) -> dict:
    if not is_supported_file(original_filename):
        raise ValueError(f"Unsupported file type: {original_filename}")

    file_type = get_file_type(original_filename)
    text = extract_text(file_path, file_type)
    chunks = chunk_text(text)

    if not chunks:
        raise ValueError("No extractable text found in this document.")

    documents = [
        Document(
            page_content=chunk["text"],
            metadata={
                "source": original_filename,
                "chunk_id": chunk["chunk_id"],
                "file_type": file_type,
            },
        )
        for chunk in chunks
    ]

    vectorstore = get_vectorstore()
    _delete_existing(vectorstore, original_filename)
    ids = vectorstore.add_documents(documents)

    return {
        "filename": original_filename,
        "chunks_indexed": len(documents),
        "ids": ids,
    }


def ingest_upload(upload_dir: str, filename: str, content: bytes) -> dict:
    Path(upload_dir).mkdir(parents=True, exist_ok=True)
    dest = Path(upload_dir) / filename
    dest.write_bytes(content)
    return ingest_file(str(dest), filename)
