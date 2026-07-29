"""
Qdrant vector store. Runs embedded (on-disk, no server) unless QDRANT_URL is set,
per the class guide: "leave QDRANT_URL blank -> Qdrant runs embedded".
"""
from pathlib import Path

from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from .config import settings
from .llm import get_embeddings

_PERSIST_DIR = Path(__file__).resolve().parent.parent / "qdrant_data"

_client: QdrantClient | None = None
_vectorstore: Qdrant | None = None


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        if settings.qdrant_url:
            _client = QdrantClient(url=settings.qdrant_url)
        else:
            _PERSIST_DIR.mkdir(parents=True, exist_ok=True)
            _client = QdrantClient(path=str(_PERSIST_DIR))
    return _client


def _ensure_collection(client: QdrantClient, embeddings) -> None:
    name = settings.qdrant_collection
    existing = [c.name for c in client.get_collections().collections]
    if name in existing:
        return
    # Embedding dimension varies by model/provider — probe it instead of hardcoding,
    # so the collection always matches whatever embedding model is actually configured.
    probe_vector = embeddings.embed_query("dimension probe")
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=len(probe_vector), distance=Distance.COSINE),
    )


def get_vectorstore() -> Qdrant:
    global _vectorstore
    if _vectorstore is None:
        embeddings = get_embeddings()
        client = _get_client()
        _ensure_collection(client, embeddings)
        _vectorstore = Qdrant(
            client=client,
            collection_name=settings.qdrant_collection,
            embeddings=embeddings,
        )
    return _vectorstore


def get_retriever(k: int = 4):
    return get_vectorstore().as_retriever(search_kwargs={"k": k})
