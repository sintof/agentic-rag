from typing import TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict):
    question: str
    documents: list[Document]
    generation: str
    steps: list[str]
    retries: int
    web_search_used: bool
    sources: list[dict]
