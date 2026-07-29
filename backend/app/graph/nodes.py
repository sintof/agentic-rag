from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from ..config import settings
from ..llm import get_llm_flash, get_llm_lite
from ..vectorstore import get_retriever
from .state import GraphState

try:
    from langchain_community.tools.tavily_search import TavilySearchResults
except ImportError:  # optional dependency path
    TavilySearchResults = None


# ---------------------------------------------------------------------------
# Structured-output graders — the "decisions" that make this agentic, not a
# straight line. Each is a single yes/no call to the flash (critic) model.
# ---------------------------------------------------------------------------

class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Is the document relevant to the question? 'yes' or 'no'.")


class GradeHallucination(BaseModel):
    binary_score: str = Field(description="Is the answer grounded in / supported by the given facts? 'yes' or 'no'.")


class GradeAnswer(BaseModel):
    binary_score: str = Field(description="Does the answer actually resolve the question asked? 'yes' or 'no'.")


def _safe_structured_call(llm, schema, messages, default: str) -> str:
    """Structured-output calls can fail if the proxy/model doesn't support tool-calling
    reliably. Fail safe rather than crashing the whole graph run."""
    try:
        grader = llm.with_structured_output(schema)
        result = grader.invoke(messages)
        return (result.binary_score or default).strip().lower()
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def retrieve(state: GraphState) -> dict:
    question = state["question"]
    documents = get_retriever(k=4).invoke(question)
    return {
        "documents": documents,
        "steps": state.get("steps", []) + ["retrieve"],
    }


def grade_documents(state: GraphState) -> dict:
    question = state["question"]
    documents = state["documents"]
    llm = get_llm_flash()

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You grade whether a retrieved document is relevant to a user question. "
         "Give a binary score 'yes' or 'no'. 'yes' means the document contains "
         "information that helps answer the question, even partially."),
        ("human", "Retrieved document:\n\n{document}\n\nUser question: {question}"),
    ])

    filtered: list[Document] = []
    for doc in documents:
        messages = prompt.format_messages(document=doc.page_content, question=question)
        score = _safe_structured_call(llm, GradeDocuments, messages, default="yes")
        if score == "yes":
            filtered.append(doc)

    return {
        "documents": filtered,
        "steps": state.get("steps", []) + ["grade_documents"],
    }


def web_search(state: GraphState) -> dict:
    question = state["question"]
    documents = list(state.get("documents", []))

    if TavilySearchResults is not None and settings.tavily_api_key:
        try:
            tool = TavilySearchResults(k=3, tavily_api_key=settings.tavily_api_key)
            results = tool.invoke({"query": question})
            for item in results:
                documents.append(Document(
                    page_content=item.get("content", ""),
                    metadata={"source": item.get("url", "web"), "chunk_id": "web"},
                ))
        except Exception as exc:
            documents.append(Document(
                page_content=f"(Web search failed: {exc})",
                metadata={"source": "web-error", "chunk_id": "web"},
            ))
    else:
        documents.append(Document(
            page_content="(No web search configured — TAVILY_API_KEY is not set.)",
            metadata={"source": "web-unavailable", "chunk_id": "web"},
        ))

    return {
        "documents": documents,
        "web_search_used": True,
        "steps": state.get("steps", []) + ["web_search"],
    }


def generate(state: GraphState) -> dict:
    question = state["question"]
    documents = state["documents"]
    llm = get_llm_lite()

    context = "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')} / {doc.metadata.get('chunk_id', '')}]\n{doc.page_content}"
        for doc in documents
    ) or "(no context retrieved)"

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a grounded question-answering assistant. Answer the question using "
         "ONLY the provided context. Cite which source(s) you used inline like [source]. "
         "If the context does not contain enough information to answer, say so honestly "
         "instead of guessing."),
        ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"),
    ])

    messages = prompt.format_messages(context=context, question=question)
    response = llm.invoke(messages)

    sources = [
        {"source": doc.metadata.get("source", "unknown"), "chunk_id": doc.metadata.get("chunk_id", "")}
        for doc in documents
    ]

    return {
        "generation": response.content,
        "sources": sources,
        "retries": state.get("retries", 0) + 1,
        "steps": state.get("steps", []) + ["generate"],
    }


# ---------------------------------------------------------------------------
# Conditional edges — the routing decisions
# ---------------------------------------------------------------------------

def route_after_grade(state: GraphState) -> str:
    """No relevant documents survived grading -> fall back to the web."""
    return "web_search" if not state["documents"] else "generate"


def route_after_generate(state: GraphState) -> str:
    """Hallucination + answer-relevance checks. A retry cap guarantees the graph
    always terminates even if the model keeps failing a grade."""
    if state.get("retries", 0) >= settings.max_generation_retries:
        return "useful"

    llm = get_llm_flash()
    documents = state["documents"]
    generation = state["generation"]
    question = state["question"]

    facts = "\n\n".join(doc.page_content for doc in documents) or "(no context)"

    grounded_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You check whether an AI-generated answer is grounded in / supported by a "
         "set of facts. Give a binary score 'yes' or 'no'. 'yes' means every claim in "
         "the answer is supported by the facts."),
        ("human", "Facts:\n{facts}\n\nAnswer:\n{generation}"),
    ])
    grounded_messages = grounded_prompt.format_messages(facts=facts, generation=generation)
    grounded = _safe_structured_call(llm, GradeHallucination, grounded_messages, default="yes")

    if grounded != "yes":
        return "not_grounded"

    relevance_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You check whether an answer actually resolves the question that was asked. "
         "Give a binary score 'yes' or 'no'."),
        ("human", "Question: {question}\n\nAnswer:\n{generation}"),
    ])
    relevance_messages = relevance_prompt.format_messages(question=question, generation=generation)
    relevant = _safe_structured_call(llm, GradeAnswer, relevance_messages, default="yes")

    return "useful" if relevant == "yes" else "not_useful"
