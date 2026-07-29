from langgraph.graph import END, StateGraph

from . import nodes
from .state import GraphState

_compiled = None


def build_graph():
    g = StateGraph(GraphState)

    g.add_node("retrieve", nodes.retrieve)
    g.add_node("grade_documents", nodes.grade_documents)
    g.add_node("web_search", nodes.web_search)
    g.add_node("generate", nodes.generate)

    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "grade_documents")

    g.add_conditional_edges(
        "grade_documents",
        nodes.route_after_grade,
        {"web_search": "web_search", "generate": "generate"},
    )
    g.add_edge("web_search", "generate")

    g.add_conditional_edges(
        "generate",
        nodes.route_after_generate,
        {"useful": END, "not_grounded": "generate", "not_useful": "web_search"},
    )

    return g.compile()


def get_graph():
    global _compiled
    if _compiled is None:
        _compiled = build_graph()
    return _compiled


def run(question: str) -> dict:
    graph = get_graph()
    initial_state: GraphState = {
        "question": question,
        "documents": [],
        "generation": "",
        "steps": [],
        "retries": 0,
        "web_search_used": False,
        "sources": [],
    }
    return graph.invoke(initial_state)
