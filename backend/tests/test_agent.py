"""
Routing tests for the agentic RAG graph: happy path, web fallback, self-correction.
Requires GEMINI_API_KEY in backend/.env (real proxy calls — no mocking, since the
whole point is to prove the actual routing decisions the live model makes).

Run from backend/: pytest tests/test_agent.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from app.graph.graph import run  # noqa: E402
from app.ingest import ingest_file  # noqa: E402

SAMPLE_DIR = Path(__file__).resolve().parent.parent.parent / "sample_docs"


def setup_module():
    ingest_file(str(SAMPLE_DIR / "photosynthesis.txt"), "photosynthesis.txt")
    ingest_file(str(SAMPLE_DIR / "water_cycle.pdf"), "water_cycle.pdf")


def test_happy_path_answers_from_documents():
    result = run("What limits the rate of photosynthesis at low light levels?")
    assert "retrieve" in result["steps"]
    assert "generate" in result["steps"]
    assert result["web_search_used"] is False
    assert "light intensity" in result["generation"].lower()
    assert result["sources"], "expected at least one cited source"


def test_multimodal_image_fact_is_retrievable():
    result = run("How many days does the condensation stage of the water cycle average?")
    assert "2" in result["generation"]
    assert any(s["source"] == "water_cycle.pdf" for s in result["sources"])


def test_web_fallback_triggers_on_out_of_document_question():
    result = run("What is the capital of France?")
    assert "web_search" in result["steps"]
    assert result["web_search_used"] is True


def test_self_correction_loop_terminates():
    """An out-of-scope question with no web key configured will keep failing the
    answer-relevance grade — this proves the retry cap actually stops the loop
    instead of looping forever."""
    result = run("What is the capital of France and what year was the Eiffel Tower built?")
    assert result["retries"] <= 3  # MAX_GENERATION_RETRIES(2) + 1 generate calls, hard cap
    assert result["generation"]  # still produced *some* answer, didn't crash
