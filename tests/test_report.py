import pytest
from backend.pipelines.report_pipeline import (
    build_citations,
    format_citations_block,
    assemble_report,
)


# ── build_citations ───────────────────────────────────────────────────────────

def test_build_citations_deduplicates_sources():
    """Multiple chunks from same source produce one citation."""
    metadatas = [
        {"source": "paper.pdf", "page_number": 1, "type": "pdf"},
        {"source": "paper.pdf", "page_number": 3, "type": "pdf"},
        {"source": "paper.pdf", "page_number": 5, "type": "pdf"},
    ]
    chunks = ["chunk one", "chunk two", "chunk three"]

    citations = build_citations(metadatas, chunks)

    assert len(citations) == 1
    assert citations[0]["source"] == "paper.pdf"
    assert sorted(citations[0]["pages"]) == [1, 3, 5]


def test_build_citations_multiple_sources():
    metadatas = [
        {"source": "doc_a.pdf", "page_number": 1, "type": "pdf"},
        {"source": "doc_b.txt", "page_number": None, "type": "text"},
    ]
    chunks = ["chunk a", "chunk b"]

    citations = build_citations(metadatas, chunks)

    assert len(citations) == 2
    sources = {c["source"] for c in citations}
    assert "doc_a.pdf" in sources
    assert "doc_b.txt" in sources


def test_build_citations_empty_inputs():
    citations = build_citations([], [])
    assert citations == []


def test_build_citations_assigns_sequential_numbers():
    metadatas = [
        {"source": "a.pdf", "type": "pdf"},
        {"source": "b.pdf", "type": "pdf"},
        {"source": "c.pdf", "type": "pdf"},
    ]
    chunks = ["x", "y", "z"]
    citations = build_citations(metadatas, chunks)
    numbers = [c["citation_number"] for c in citations]
    assert numbers == [1, 2, 3]


# ── format_citations_block ────────────────────────────────────────────────────

def test_format_citations_block_empty():
    result = format_citations_block([])
    assert "No source documents" in result


def test_format_citations_block_renders_markdown():
    citations = [
        {
            "citation_number": 1,
            "source": "test.pdf",
            "type": "pdf",
            "pages": [2, 4],
            "excerpt": "This is an excerpt from the document",
        }
    ]
    result = format_citations_block(citations)
    assert "[1]" in result
    assert "test.pdf" in result
    assert "excerpt" in result.lower()


# ── assemble_report ───────────────────────────────────────────────────────────

def test_assemble_report_contains_required_sections():
    report = assemble_report(
        query="What is RAG?",
        plan=[{"step": 1, "task": "Define RAG", "reason": "Foundation"}],
        summary="RAG combines retrieval with generation.",
        citations=[],
        evaluation={
            "confidence_score": 0.75,
            "passed": True,
            "hallucination_flags": [],
            "reasoning": "Well grounded.",
            "retry_count": 0,
        },
        model="llama3",
        report_id="abc12345",
    )

    assert "# Research Report" in report
    assert "What is RAG?" in report
    assert "Executive Summary" in report
    assert "Research Plan" in report
    assert "Sources & Citations" in report
    assert "Evaluation Metadata" in report
    assert "abc12345" in report


def test_assemble_report_includes_plan_steps():
    report = assemble_report(
        query="Test query",
        plan=[
            {"step": 1, "task": "Step one task", "reason": "Reason one"},
            {"step": 2, "task": "Step two task", "reason": "Reason two"},
        ],
        summary="Test summary.",
        citations=[],
        evaluation={
            "confidence_score": 0.8,
            "passed": True,
            "hallucination_flags": [],
            "reasoning": "OK.",
            "retry_count": 0,
        },
        model="llama3",
        report_id="test001",
    )

    assert "Step one task" in report
    assert "Step two task" in report


def test_assemble_report_flags_below_threshold():
    report = assemble_report(
        query="Test",
        plan=[],
        summary="Summary.",
        citations=[],
        evaluation={
            "confidence_score": 0.3,
            "passed": False,
            "hallucination_flags": ["Unsupported claim X"],
            "reasoning": "Low confidence.",
            "retry_count": 2,
        },
        model="llama3",
        report_id="low001",
    )

    assert "Below Threshold" in report
    assert "Unsupported claim X" in report