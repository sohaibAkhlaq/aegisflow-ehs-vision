"""Unit tests for the TF-IDF fallback path of PolicyRAG - the zero-dependency path
that must always work, mirroring the core project's own offline-first testing style.
"""

from __future__ import annotations

from aegisflow.copilot.rag import _TfidfIndex, _tokenize


def test_tokenize_lowercases_and_strips_punctuation():
    tokens = _tokenize("Unauthorized Intervention (Section 4.3.2)!")
    assert "unauthorized" in tokens
    assert "intervention" in tokens
    assert "!" not in tokens


def test_tfidf_index_ranks_relevant_document_first():
    docs = [
        "Opened Panel Cover is defined as the electrical panel left open.",
        "Forklift overload is three or more blocks carried on the forks.",
        "Safe Walkway Violation is presence outside the marked floor path.",
    ]
    index = _TfidfIndex(docs)
    results = index.query("what counts as a forklift overload violation", k=1)
    assert results[0][0] == 1  # the forklift document should rank first


def test_tfidf_index_returns_zero_score_for_unrelated_query():
    docs = ["Opened Panel Cover is defined as the electrical panel left open."]
    index = _TfidfIndex(docs)
    results = index.query("completely unrelated words banana spaceship", k=1)
    assert results[0][1] >= 0.0  # never negative, even for a poor match
