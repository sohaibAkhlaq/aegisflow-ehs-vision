"""Retrieval-augmented generation over the compliance policy.

Two retrieval backends, chosen automatically:

* **ChromaDB + sentence-transformers** (preferred) - real vector embeddings, persisted
  to ``artifacts/chroma_db/``.
* **Pure-Python TF-IDF fallback** (zero extra dependencies) - used automatically if
  ``chromadb`` or ``sentence-transformers`` is not installed, or if the embedding
  model cannot be downloaded (e.g. no access to huggingface.co from this network).

This mirrors the core pipeline's own "offline must always work" rule (see
``CLAUDE.md``): the copilot's RAG answer quality improves with the real embedding
stack installed, but a missing optional dependency must never make the feature
simply not work.

The corpus is built from two sources, not just raw PDF text:

1. ``artifacts/policy/rules.json`` - already-parsed, already-validated rule records
   (produced by ``aegisflow.policy``). These are the highest-quality chunks: each one
   is guaranteed to be a faithful, section-tagged excerpt of the real PDF.
2. The raw PDF text, paragraph-chunked, as broader context for questions that fall
   outside the four core rules (e.g. "what camera infrastructure does the policy
   describe?").
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from aegisflow.copilot import CopilotDependencyError


@dataclass
class RetrievedChunk:
    text: str
    section_ref: str
    score: float
    source: str  # "rule" | "pdf_paragraph"


def _load_rule_chunks(rules_path: Path) -> list[tuple[str, str]]:
    """Turn each parsed rule into one rich text chunk, tagged with its section ref."""
    if not rules_path.exists():
        return []
    data = json.loads(rules_path.read_text(encoding="utf-8"))
    chunks: list[tuple[str, str]] = []
    for rule in data.get("rules", []):
        text = (
            f"Behavior: {rule.get('behavior_class', '')}. "
            f"Domain: {rule.get('domain', '')}. "
            f"Unsafe: {rule.get('unsafe_description', '')} "
            f"Safe: {rule.get('safe_description', '')} "
            f"Callout ({rule.get('callout', '')}): {rule.get('callout_text', '')}"
        )
        chunks.append((text, rule.get("section_ref", "Unknown section")))
    return chunks


def _load_pdf_paragraph_chunks(pdf_path: Path) -> list[tuple[str, str]]:
    """Paragraph-chunk the raw PDF as broader-context fallback material.

    Lazily imports PyMuPDF (already a core dependency of the main pipeline's policy
    parser, so this never needs a new install).
    """
    if not pdf_path.exists():
        return []
    try:
        import fitz  # PyMuPDF - already required by aegisflow.policy.extract
    except ImportError as exc:  # pragma: no cover - PyMuPDF is a core dependency
        raise CopilotDependencyError("PDF paragraph indexing", "pymupdf") from exc

    chunks: list[tuple[str, str]] = []
    doc = fitz.open(pdf_path)
    section_pattern = re.compile(r"^(\d+(?:\.\d+)*)\s")
    current_section = "Unknown section"
    for page in doc:
        for para in page.get_text().split("\n\n"):
            para = para.strip()
            if not para or len(para) < 40:
                continue
            match = section_pattern.match(para)
            if match:
                current_section = f"Section {match.group(1)}"
            chunks.append((para, current_section))
    return chunks


# --------------------------------------------------------------------------- #
# Pure-Python fallback: TF-IDF + cosine similarity, no extra dependency at all
# --------------------------------------------------------------------------- #


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class _TfidfIndex:
    """Minimal TF-IDF index. Good enough for a few dozen policy chunks; not intended
    to scale beyond this project's corpus size."""

    def __init__(self, documents: list[str]) -> None:
        self.documents = documents
        self.doc_tokens = [_tokenize(doc) for doc in documents]
        self.doc_freq: Counter[str] = Counter()
        for tokens in self.doc_tokens:
            self.doc_freq.update(set(tokens))
        self.n_docs = len(documents)

    def _vector(self, tokens: list[str]) -> dict[str, float]:
        term_freq = Counter(tokens)
        vector: dict[str, float] = {}
        for term, tf in term_freq.items():
            df = self.doc_freq.get(term, 0)
            idf = math.log((self.n_docs + 1) / (df + 1)) + 1.0
            vector[term] = tf * idf
        return vector

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        shared = set(a) & set(b)
        numerator = sum(a[t] * b[t] for t in shared)
        norm_a = math.sqrt(sum(v * v for v in a.values())) or 1e-9
        norm_b = math.sqrt(sum(v * v for v in b.values())) or 1e-9
        return numerator / (norm_a * norm_b)

    def query(self, text: str, k: int) -> list[tuple[int, float]]:
        query_vector = self._vector(_tokenize(text))
        scored = [
            (i, self._cosine(query_vector, self._vector(tokens)))
            for i, tokens in enumerate(self.doc_tokens)
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]


class PolicyRAG:
    """Retrieval over the compliance policy. Call :meth:`build_index` once, then
    :meth:`query` as many times as needed."""

    def __init__(
        self,
        pdf_path: Path,
        rules_path: Path,
        persist_dir: Path,
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.pdf_path = pdf_path
        self.rules_path = rules_path
        self.persist_dir = persist_dir
        self.embedding_model = embedding_model
        self._backend: str | None = None
        self._chunks: list[tuple[str, str, str]] = []  # (text, section_ref, source)
        self._tfidf: _TfidfIndex | None = None
        self._chroma_collection = None

    def build_index(self) -> str:
        """Build (or rebuild) the retrieval index. Returns which backend was used:
        ``"chromadb"`` or ``"tfidf_fallback"``."""
        rule_chunks = [(t, s, "rule") for t, s in _load_rule_chunks(self.rules_path)]
        pdf_chunks = [(t, s, "pdf_paragraph") for t, s in _load_pdf_paragraph_chunks(self.pdf_path)]
        self._chunks = rule_chunks + pdf_chunks

        try:
            self._build_chroma_index()
            self._backend = "chromadb"
        except (ImportError, CopilotDependencyError, OSError):
            # OSError covers "couldn't download the embedding model" - no network
            # access to huggingface.co, or offline machine. Fall back silently.
            self._tfidf = _TfidfIndex([c[0] for c in self._chunks])
            self._backend = "tfidf_fallback"
        return self._backend

    def _build_chroma_index(self) -> None:
        import chromadb
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(self.embedding_model)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(self.persist_dir))
        collection = client.get_or_create_collection("policy_sections")
        # Rebuild fresh each time - the policy is small, re-embedding is cheap, and it
        # avoids a stale-index bug if the PDF changed since the last build.
        existing_ids = collection.get()["ids"]
        if existing_ids:
            collection.delete(ids=existing_ids)

        texts = [c[0] for c in self._chunks]
        embeddings = model.encode(texts, show_progress_bar=False).tolist()
        collection.add(
            ids=[str(i) for i in range(len(texts))],
            embeddings=embeddings,
            documents=texts,
            metadatas=[{"section_ref": c[1], "source": c[2]} for c in self._chunks],
        )
        self._chroma_collection = collection
        self._chroma_model = model

    def query(self, question: str, k: int = 3) -> list[RetrievedChunk]:
        if self._backend is None:
            self.build_index()

        if self._backend == "chromadb":
            query_embedding = self._chroma_model.encode([question]).tolist()
            result = self._chroma_collection.query(query_embeddings=query_embedding, n_results=k)
            out: list[RetrievedChunk] = []
            for doc, meta, dist in zip(
                result["documents"][0], result["metadatas"][0], result["distances"][0]
            ):
                out.append(
                    RetrievedChunk(
                        text=doc,
                        section_ref=meta["section_ref"],
                        score=1.0 - dist,  # cosine distance -> similarity
                        source=meta["source"],
                    )
                )
            return out

        assert self._tfidf is not None
        top = self._tfidf.query(question, k)
        return [
            RetrievedChunk(
                text=self._chunks[i][0],
                section_ref=self._chunks[i][1],
                score=score,
                source=self._chunks[i][2],
            )
            for i, score in top
            if score > 0
        ]
