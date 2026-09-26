"""LangChain/LCEL wrapper around Groq for the copilot's answer generation.

Uses ``langchain-groq``'s ``ChatGroq`` and LCEL's ``prompt | model | parser`` composition
style, kept in its own module and never imported by the core pipeline
(``aegisflow.llm.groq_provider`` keeps calling the raw Groq SDK directly - that module
is graded, deterministic, and stays untouched).

If ``langchain`` / ``langchain-groq`` are not installed, :func:`build_rag_answer_chain`
raises :class:`~aegisflow.copilot.CopilotDependencyError` with an install hint, rather
than failing at import time - so the rest of the copilot (e.g. the TF-IDF RAG fallback)
still works without LangChain installed.
"""

from __future__ import annotations

from typing import Any

from aegisflow.copilot import CopilotDependencyError
from aegisflow.copilot.rag import RetrievedChunk

SYSTEM_PROMPT = """You are the AegisFlow EHS policy assistant. Answer the user's \
question ONLY using the provided policy excerpts below. If the excerpts do not \
contain the answer, say so plainly rather than guessing.

Always cite the section number(s) your answer relies on, in the form "(Section X.X.X)".

Policy excerpts:
{context}
"""


def _format_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(f"[{c.section_ref}] {c.text}" for c in chunks)


def build_rag_answer_chain(model_name: str = "openai/gpt-oss-120b", temperature: float = 0.0) -> Any:
    """Build the LCEL chain: prompt -> ChatGroq -> string parser.

    Returns a Runnable with an ``.invoke({"question": ..., "context": ...})`` method,
    exactly the LCEL pattern from Week 1 of the curriculum this demonstrates.
    """
    try:
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_groq import ChatGroq
    except ImportError as exc:
        raise CopilotDependencyError("The LCEL answer chain", "langchain-groq") from exc

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )
    model = ChatGroq(model=model_name, temperature=temperature)
    parser = StrOutputParser()

    # The LCEL pipe operator: this IS the pattern, not a metaphor for it.
    chain = prompt | model | parser
    return chain


async def answer_with_citations(
    question: str, chunks: list[RetrievedChunk], model_name: str = "openai/gpt-oss-120b"
) -> tuple[str, list[str]]:
    """High-level helper: run the RAG answer chain, return (answer_text, section_refs).

    Falls back to the core project's own ``aegisflow.llm`` provider abstraction
    (``build_provider().complete_json``) if LangChain isn't installed - so the
    copilot's chat feature still answers questions without the optional LangChain
    dependency, just without demonstrating LCEL specifically. That fallback also means
    a plain ``offline`` provider setting degrades this feature to "no answer, clear
    message why" rather than a crash, exactly like the core pipeline's own LLM call
    sites.
    """
    context = _format_context(chunks)
    citations = sorted({c.section_ref for c in chunks})

    try:
        chain = build_rag_answer_chain(model_name=model_name)
        # .invoke() is synchronous; LangChain's sync LCEL API is used here rather than
        # .ainvoke() to keep this demonstration minimal. For higher request volume,
        # swap to .ainvoke() so this doesn't block the event loop.
        answer = chain.invoke({"question": question, "context": context})
        return answer, citations
    except CopilotDependencyError:
        from aegisflow.core.errors import LLMProviderError
        from aegisflow.core.settings import get_settings
        from aegisflow.llm import build_provider

        settings = get_settings()
        provider = build_provider(settings)
        if provider.is_offline:
            return (
                "The copilot needs a configured LLM provider (Groq or Gemini) to "
                "answer questions; this instance is running with AEGISFLOW_LLM_PROVIDER"
                "=offline, so I can only show you the raw retrieved policy excerpts "
                "below rather than a generated answer.\n\n" + context,
                citations,
            )
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        }
        try:
            result = await provider.complete_json(
                prompt=f"Question: {question}", schema=schema, system=SYSTEM_PROMPT.format(context=context)
            )
            return result.get("answer", ""), citations
        except LLMProviderError as exc:
            return f"The LLM provider call failed: {exc}", citations
