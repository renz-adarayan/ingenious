"""Answer generation component for Azure AI Search RAG pipeline.

This module defines `AnswerGenerator`, which formats retrieved source chunks
into a prompt and calls an LLM (e.g., Azure OpenAI) to synthesize a final
answer. It is intentionally lightweight and test-friendly:
- If an LLM client is not injected, a small AsyncMock-shaped stub is created,
  so tests can patch/await `chat.completions.create` and `close`.
- The prompt renderer provides *both* `context` and `sources` keys to support
  templates that reference either placeholder.
- Exceptions from the LLM call are allowed to bubble; tests may assert them.

Usage:
    gen = AnswerGenerator(cfg, llm_client)
    answer = await gen.generate("question", chunks)
    await gen.close()
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock

from ingenious.services.azure_search.config import SearchConfig

# ------------------------------ Constants ------------------------------------

DEFAULT_TEMPERATURE: float = 0.2

# Default RAG prompt template.
# Note: We deliberately include {context}. The generator supplies *both*
# 'context' and 'sources' when formatting, so prompts using {sources}
# continue to work without changes.
DEFAULT_RAG_PROMPT: str = (
    "System:\n"
    "You are an intelligent assistant designed to answer user questions based "
    "strictly on the provided context.\n"
    "Instructions:\n"
    "1. Analyze the user's question.\n"
    "2. Review the provided context (Source Chunks).\n"
    "3. Synthesize a comprehensive answer using only information found in the "
    "context.\n"
    "4. If the context does not contain the answer, state clearly that the "
    "information is not available in the provided sources.\n"
    "5. Do not use any external knowledge or make assumptions beyond the given "
    "context.\n"
    "6. Cite the sources used by referencing the source number (e.g., [Source 1], "
    "[Source 2]).\n"
    "Context (Source Chunks):\n"
    "{context}"
)


class AnswerGenerator:
    """Synthesize a final answer from top‑N retrieved chunks with an LLM.

    Why:
        In a RAG pipeline, retrieved text chunks must be coherently presented
        to a generation model. This class encapsulates that formatting and the
        call to the LLM, while remaining easy to unit test.

    Notes:
        - If `llm_client` is not provided, we construct a stub client exposing
          `.chat.completions.create(...)` and `.close()` as async callables so
          tests can patch/await them without a real network.
        - The prompt renderer supports both `{context}` and `{sources}` keys to
          avoid brittle coupling between templates and code.
    """

    def __init__(self, config: SearchConfig, llm_client: Optional[Any] = None) -> None:
        """Initialize the generator with config and an optional LLM client.

        Args:
            config: Validated `SearchConfig` containing generation deployment.
            llm_client: Optional async LLM client. If None, a test-friendly
                stub client is created.

        Side effects:
            Stores a flag tracking ownership, so `close()` only awaits the
            internal client when this class created it.
        """
        self._cfg = config
        self.rag_prompt_template: str = DEFAULT_RAG_PROMPT

        if llm_client is None:
            # Test-friendly stub client shaped like AsyncAzureOpenAI:
            # client.chat.completions.create(...) -> awaitable
            chat_stub = type("ChatStub", (), {})()
            completions_stub = type("CompletionsStub", (), {})()
            setattr(completions_stub, "create", AsyncMock())
            setattr(chat_stub, "completions", completions_stub)
            self._llm_client = type("ClientStub", (), {})()
            setattr(self._llm_client, "chat", chat_stub)
            setattr(self._llm_client, "close", AsyncMock())
            self._owns_llm = True
        else:
            self._llm_client = llm_client
            self._owns_llm = False

    # ------------------------------ Internals --------------------------------

    def _format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Return a human-readable context string from retrieved chunks.

        Each chunk is annotated with a stable source ordinal to enable simple
        bracket-style citations in downstream answers.

        Args:
            chunks: Retrieved result dictionaries.

        Returns:
            A newline-separated string with `[Source i]` headers and content.
        """
        lines: List[str] = []
        for i, ch in enumerate(chunks, start=1):
            content = ch.get(self._cfg.content_field, "N/A")
            lines.append(f"[Source {i}] {content}")
        return "\n---\n".join(lines)

    def _render_prompt(self, question: str, chunks: List[Dict[str, Any]]) -> str:
        """Render the prompt for the LLM, providing both 'context' and 'sources'.

        Why:
            Different prompt templates historically used either `{context}` or
            `{sources}`. Supplying both guards against KeyError if a template
            changes without code changes.

        Args:
            question: The user question.
            chunks: Retrieved result dictionaries.

        Returns:
            The fully rendered prompt string.
        """
        sources_str = self._format_context(chunks)
        return self.rag_prompt_template.format(
            question=question,
            context=sources_str,
            sources=sources_str,
        )

    # --------------------------------- API -----------------------------------

    async def generate(self, question: str, chunks: List[Dict[str, Any]]) -> str:
        """Generate an answer using the configured LLM.

        The method short-circuits when no chunks are provided and otherwise
        delegates to the LLM client. Exceptions from the LLM call are allowed
        to propagate (tests may assert them), ensuring failures are visible to
        callers.

        Args:
            question: The user's question.
            chunks: Retrieved result dictionaries to use as context.

        Returns:
            The model's answer content string on success; a plain error message
            only in the unlikely case the SDK returns a shape without content.

        Raises:
            Whatever exception the underlying LLM client raises, e.g.,
            `RuntimeError` from a test stub.
        """
        if not chunks:
            return "I could not find any relevant information to answer your question."

        prompt = self._render_prompt(question, chunks)

        res = await self._llm_client.chat.completions.create(
            model=self._cfg.generation_deployment_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=DEFAULT_TEMPERATURE,
        )

        msg = getattr(res.choices[0], "message", None)
        if msg and getattr(msg, "content", None):
            return str(msg.content)
        return "An error occurred while generating an answer."

    async def close(self) -> None:
        """Close the underlying client if this instance created it.

        Why:
            Ensures graceful shutdown and prevents connection leaks during
            tests and production use alike.
        """
        if self._owns_llm:
            await self._llm_client.close()
