"""
Retriever utilities for querying a PGVector-backed PostgreSQL store.

Why this file exists:
- Encapsulates query-time retrieval logic.
- Makes it easy to reuse retrieval in scripts, APIs, and evaluation code.
- Provides both:
  1. raw similarity search helpers
  2. a retriever object for future chain / agent integration
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from matplotlib.pylab import source

from packages.retrieval.embed import get_embeddings, EmbeddingConfig
from packages.retrieval.vector_store import get_vector_store, PostgresConfig


@dataclass(frozen=True)
class RetrievalConfig:
    """
    Typed retrieval config.

    Attributes:
        k:
            Number of top documents to return.
        score_threshold:
            Optional threshold to filter weak matches.
            Right now we expose scores but do not hard-filter them automatically.
    """
    k: int = 4
    score_threshold: float | None = None


class PGVectorRetriever:
    """
    Small service wrapper around the LangChain PGVector store.

    This class:
    - creates embeddings
    - connects to the configured collection
    - exposes helper methods for search

    You can later extend this with:
    - metadata filtering
    - hybrid retrieval
    - reranking
    - query rewriting
    """

    def __init__(
        self,
        embedding_config: EmbeddingConfig | None = None,
        postgres_config: PostgresConfig | None = None,
        retrieval_config: RetrievalConfig | None = None,
    ) -> None:
        self.embedding_config = embedding_config or EmbeddingConfig()
        self.postgres_config = postgres_config or PostgresConfig()
        self.retrieval_config = retrieval_config or RetrievalConfig()

        # Create the embeddings object used for query encoding.
        self.embeddings = get_embeddings(self.embedding_config)

        # Connect to the PGVector store using the same collection used at index time.
        self.vector_store = get_vector_store(
            embeddings=self.embeddings,
            config=self.postgres_config,
        )

    def similarity_search(
        self,
        query: str,
        k: int | None = None,
        filter: dict[str, Any] | None = None,
    ):
        """
        Return the top-k most similar documents for a text query.

        Args:
            query:
                User query string.
            k:
                Number of documents to return. Falls back to config default.
            filter:
                Optional metadata filter. Keep this here because it is useful later
                if need to do source-specific retrieval (for example only FastAPI docs).

        Returns:
            List[Document]
        """
        k = k or self.retrieval_config.k

        return self.vector_store.similarity_search(
            query=query,
            k=k,
            filter=filter,
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: int | None = None,
        filter: dict[str, Any] | None = None,
    ):
        """
        Return the top-k most similar documents together with similarity scores.

        Why use this:
        - Helpful for debugging retrieval quality.
        - Useful for logging and benchmarking.
        - Lets you inspect whether retrieved chunks are clearly relevant.

        Returns:
            List[Tuple[Document, float]]
        """
        k = k or self.retrieval_config.k

        return self.vector_store.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter,
        )

    def as_langchain_retriever(
        self,
        k: int | None = None,
        filter: dict[str, Any] | None = None,
    ):
        """
        Return a retriever object compatible with LangChain chains / graphs.

        This is especially useful later when connecting retrieval into:
        - retrieval QA chains
        - LangGraph workflows
        - custom agent nodes

        Notes:
        - LangChain vector stores commonly expose as_retriever(search_kwargs=...).
        - We pass search kwargs so you can control k and metadata filters.
        """
        k = k or self.retrieval_config.k

        return self.vector_store.as_retriever(
            search_kwargs={
                "k": k,
                "filter": filter,
            }
        )

    @staticmethod
    def format_document(doc) -> str:
        """
        Nicely format a retrieved document for CLI display.

        This keeps the search script simple and provides a clean way to inspect
        what is actually coming back from the vector store.
        """
        metadata = doc.metadata or {}
        title = metadata.get("title", "untitled")
        sections = []
        for i in range(1, 5):
            if metadata[f"section_h{i}"]:
                sections.append(metadata[f"section_h{i}"])
            else:
                sections.append(sections[-1] if sections else "Nil")
        url = metadata.get("url", "no-url")
        chunk_ind = metadata.get("chunk_index", "no-chunk-id")

        return (
            f"Title    : {title}\n"
            f"Sections  : {sections}\n"
            f"Chunk Index : {chunk_ind}\n"
            f"URL      : {url}\n"
            f"Content  : {doc.page_content}\n"
        )