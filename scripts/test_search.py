"""
Simple CLI script for testing PGVector similarity search.

What this script does:
- Connects to your existing PGVector collection
- Encodes a user query with the same embedding model family
- Runs similarity search
- Prints retrieved chunks and optional scores

Usage examples:
python scripts/test_search.py --query "How do I test FastAPI endpoints?"
python scripts/test_search.py --query "What is LangGraph used for?" --k 5 --with-scores
python scripts/test_search.py --query "How do I test FastAPI endpoints?" --source fastapi-docs

Why this is useful:
- Lets you verify the index was built correctly
- Helps inspect retrieval quality before building generation
- Gives you a debugging surface for chunking / metadata / scoring issues

References:
- PGVector similarity search reference:
  https://reference.langchain.com/python/langchain-postgres/vectorstores/PGVector/similarity_search
- PGVector integration docs:
  https://docs.langchain.com/oss/python/integrations/vectorstores/pgvector
"""

from __future__ import annotations

import argparse

from packages.retrieval.embed import EmbeddingConfig
from packages.retrieval.retriever import PGVectorRetriever, RetrievalConfig
from packages.retrieval.vector_store import PostgresConfig


def main():
    """
    Parse CLI arguments, run a similarity search, and print results.

    This script intentionally stays simple:
    - no API
    - no reranking
    - no generation
    just pure retrieval inspection
    """
    parser = argparse.ArgumentParser(description="Test PGVector search from CLI.")
    parser.add_argument(
        "--query",
        required=True,
        help="Natural-language query to search for.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Number of top documents to return.",
    )
    parser.add_argument(
        "--with-scores",
        action="store_true",
        help="Print similarity scores along with documents.",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Optional metadata filter: source=<value>.",
    )
    parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-small",
        help="Embedding model name used at query time.",
    )
    args = parser.parse_args()

    # Build an optional metadata filter.
    # This is useful if you indexed multiple corpora into one collection
    # and want to restrict search to a specific source.
    metadata_filter = None
    if args.source:
        metadata_filter = {"source": args.source}

    # Instantiate the retrieval service.
    retriever = PGVectorRetriever(
        embedding_config=EmbeddingConfig(model_name=args.embedding_model),
        postgres_config=PostgresConfig(),
        retrieval_config=RetrievalConfig(k=args.k),
    )

    print("\n=== QUERY ===")
    print(args.query)

    if metadata_filter:
        print("\n=== METADATA FILTER ===")
        print(metadata_filter)

    print("\n=== RESULTS ===")

    if args.with_scores:
        results = retriever.similarity_search_with_score(
            query=args.query,
            k=args.k,
            filter=metadata_filter,
        )

        if not results:
            print("No results found.")
            return

        for rank, (doc, score) in enumerate(results, start=1):
            print(f"\n--- Result {rank} ---")
            print(f"Score    : {score}")
            print(retriever.format_document(doc))

    else:
        results = retriever.similarity_search(
            query=args.query,
            k=args.k,
            filter=metadata_filter,
        )

        if not results:
            print("No results found.")
            return

        for rank, doc in enumerate(results, start=1):
            print(f"\n--- Result {rank} ---")
            print(retriever.format_document(doc))


if __name__ == "__main__":
    main()