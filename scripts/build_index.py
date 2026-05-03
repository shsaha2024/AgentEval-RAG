"""
Build a pgvector index from processed JSONL chunk files.

Expected input format:
Each line in the JSONL file should look like:
{"chunk_id": "246504830bd274c4__00000", 
"doc_id": "246504830bd274c4", 
"source_path": "data\\raw\\fastapi\\fastapi_deployment.json", 
"title": "Deployment - FastAPI", 
"url": "https://fastapi.tiangolo.com/deployment/", "
chunk_index": 0, 
"section_h1": null, "section_h2": null, "section_h3": null, "section_h4": null, 
"text": "Deployment",
"char_count": 10}

Why this script exists:
- Provides a reproducible indexing entry point.
- Turns processed corpus files into LangChain Documents.
- Loads embeddings into PostgreSQL/pgvector.

References:
- LangChain PGVector docs:
  https://docs.langchain.com/oss/python/integrations/vectorstores/pgvector
- PGVector API reference:
  https://reference.langchain.com/python/langchain-postgres/vectorstores/PGVector
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from langchain_core.documents import Document
import torch

from packages.retrieval.embed import get_embeddings, EmbeddingConfig
from packages.retrieval.vector_store import get_vector_store, PostgresConfig

from math import ceil
from tqdm.auto import tqdm

def load_jsonl_documents(path: str | Path) -> list[Document]:
    """
    Load processed JSONL chunks and convert them to LangChain Document objects.

    Args:
        path:
            Path to a JSONL file.

    Returns:
        List of LangChain Document objects.

    Raises:
        ValueError:
            If a row is missing the required 'text' field.
    """
    path = Path(path)
    documents: list[Document] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            row = json.loads(line)

            text = row.get("text")
            if not text:
                raise ValueError(
                    f"Missing 'text' field in {path} at line {line_number}"
                )
            metadata = {
                "title": row.get("title", ""),
                "chunk_index": row.get("chunk_index", 0),
                "url": row.get("url", "")
            }
            for i in range(1, 5):
                section_key = f"section_h{i}"
                metadata[section_key] = row.get(section_key, None)
            documents.append(
                Document(
                    page_content=text,
                    metadata=metadata,
                )
            )

    return documents

def batched(iterable, batch_size):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]

def main():
    """
    CLI entry point for building the index.

    Steps:
    1. Read processed JSONL chunks.
    2. Create embeddings client.
    3. Initialize PGVector store.
    4. Add documents to the collection.
    """
    parser = argparse.ArgumentParser(description="Build pgvector index from JSONL.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to processed JSONL file.",
    )
    parser.add_argument(
        "--embedding-model",
        default= "BAAI/bge-small-en-v1.5", # other options: "text-embedding-3-small", "sentence-transformers/all-MiniLM-L6-v2"
        help="Embedding model name to use.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Device for local embedding inference.",
    )
    parser.add_argument(
        "--batch_size",
        default=32,
        type = int,
        help="Batch size for embedding computation.",
    )
    args = parser.parse_args()

    # Load processed documents from disk.
    documents = load_jsonl_documents(args.input)

    if not documents:
        raise ValueError("No documents found in the provided JSONL file.")
    print(f"Loaded {len(documents)} documents from {args.input}.")
    # Create embeddings object.
    embeddings = get_embeddings(
        EmbeddingConfig(model_name=args.embedding_model, device = args.device, normalize_embeddings=True)
    )
    print(f"Initialized embeddings with model '{args.embedding_model}'.")
    #Initialize PGVector-backed store.
    vector_store = get_vector_store(
        embeddings=embeddings,
        config=PostgresConfig(),
    )
    print("Initialized PGVector store with provided database configuration.")
    # Add documents to Postgres.
    # PGVector computes embeddings and stores them together with metadata.
    chunk = 100
    batches = list(batched(documents, chunk))
    for batch in tqdm(batches, total=len(batches), desc="Indexing documents"):
        vector_store.add_documents(batch)

    print(
        f"Indexed {len(documents)} documents into collection "
        # f"'{PostgresConfig().collection_name}'."
    )


if __name__ == "__main__":
    main()