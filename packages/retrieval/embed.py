"""
Embedding utilities for the RAG system.

Why this file exists:
- Keeps embedding-provider logic in one place.
- Makes it easy to swap models later without touching the vector store code.
- Supports clean dependency injection for indexing and querying.

Current default:
- OpenAI embeddings via LangChain, because PGVector accepts any LangChain-compatible
  embeddings object and OpenAI is the simplest production-style starting point.

"""

from __future__ import annotations

import os
from dataclasses import dataclass
import yaml

from langchain_openai import OpenAIEmbeddings

config = yaml.safe_load(open("configs/embedding.yaml", 'r'))


@dataclass(frozen=True)
class EmbeddingConfig:
    """
    Typed config object for embedding settings.

    Can later extend this with:
    - provider: "openai" | "huggingface"
    - dimensions
    - batch size
    - timeout
    """
    model_name: str = "text-embedding-3-small"


def get_embeddings(config: EmbeddingConfig | None = None):
    """
    Return a LangChain embeddings object.

    Args:
        config:
            Optional EmbeddingConfig. If omitted, a sensible default is used.

    Returns:
        A LangChain embeddings instance.

    Raises:
        ValueError:
            If OPENAI_API_KEY is not set in the environment.

    Notes:
    - PGVector accepts a LangChain embeddings object.
    - OpenAIEmbeddings is used here for simplicity.
    - If you want a fully open-source stack later, you can replace this with
      Hugging Face embeddings without changing the vector store interface.
    """
    config = config or EmbeddingConfig()

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. Export it before running indexing."
        )

    return OpenAIEmbeddings(
        model=config.model_name,
        api_key=openai_api_key,
    )