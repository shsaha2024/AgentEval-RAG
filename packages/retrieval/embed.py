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

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
import torch
#If using OpenAI, make sure to set OPENAI_API_KEY in your environment.
# from langchain_openai import OpenAIEmbeddings
# from dotenv import load_dotenv, find_dotenv
# load_dotenv(find_dotenv())

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
    model_name: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    device: str = os.getenv("EMBEDDING_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    normalize_embeddings: bool = True
    batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

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

    #if using OpenAI, ensure API key is set
    # openai_api_key = os.getenv("OPENAI_API_KEY")
    # if not openai_api_key:
    #     raise ValueError(
    #         "OPENAI_API_KEY is not set. Export it before running indexing."
    #     )
    # return OpenAIEmbeddings(
    #     model=config.model_name,
    #     api_key=openai_api_key,
    # )
    return HuggingFaceEmbeddings(
        model_name=config.model_name,
        model_kwargs={
            "device": config.device,
        },
        encode_kwargs={
            "normalize_embeddings": config.normalize_embeddings,
            "batch_size": config.batch_size,
        },
        # Optional local cache directory:
        # cache_folder=os.getenv("HF_HOME")
    )

