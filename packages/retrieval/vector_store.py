"""
PostgreSQL + pgvector vector-store utilities.

Why this file exists:
- Centralizes DB connection setup.
- Ensures the pgvector extension exists.
- Returns a configured LangChain PGVector store that can be reused for
  indexing and querying.

References:
- LangChain PGVector docs:
  https://docs.langchain.com/oss/python/integrations/vectorstores/pgvector
- PGVector API reference:
  https://reference.langchain.com/python/langchain-postgres/vectorstores/PGVector
- pgvector-python docs:
  https://github.com/pgvector/pgvector-python
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
import re
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from langchain_postgres import PGVector

@dataclass(frozen=True)
class PostgresConfig:
    host: str = os.getenv("POSTGRES_HOST", "localhost")
    port: int = int(os.getenv("POSTGRES_PORT", "6024"))
    database: str = os.getenv("POSTGRES_DB", "langchain")
    user: str = os.getenv("POSTGRES_USER", "langchain")
    password: str = os.getenv("POSTGRES_PASSWORD", "langchain")
    collection_name: str = os.getenv("PGVECTOR_COLLECTION", "agenteval_docs")

    def connection_string(self) -> str:
        """
        Return a SQLAlchemy/Psycopg3 connection string.

        LangChain PGVector docs use the format:
        postgresql+psycopg://user:password@host:port/db
        """
        return (
            f"postgresql+psycopg://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )


def ensure_vector_extension(config: PostgresConfig | None = None) -> None:
    config = config or PostgresConfig()

    engine = create_engine(
        config.connection_string(),
        pool_pre_ping=True,
    )

    try:
        with engine.begin() as connection:
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
    except OperationalError as e:
        raise RuntimeError(
            "Could not connect to PostgreSQL.\n"
            f"Tried: host={config.host}, port={config.port}, db={config.database}, user={config.user}\n"
            "Check that:\n"
            "1. PostgreSQL/pgvector container is running\n"
            "2. The Docker port mapping matches your POSTGRES_PORT\n"
            "3. Username/password/database are correct\n"
            "4. You are using 127.0.0.1 instead of localhost if localhost causes issues\n"
            f"Original error: {e}"
        ) from e

def get_vector_store(embeddings, config: PostgresConfig | None = None) -> PGVector:
    """
    Return a configured LangChain PGVector vector store.

    Args:
        embeddings:
            A LangChain-compatible embeddings object.
        config:
            Optional database config.

    Returns:
        An initialized PGVector store.

    Notes:
    - use_jsonb=True stores metadata in JSONB, which is a strong default
      for flexible filtering and future extension.
    - collection_name acts like a logical namespace for your document set.
    """
    config = config or PostgresConfig()

    ensure_vector_extension(config)

    return PGVector(
        embeddings=embeddings,
        collection_name=config.collection_name,
        connection=config.connection_string(),
        use_jsonb=True,
    )