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

from sqlalchemy import create_engine, text
from langchain_postgres import PGVector


@dataclass(frozen=True)
class PostgresConfig:
    """
    Typed config object for PostgreSQL connectivity.

    Default values are local-development friendly.
    """
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
    """
    Ensure the pgvector extension exists in the target database.

    Why this matters:
    - PostgreSQL does not automatically enable the extension in each database.
    - The pgvector docs show that you must run:
      CREATE EXTENSION IF NOT EXISTS vector;

    This function runs that statement once before indexing/searching.
    """
    config = config or PostgresConfig()

    engine = create_engine(config.connection_string())

    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))


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