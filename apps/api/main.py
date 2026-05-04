from typing import Any, Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ------------------------------------------------------------
# Goal:
# Expose existing retrieval/search functionality through
# a typed FastAPI interface.
#
# At this stage, we are NOT generating answers yet.
# We are only returning retrieved documents/chunks.
# ------------------------------------------------------------

app = FastAPI(
    title="AgentEval-RAG Retrieval API",
    version="0.1.0",
    description="FastAPI service for similarity search over indexed documents."
)

# ------------------------------------------------------------
# Response model for a single retrieved chunk.
#
# Why define this?
# - FastAPI will validate the output shape.
# - Swagger docs will show exactly what users get back.
# - You can keep internal document objects separate from
#   your public API contract.
# ------------------------------------------------------------
class SearchHit(BaseModel):
    chunk_id: str = Field(..., description="Unique identifier for the retrieved chunk")
    text: str = Field(..., description="Chunk text returned by the retriever")
    source: str | None = Field(default=None, description="Original source/document name")
    score: float | None = Field(default=None, description="Similarity score if requested")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Extra metadata")

# ------------------------------------------------------------
# Request model for POST /query
#
# Why a Pydantic model?
# - FastAPI automatically parses JSON into Python objects.
# - Invalid requests are rejected with a helpful 422 error.
# - The schema appears automatically in docs.
# ------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User search query")
    k: int = Field(default=5, ge=1, le=20, description="Number of results to return")
    mode: Literal["similarity", "similarity_with_scores"] = Field(
        default="similarity",
        description="Retrieval mode"
    )
    source: str | None = Field(
        default=None,
        description="Optional source filter, e.g. restrict search to one document"
    )

# ------------------------------------------------------------
# Response model for POST /query
#
# Why wrap hits in a larger response object?
# - Easier to extend later with latency, trace_id, debug info
# - Cleaner contract for frontend or future agent nodes
# ------------------------------------------------------------
class QueryResponse(BaseModel):
    query: str
    mode: str
    k: int
    total_hits: int
    hits: list[SearchHit]

# ------------------------------------------------------------
# This is a placeholder adapter around your existing search code.
#
# In your project, replace the inside of this class with calls to
# the exact search utility you already use in test_search.
#
# The API layer should NOT know retrieval details.
# Its job is only to validate requests, call a retriever, and
# shape the response.
# ------------------------------------------------------------
class RetrieverService:
    def similarity_search(self, query: str, k: int, source: str | None = None):
        """
        Replace this stub with your current similarity search logic.

        Expected return:
            list of document-like objects or dicts
        """
        # Example mocked result format
        docs = [
            {
                "chunk_id": "chunk_001",
                "text": "FastAPI automatically validates request bodies using Pydantic models.",
                "source": source or "fastapi_docs",
                "metadata": {"section": "request-body"}
            },
            {
                "chunk_id": "chunk_002",
                "text": "Response models define the output schema and improve API safety.",
                "source": source or "fastapi_docs",
                "metadata": {"section": "response-model"}
            },
        ]
        return docs[:k]

    def similarity_search_with_scores(self, query: str, k: int, source: str | None = None):
        """
        Replace this stub with your current search-with-scores logic.

        Expected return:
            list of tuples like (doc, score)
            or another format that you normalize below.
        """
        docs_with_scores = [
            (
                {
                    "chunk_id": "chunk_001",
                    "text": "FastAPI automatically validates request bodies using Pydantic models.",
                    "source": source or "fastapi_docs",
                    "metadata": {"section": "request-body"}
                },
                0.9123
            ),
            (
                {
                    "chunk_id": "chunk_002",
                    "text": "Response models define the output schema and improve API safety.",
                    "source": source or "fastapi_docs",
                    "metadata": {"section": "response-model"}
                },
                0.8744
            ),
        ]
        return docs_with_scores[:k]


retriever = RetrieverService()

# ------------------------------------------------------------
# Simple health endpoint.
#
# Why include this now?
# - Useful for deployment and Docker later.
# - Lets you verify the app is alive without testing retrieval.
# - Standard production API practice.
# ------------------------------------------------------------
@app.get("/health")
def health_check():
    return {"status": "ok"}

# ------------------------------------------------------------
# Main retrieval endpoint.
#
# Workflow:
# 1. FastAPI validates the request body against QueryRequest.
# 2. We choose the retrieval method based on the mode.
# 3. We normalize internal search results into SearchHit objects.
# 4. FastAPI validates the final response against QueryResponse.
# ------------------------------------------------------------
@app.post("/query", response_model=QueryResponse)
def query_documents(payload: QueryRequest) -> QueryResponse:
    try:
        if payload.mode == "similarity":
            raw_hits = retriever.similarity_search(
                query=payload.query,
                k=payload.k,
                source=payload.source,
            )

            hits = [
                SearchHit(
                    chunk_id=doc["chunk_id"],
                    text=doc["text"],
                    source=doc.get("source"),
                    score=None,
                    metadata=doc.get("metadata", {}),
                )
                for doc in raw_hits
            ]

        elif payload.mode == "similarity_with_scores":
            raw_hits = retriever.similarity_search_with_scores(
                query=payload.query,
                k=payload.k,
                source=payload.source,
            )

            hits = [
                SearchHit(
                    chunk_id=doc["chunk_id"],
                    text=doc["text"],
                    source=doc.get("source"),
                    score=score,
                    metadata=doc.get("metadata", {}),
                )
                for doc, score in raw_hits
            ]

        else:
            # This is mostly defensive, because the Literal type
            # already restricts allowed values.
            raise HTTPException(status_code=400, detail="Unsupported retrieval mode")

        return QueryResponse(
            query=payload.query,
            mode=payload.mode,
            k=payload.k,
            total_hits=len(hits),
            hits=hits,
        )

    except Exception as exc:
        # In production, replace this with structured logging.
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(exc)}")