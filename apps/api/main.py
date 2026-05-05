from typing import Any, Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from packages.retrieval.retriever import PGVectorRetriever, RetrievalConfig
from packages.retrieval.vector_store import PostgresConfig
from packages.retrieval.embed import EmbeddingConfig

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
# - Can keep internal document objects separate from public API contract.
# ------------------------------------------------------------
class SearchHit(BaseModel):
    text: str = Field(..., description="Chunk text returned by the retriever")
    url: str | None = Field(default=None, description="Original source/document url.")
    score: float | None = Field(default=None, description="Similarity score if requested")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Extra metadata") #Keys: Sections, title, chunk_index

def get_metadata(doc: dict[str, Any]) -> dict[str, Any]:
    metadata = {}
    metadata["title"] = doc.metadata.get("title", "")
    metadata["chunk_index"] = doc.metadata.get("chunk_index", 0)
    metadata["Sections"] = []
    if doc.metadata.get("section_h1",None):
        metadata["Sections"] = [doc.metadata.get("section_h1", "")]
        curr = 1
        while curr<5:
            curr+=1
            if doc.metadata.get(f"section_h{curr}", None):
                metadata["Sections"].append(doc.metadata.get(f"section_h{curr}", ""))
            else:
                break
    return metadata
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
    k: int = Field(default=4, ge=1, le=10, description="Number of results to return")
    mode: Literal["similarity", "similarity_with_scores"] = Field(
        default="similarity",
        description="Retrieval mode"
    )
    source: str | None = Field(
        default=None,
        description="Optional source filter, e.g. restrict search to one document class (fastapi or langchain)."
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
# The API layer should NOT know retrieval details.
# Its job is only to validate requests, call a retriever, and
# shape the response.
# ------------------------------------------------------------
class RetrieverService:
    def __init__(self, query: str, k:int, model: str = "BAAI/bge-small-en-v1.5", source: str | None = None):
        self.query = query
        self.k = k
        self.retriever = PGVectorRetriever(
        embedding_config=EmbeddingConfig(model_name=model),
        postgres_config=PostgresConfig(),
        retrieval_config=RetrievalConfig(k=k),
    )
        if source=="fastapi":
            self.source = {"url": {"$like": "%"+self.source+".tiangolo.com"+"%"}}
        elif source=="langchain":
            self.source = {"url": {"$like": "%"+self.source+".com"+"%"}}
        elif source is not None:
            self.source = None
            print(f"Warning: unrecognized source '{self.source}'. Source must be either 'fastapi' or 'langchain'. No metadata filter will be applied.")
        else:
            self.source = None

    def similarity_search(self):
        """
        Expected return:
            list of document-like objects or dicts
        """
        results = self.retriever.similarity_search(
            query=self.query,
            k=self.k,
            filter=self.source,
        )
        return results
              
        # Example mocked result format
        # docs = [
        #     {
        #         "chunk_id": "chunk_001",
        #         "text": "FastAPI automatically validates request bodies using Pydantic models.",
        #         "source": source or "fastapi_docs",
        #         "metadata": {"section": "request-body"}
        #     }
        # ]

    def similarity_search_with_scores(self):
        """
        Expected return:
            list of tuples like (doc, score)
            or another format that you normalize below.
        """
        results = self.retriever.similarity_search_with_score(
            query=self.query,
            k=self.k,
            filter=self.source,
        )
        for ind, (doc,score) in enumerate(results):
            results[ind] = (doc, 1-score)  # convert distance to similarity for better interpretability       
        return results
        # docs_with_scores = [
        #     (
        #         {
        #             "chunk_id": "chunk_001",
        #             "text": "FastAPI automatically validates request bodies using Pydantic models.",
        #             "source": source or "fastapi_docs",
        #             "metadata": {"section": "request-body"}
        #         },
        #         0.9123
        #     )
        # ]


# ------------------------------------------------------------
# Simple health endpoint.
#
# Why include this now?
# - Useful for deployment and Docker later.
# - Verifies the app is alive without testing retrieval.
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
    service = RetrieverService(query = payload.query, k=payload.k, source=payload.source)
    try:
        if payload.mode == "similarity":
            raw_hits = service.similarity_search()

            hits = [
                SearchHit(
                    text=doc.page_content,
                    url=doc.metadata.get("url"),
                    score=None,
                    metadata=get_metadata(doc), #Keys: Sections, title, chunk_index
                )
                for doc in raw_hits
            ]

        elif payload.mode == "similarity_with_scores":
            raw_hits = service.similarity_search_with_scores()

            hits = [
                SearchHit(
                    text=doc.page_content,
                    url=doc.metadata.get("url"),
                    score=score,
                    metadata=get_metadata(doc), #Keys: Sections, title, chunk_index
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