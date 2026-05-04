from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """
    Input contract for retrieval requests.

    We keep this request model generic so the same endpoint can support:
    - plain similarity search
    - fixed-source search
    - returning scores
    - different top-k values
    """
    query: str = Field(..., min_length=1, description="User search query")
    k: int = Field(5, ge=1, le=50, description="Number of results to return")
    source_filter: Optional[str] = Field(
        default=None,
        description="Optional source/document collection filter"
    )
    return_scores: bool = Field(
        default=False,
        description="Whether to include similarity scores in the response"
    )


class RetrievedChunk(BaseModel):
    """
    Standardized shape for one retrieved item.

    This is useful even if your internal service returns a different object type.
    We normalize the output here so the API stays stable.
    """
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: Optional[float] = None


class SearchResponse(BaseModel):
    """
    Output contract for retrieval responses.

    Returning a response_model is useful because FastAPI uses it for:
    - response validation
    - serialization
    - automatic API docs generation
    """
    query: str
    k: int
    source_filter: Optional[str] = None
    return_scores: bool
    results: List[RetrievedChunk]