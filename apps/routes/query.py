from fastapi import APIRouter, FastAPI
from pydantic import BaseModel
from packages.agents.graph import build_rag_graph

router = APIRouter()
graph = build_rag_graph()

class QueryRequest(BaseModel):
    question: str

@router.post("/query")
def query_endpoint(request: QueryRequest):
    result = graph.invoke({"question": request.question})
    return {
        "question": request.question,
        "answer": result.get("final_answer"),
        "citations": result.get("citations", []),
        "retrieval_mode": result.get("retrieval_mode"),
        "query_type": result.get("query_type"),
    }

app = FastAPI()
app.include_router(router)