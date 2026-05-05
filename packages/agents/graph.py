"""
Convert the QA pipeline into a LangGraph workflow.

Assumptions:
- You already have retrieval code from steps 4-5.
- You already have a FastAPI service from step 8.
- You have some LLM client or wrapper available.

This file focuses on orchestration only.
"""

from typing import TypedDict, Literal, List, Dict, Any
from langgraph.graph import StateGraph, START, END

# -------------------------------------------------------------------
# 1. Define the shared graph state
# -------------------------------------------------------------------
# LangGraph's StateGraph works by passing a shared state object
# between nodes. Each node reads from the current state and returns
# only the pieces it wants to update.
#
# This follows LangGraph's model: State -> Partial<State>.
# [web:52][web:46]

class RAGState(TypedDict, total=False):
    question: str
    query_type: str
    retrieval_mode: str
    retrieved_docs: List[Dict[str, Any]]
    reranked_docs: List[Dict[str, Any]]
    draft_answer: str
    citations: List[str]
    final_answer: str
    needs_rerank: bool


# -------------------------------------------------------------------
# 2. Plug-in interfaces to your existing code
# -------------------------------------------------------------------
# Replace these with imports from your own project.
# The graph should call your retrieval / generation functions, not
# re-implement them here.

def classify_query(question: str) -> str:
    """
    Simple query classification.
    You can replace this with an LLM-based classifier later.

    Returns labels like:
    - 'factual'
    - 'multi_hop'
    - 'summary'
    """
    q = question.lower()
    if "compare" in q or "difference" in q:
        return "multi_hop"
    if "summarize" in q or "overview" in q:
        return "summary"
    return "factual"


def choose_retrieval_mode(query_type: str) -> str:
    """
    Rule-based router for now.
    Later, this can become more sophisticated.

    Example policy:
    - factual -> dense
    - multi_hop -> hybrid
    - summary -> hybrid
    """
    if query_type in {"multi_hop", "summary"}:
        return "hybrid"
    return "dense"


def dense_retrieve(question: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Replace with your step-5 dense retriever.
    Expected doc structure:
    {
        'chunk_id': 'doc_12_chunk_3',
        'text': '...',
        'source': 'https://...',
        'score': 0.84
    }
    """
    # TODO: import and call your actual retriever
    return [
        {
            "chunk_id": "dense_1",
            "text": "Dense retrieval result 1 about the question.",
            "source": "https://example.com/doc1",
            "score": 0.84,
        },
        {
            "chunk_id": "dense_2",
            "text": "Dense retrieval result 2 with supporting details.",
            "source": "https://example.com/doc2",
            "score": 0.77,
        },
    ]


def hybrid_retrieve(question: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Replace with your hybrid retriever if available.
    If hybrid is not ready yet, you can temporarily point this to
    dense_retrieve() and still keep the graph structure intact.
    """
    return [
        {
            "chunk_id": "hybrid_1",
            "text": "Hybrid retrieval result 1 with richer evidence.",
            "source": "https://example.com/doc3",
            "score": 0.89,
        },
        {
            "chunk_id": "hybrid_2",
            "text": "Hybrid retrieval result 2 with additional context.",
            "source": "https://example.com/doc4",
            "score": 0.80,
        },
    ]


def simple_rerank(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Placeholder reranker.
    In production, replace with a cross-encoder or learned reranker.
    For now, sort by score descending.
    """
    return sorted(docs, key=lambda d: d.get("score", 0.0), reverse=True)


def generate_answer(question: str, docs: List[Dict[str, Any]]) -> str:
    """
    Replace with your actual LLM call.

    A good pattern is to build a prompt that includes:
    - the question
    - retrieved context
    - instructions to stay grounded
    - instructions to cite sources
    """
    joined_context = "\n\n".join(
        f"[{i+1}] {doc['text']}" for i, doc in enumerate(docs)
    )
    return (
        f"Question: {question}\n\n"
        f"Grounded answer based on retrieved evidence:\n{joined_context}\n\n"
        f"Synthesized response: This is a placeholder answer generated from the retrieved context."
    )


def extract_citations(docs: List[Dict[str, Any]]) -> List[str]:
    """
    Pull source URLs or source ids from the documents used.
    """
    return [doc["source"] for doc in docs]


# -------------------------------------------------------------------
# 3. Define LangGraph nodes
# -------------------------------------------------------------------
# Each node:
# - reads the current state
# - returns a partial update dictionary
#
# This is the core LangGraph style described in the docs.
# [web:46][web:52]

def classify_node(state: RAGState) -> Dict[str, Any]:
    """
    Determine what kind of question this is.
    This lets the workflow adapt retrieval strategy.
    """
    question = state["question"]
    query_type = classify_query(question)
    return {"query_type": query_type}


def retrieval_mode_node(state: RAGState) -> Dict[str, Any]:
    """
    Choose which retriever to use based on query type.
    """
    mode = choose_retrieval_mode(state["query_type"])
    needs_rerank = mode == "hybrid"
    return {
        "retrieval_mode": mode,
        "needs_rerank": needs_rerank,
    }


def retrieve_node(state: RAGState) -> Dict[str, Any]:
    """
    Execute retrieval using the chosen retrieval mode.
    """
    question = state["question"]
    mode = state["retrieval_mode"]

    if mode == "hybrid":
        docs = hybrid_retrieve(question, k=5)
    else:
        docs = dense_retrieve(question, k=5)

    return {"retrieved_docs": docs}


def rerank_node(state: RAGState) -> Dict[str, Any]:
    """
    Optional reranking step.
    For a first version, this can be very simple.
    Later, use a cross-encoder reranker here.
    """
    docs = state["retrieved_docs"]
    reranked = simple_rerank(docs)
    return {"reranked_docs": reranked}


def generate_node(state: RAGState) -> Dict[str, Any]:
    """
    Draft the answer from the best available evidence.
    If reranked docs exist, use them; otherwise use retrieved docs.
    """
    question = state["question"]
    docs = state.get("reranked_docs") or state.get("retrieved_docs", [])

    answer = generate_answer(question, docs)
    citations = extract_citations(docs)

    return {
        "draft_answer": answer,
        "citations": citations,
    }


def finalize_node(state: RAGState) -> Dict[str, Any]:
    """
    Final packaging step.
    In later steps, you can add groundedness checks, formatting,
    confidence notes, or a response schema here.
    """
    final_answer = state["draft_answer"]
    return {"final_answer": final_answer}


# -------------------------------------------------------------------
# 4. Conditional routing functions
# -------------------------------------------------------------------
# LangGraph supports conditional edges so execution can branch
# depending on the current state. This is what makes the workflow
# "agentic" rather than a fixed sequence.
# [web:17][web:46]

def route_after_retrieval_mode(state: RAGState) -> Literal["retrieve_node"]:
    """
    This function exists mainly to make the routing explicit.
    You can later expand this to choose different retriever nodes.
    """
    return "retrieve_node"


def route_after_retrieve(state: RAGState) -> Literal["rerank_node", "generate_node"]:
    """
    Decide whether to rerank or go directly to generation.
    """
    if state.get("needs_rerank", False):
        return "rerank_node"
    return "generate_node"


# -------------------------------------------------------------------
# 5. Build the graph
# -------------------------------------------------------------------
# The workflow is:
# START
#   -> classify_node
#   -> retrieval_mode_node
#   -> retrieve_node
#   -> (rerank_node or generate_node)
#   -> generate_node
#   -> finalize_node
#   -> END
#
# This matches the workflow pattern discussed in LangGraph docs.
# [web:17][web:46]

def build_rag_graph():
    builder = StateGraph(RAGState)

    # Register nodes
    builder.add_node("classify_node", classify_node)
    builder.add_node("retrieval_mode_node", retrieval_mode_node)
    builder.add_node("retrieve_node", retrieve_node)
    builder.add_node("rerank_node", rerank_node)
    builder.add_node("generate_node", generate_node)
    builder.add_node("finalize_node", finalize_node)

    # Fixed edges
    builder.add_edge(START, "classify_node")
    builder.add_edge("classify_node", "retrieval_mode_node")

    # Explicit routing after choosing retrieval mode
    builder.add_conditional_edges(
        "retrieval_mode_node",
        route_after_retrieval_mode,
        {
            "retrieve_node": "retrieve_node",
        },
    )

    # Branch after retrieval
    builder.add_conditional_edges(
        "retrieve_node",
        route_after_retrieve,
        {
            "rerank_node": "rerank_node",
            "generate_node": "generate_node",
        },
    )

    # If reranking happens, then generate
    builder.add_edge("rerank_node", "generate_node")

    # Finalize and stop
    builder.add_edge("generate_node", "finalize_node")
    builder.add_edge("finalize_node", END)

    # Compile into an executable graph object
    graph = builder.compile()
    return graph


# -------------------------------------------------------------------
# 6. Simple local test
# -------------------------------------------------------------------
# This lets you run the graph directly before wiring it into FastAPI.

if __name__ == "__main__":
    graph = build_rag_graph()

    result = graph.invoke(
        {
            "question": "Compare dense retrieval and hybrid retrieval for technical question answering."
        }
    )

    print("\nFINAL STATE:\n")
    for key, value in result.items():
        print(f"{key}: {value}\n")