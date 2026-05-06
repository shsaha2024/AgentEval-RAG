"""
Convert the QA pipeline into a LangGraph workflow.

This file focuses on orchestration.
"""

from time import time
from typing import TypedDict, Literal, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from google import genai 
from dotenv import load_dotenv
import os
import yaml
from packages.retrieval.retriever import PGVectorRetriever, RetrievalConfig
from packages.retrieval.vector_store import PostgresConfig
from packages.retrieval.embed import EmbeddingConfig
from sentence_transformers import CrossEncoder


load_dotenv()  # Load environment variables from .env file
config = yaml.safe_load(open("configs/agent.yaml", 'r'))

api_key = os.getenv("GEMINI_API_KEY") #extracting api key from .env file, make sure to set it before running the code
client = genai.Client(api_key=api_key)
backup_model = config.get("backup_model", "gemini-2.5-flash-lite")  # Optional backup model in case the primary fails
model = config.get("model", "gemini-2.5-flash")  # Default to "gemini-2.5-flash" if not specified
#sample usage: client.models.generate_content( model="gemini-2.5-flash", contents="Explain how AI works in a few words")

# -------------------------------------------------------------------
# The shared graph state
# -------------------------------------------------------------------
# LangGraph's StateGraph works by passing a shared state object
# between nodes. Each node reads from the current state and returns
# only the pieces it wants to update.
#
# This follows LangGraph's model: State -> Partial<State>.

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
    original_question: str

# -------------------------------------------------------------------
# Plug-in interfaces to existing code
# -------------------------------------------------------------------
# Replace these with imports from your own project.
# The graph should call your retrieval / generation functions, not
# re-implement them here.

def classify_query(question: str) -> str:
    """
    Simple query classification.
    Returns labels like:
    - 'factual'
    - 'multi_hop'
    - 'summary'
    """
    prompt = f"Classify the following question as one of the labels in the array ['factual', 'multi_hop', 'summary'] by responding with only the chosen label and nothing else:\n\
          Question: {question}"
    try:
        response = client.models.generate_content(model=model, contents=prompt)
    except Exception as e:
        print(f"Error generating content: {e}")
        response = client.models.generate_content(model=backup_model, contents=prompt)
    category = response.text.strip()
    if category not in {"factual", "multi_hop", "summary"}:
        print(f"Warning: unrecognized query type '{category}'. Defaulting to 'factual'.")
        return "factual"
    else:
        return category

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


def dense_retrieve(question: str, k: int = 4) -> List[Dict[str, Any]]:
    """
    Expected doc structure:
    {
        'chunk_id': 'doc_12_chunk_3',
        'text': '...',
        'source': 'https://...',
        'score': 0.84
    }
    """
    retriever = PGVectorRetriever(
        embedding_config=EmbeddingConfig(model_name="BAAI/bge-small-en-v1.5"),
        postgres_config=PostgresConfig(),
        retrieval_config=RetrievalConfig(k=k))
    results = retriever.similarity_search_with_score(
            query=question,
            k=k,
            filter=None
        )
    response = []
    sections = []
    for doc, score in (results):
        if doc.metadata.get("section_h1",None):
            sections = [doc.metadata.get("section_h1", "")]
            curr = 1
            while curr<5:
                curr+=1
                if doc.metadata.get(f"section_h{curr}", None):
                    sections.append(doc.metadata.get(f"section_h{curr}", ""))
                else:
                    break
        response.append({
            "chunk_id": doc.metadata.get("chunk_index",0),
            "sections": sections,
            "text": doc.page_content,
            "source": doc.metadata.get("url",""),
            "score": 1-score,  # convert distance to similarity for better interpretability
        })
    return response


def hybrid_retrieve(question: str, k: int = 4) -> List[Dict[str, Any]]:
    """
    Replace with your hybrid retriever if available.
    If hybrid is not ready yet, you can temporarily point this to
    dense_retrieve() and still keep the graph structure intact.
    """
    return dense_retrieve(question, k)


def simple_rerank(query: str, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Cross-encoder reranker. Can use LLM-as-a-Judge for more complex logic later.
    """
    model = CrossEncoder(config.get("reranker", "BAAI/bge-reranker-base"))
    pairs = [(query, doc["text"]) for doc in docs]
    scores = model.predict(pairs)
    reranked_results = sorted(
    zip(docs, scores), key=lambda x: x[1], reverse=False # making it increasing order of relevance for LLM to rerank later, since LLMs generally prefer to see less relevant info first and then more relevant info
)
    return [doc for doc, score in reranked_results]


def generate_answer(question: str, docs: List[Dict[str, Any]]) -> str:
    """
    Generate an answer based on the retrieved documents.
    A good pattern is to build a prompt that includes:
    - the question
    - retrieved context
    - instructions to stay grounded
    - instructions to cite sources
    """
    joined_context = "\n\n".join(
        f"[{i+1}, with {doc['sections']} as Section co-ordinate,] {doc['text']}" for i, doc in enumerate(docs)
    )
    try:
        response = client.models.generate_content(model=model, contents=question+joined_context)
    except Exception as e:
        print(f"Error generating content: {e}")
        response = client.models.generate_content(model=backup_model, contents=question+joined_context)
    return (
        f"Grounded answer based on retrieved evidence:\n{joined_context}\n\n"
        f"For the question: {question}\n\n"
        f"{response.text}."
    )


def extract_citations(docs: List[Dict[str, Any]]) -> List[str]:
    """
    Pull source URLs or source ids from the documents used.
    """
    return [doc["source"] for doc in docs[::-1]] # reverse order to match answer generation


# -------------------------------------------------------------------
# Define LangGraph nodes
# -------------------------------------------------------------------
# Each node:
# - reads the current state
# - returns a partial update dictionary

def classify_node(state: RAGState) -> Dict[str, Any]:
    """
    Determine what kind of question this is.
    This lets the workflow adapt retrieval strategy.
    """
    question = state["question"]
    query_type = classify_query(question)
    print("Classified query type:", query_type)
    return {"query_type": query_type}


def retrieval_mode_node(state: RAGState) -> Dict[str, Any]:
    """
    Choose which retriever to use based on query type.
    """
    mode = choose_retrieval_mode(state["query_type"])
    needs_rerank = mode == "hybrid"
    print("Chosen retrieval mode:", mode, "Needs rerank?", needs_rerank)
    return {"retrieval_mode": mode, "needs_rerank": needs_rerank}


def retrieve_node(state: RAGState) -> Dict[str, Any]:
    """
    Execute retrieval using the chosen retrieval mode.
    """
    question = state["question"]
    mode = state["retrieval_mode"]

    if mode == "hybrid":
        docs = hybrid_retrieve(question, k=config.get("num_docs", config.get("num_docs", 4)))
    else:
        docs = dense_retrieve(question, k=config.get("num_docs", config.get("num_docs", 4)))    
    print(f"Retrieved {len(docs)} documents using {mode} retrieval.")
    return {"retrieved_docs": docs}


def add_context_node(state: RAGState) -> Dict[str, Any]:
    original_question = state["question"]
    retrieved = state.get("retrieved_docs", [])
    context_text = "\n\n".join(retrieved[0]["text"]) #add more context to the query
    return {"question": original_question + "\n\nContext:\n" + context_text, "original_question": original_question} #update the question in the state to include retrieved context before retrieval


def rerank_node(state: RAGState) -> Dict[str, Any]: # takes addtional context, retrieves docs and reranks them before passing to generation node
    """
    Optional reranking step.
    """
    docs = dense_retrieve(state["question"], k=config.get("num_docs", 4)*2) # retrieve more docs for reranking
    reranked = simple_rerank(state["question"], docs)[-config.get("num_docs", 4):][::-1] # then cut back down to desired number of docs after reranking
    print("Reranked documents.")
    return {"reranked_docs": reranked}


def generate_node(state: RAGState) -> Dict[str, Any]:
    """
    Draft the answer from the best available evidence.
    If reranked docs exist, use them; otherwise use retrieved docs.
    """
    question = state.get("original_question","") or state["question"]
    docs = state.get("reranked_docs") or state.get("retrieved_docs", [])
    answer = generate_answer(question, docs)
    citations = extract_citations(docs)
    print("Generated answer and extracted citations.")
    return {"draft_answer": answer, "citations": citations}


def finalize_node(state: RAGState) -> Dict[str, Any]:
    """
    Final packaging step.
    In later steps, can add groundedness checks, formatting,
    confidence notes, or a response schema here.
    """
    final_answer = state.get("draft_answer","Nothing generated.")
    print("Finalizing answer.")
    return  {"final_answer":final_answer, "citations": state.get("citations", [])} # For now, just pass everything through without changes


# -------------------------------------------------------------------
# Conditional routing functions
# -------------------------------------------------------------------
# LangGraph supports conditional edges so execution can branch
# depending on the current state. This is what makes the workflow
# "agentic" rather than a fixed sequence.

def route_after_retrieve_node(state: RAGState) -> Literal["generate_node","add_context_node"]:
    """
    This function exists mainly to make the routing explicit.
    Can later expand this to choose different retriever nodes.
    """
    if state["retrieval_mode"] == "hybrid":
        return "add_context_node"  # a future node that adds retrieved context to the question before retrieval
    return "generate_node"  

# -------------------------------------------------------------------
# Building the graph
# -------------------------------------------------------------------
# The workflow is:
# START
#   -> classify_node
#   -> retrieval_mode_node
#   -> retrieve_node
#   -> rerank_node or generate_node
#   |   |-> retrieve_node
#   -> finalize_node
#   -> END


def build_rag_graph():
    builder = StateGraph(RAGState)

    # Register nodes
    builder.add_node("classify_node", classify_node)
    builder.add_node("retrieval_mode_node", retrieval_mode_node)
    builder.add_node("retrieve_node", retrieve_node)
    builder.add_node("rerank_node", rerank_node)
    builder.add_node("generate_node", generate_node)
    builder.add_node("finalize_node", finalize_node)
    builder.add_node("add_context_node", add_context_node)

    # Fixed edges
    builder.add_edge(START, "classify_node")
    builder.add_edge("classify_node", "retrieval_mode_node")
    builder.add_edge("retrieval_mode_node", "retrieve_node")  

    # Explicit routing after retrieval 
    builder.add_conditional_edges(
        "retrieve_node",
        route_after_retrieve_node,
        {
            "add_context_node":"add_context_node", # for hybrid, add a context-building step before retrieval
            "generate_node": "generate_node",
        },
    )

    builder.add_edge("add_context_node", "rerank_node")  
    builder.add_edge("rerank_node", "generate_node")
    # Finalize and stop
    builder.add_edge("generate_node", "finalize_node")
    builder.add_edge("finalize_node", END)

    # Compile into an executable graph object
    graph = builder.compile()
    return graph


# -------------------------------------------------------------------
# Simple local test
# -------------------------------------------------------------------
# This lets you run the graph directly before wiring it into FastAPI.

if __name__ == "__main__":
    graph = build_rag_graph()
    print(time())
    result = graph.invoke(
        {
            "question": "Explain the fundamental commands of stategraph in langgraph and how it relates to broader agentic AI design patterns?"
        }
    )

    print("\nFINAL STATE:\n")
    for key, value in result.items():
        if key in ["citations", "final_answer"]:
            print(f"{key}: {value}\n")
    print(time())