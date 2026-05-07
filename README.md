# AgentEval-RAG

AgentEval-RAG is an open-source, production-style **agentic RAG backend** for grounded question answering over technical documentation. It is designed to show how a retrieval system evolves into a modular AI engineering platform: reproducible corpus ingestion, pgvector-based semantic retrieval, FastAPI service endpoints, and a LangGraph workflow for query routing, retrieval, reranking, and answer generation.

The current version already supports:
- Open-source document download and preprocessing
- Markdown-aware chunking with metadata preservation
- pgvector indexing over PostgreSQL
- Dense retrieval with optional score inspection and source filtering
- FastAPI endpoints for search and question answering
- LangGraph-based orchestration for multi-step RAG
- Basic test coverage for retrieval and API behavior

This project is built as a serious backend foundation for:
- grounded QA over open corpora
- retrieval benchmarking
- future hybrid / graph-enhanced retrieval experiments
- production-oriented AI engineering portfolios

---

## Why this project

Many RAG demos jump directly to “chat with documents.” In practice, the hard part is building a system that is:
- reproducible
- inspectable
- testable
- modular
- easy to extend into evaluation and deployment workflows

AgentEval-RAG takes a software-engineering-first approach. Instead of treating retrieval as a hidden helper, it makes ingestion, indexing, retrieval, API contracts, and orchestration explicit and reusable.

That makes the project useful both as:
- a GitHub portfolio project for AI engineering / agentic AI roles
- a backend foundation for future benchmark and product work

---

## Current status

### Implemented
- Project scoping and retrieval-first architecture
- Open-source documentation download scripts
- Corpus construction and JSONL chunk generation
- Metadata-aware chunking for technical documents
- Embedding pipeline
- PostgreSQL + pgvector vector store integration
- Similarity search and similarity-with-score retrieval
- Source-constrained retrieval options
- FastAPI retrieval API
- LangGraph RAG workflow with:
  - query classification
  - retrieval mode routing
  - retrieval
  - reranking path
  - answer generation
  - citation extraction
  - final answer packaging
- Retrieval and API tests

### In progress / next
- Better prompt and answer grounding controls
- stronger reranking and filtering logic
- benchmark runner over public RAG datasets
- hybrid retrieval
- graph-enhanced retrieval
- CI / deployment hardening
- richer observability and evaluation reports

---

## Repository structure

```text
AgentEval-RAG/
├── apps/
│   └── api/
│       ├── main.py
│       └── routes/
│           └── query.py
├── packages/
│   ├── agents/
│   │   └── graph.py
│   ├── ingestion/
│   │   └── build_corpus.py
│   └── retrieval/
│       ├── embed.py
│       ├── retriever.py
│       └── vector_store.py
├── scripts/
│   ├── download_docs.py
│   └── build_index.py
├── tests/
│   ├── test_api.py
│   └── test_search.py
├── configs/
│   ├── chunking.yaml
│   └── agent.yaml
├── data/
│   ├── raw/
│   └── processed/
└── README.md
```

---

## Architecture

The system has five layers.

### 1. Data acquisition
`scripts/download_docs.py` downloads public technical documentation pages and saves them as raw JSON records. The current corpus targets public documentation sources such as LangChain and FastAPI.

### 2. Corpus building
`packages/ingestion/build_corpus.py` converts raw documents into chunked JSONL suitable for retrieval. It uses markdown-aware section splitting, text normalization, deduplication, and metadata preservation.

Each chunk stores fields such as:
- title
- url
- chunk index
- section hierarchy
- text
- character count

### 3. Indexing
`scripts/build_index.py` reads processed JSONL chunks, converts them into LangChain `Document` objects, computes embeddings, and loads them into PostgreSQL with pgvector.

### 4. Retrieval
`packages/retrieval/` contains:
- embedding configuration and model setup
- vector store configuration
- retrieval wrappers for similarity search
- score-returning search for debugging and evaluation

### 5. Agentic QA orchestration
`packages/agents/graph.py` defines a LangGraph workflow that:
1. classifies the incoming query
2. chooses a retrieval strategy
3. retrieves relevant chunks
4. optionally reranks or expands context
5. generates an answer
6. extracts citations
7. returns a finalized response

This gives the project a real agentic structure instead of a single linear chain.

---

## LangGraph workflow

The current workflow is organized around explicit graph nodes rather than one monolithic function.

High-level flow:

```text
START
  → classify_node
  → retrieval_mode_node
  → retrieve_node
      ├── factual path → generate_node
      └── hybrid/multihop path → add_context_node → rerank_node → generate_node
  → finalize_node
  → END
```

Current routing logic includes:
- query classification such as factual / multihop / summary
- retrieval-mode selection
- a reranking branch
- answer generation with retrieved context
- citation extraction from retrieved documents

This structure is intentionally designed so that later work can add:
- query rewriting
- groundedness checks
- fallback logic
- benchmark instrumentation
- hybrid or graph retrieval nodes

---

## Tech stack

- **Python**
- **FastAPI**
- **LangGraph**
- **LangChain**
- **PostgreSQL + pgvector**
- **SentenceTransformers / embedding models**
- **Pydantic**
- **pytest**
- **BeautifulSoup / requests** for public-doc ingestion

---

## Data sources

This repository is intended to use only public, reproducible data sources.

Current corpus scripts target public technical documentation, including:
- FastAPI docs
- LangChain / LangGraph docs

Large corpora are not checked into the repo directly. Instead, the project provides scripts to download and preprocess them reproducibly.

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/shsaha2024/AgentEval-RAG.git
cd AgentEval-RAG
```

### 2. Create a Python environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

If you have a `requirements.txt`:

```bash
pip install -r requirements.txt
```

Or, if you are installing manually, make sure your environment includes the libraries used in the repo:
- fastapi
- uvicorn
- pydantic
- langgraph
- langchain
- langchain-postgres
- sqlalchemy
- psycopg
- sentence-transformers
- torch
- requests
- beautifulsoup4
- pyyaml
- tqdm
- pytest
- python-dotenv
- google-genai

### 4. Start PostgreSQL with pgvector

You need a PostgreSQL instance with the `vector` extension enabled.

Set environment variables as needed:

```bash
export POSTGRES_HOST=localhost
export POSTGRES_PORT=6024
export POSTGRES_DB=langchain
export POSTGRES_USER=langchain
export POSTGRES_PASSWORD=langchain
export PGVECTOR_COLLECTION=agentevaldocs
```

If you are using the LangGraph answer-generation path, also set:

```bash
export GEMINI_API_KEY=your_api_key_here
```

### 5. Download the public docs corpus

```bash
python scripts/download_docs.py
```

### 6. Build processed chunks

```bash
python -m packages.ingestion.build_corpus
```

If your local layout differs, run the corpus builder the way your repo is currently wired.

### 7. Build the pgvector index

```bash
python scripts/build_index.py --input data/processed/chunks.jsonl
```

### 8. Start the API

```bash
uvicorn apps.api.main:app --reload
```

Open:
- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/redoc`

---

## API endpoints

### `GET /health`

Simple health check.

Example response:

```json
{
  "status": "ok"
}
```

### `POST /query`

Retrieval API over the indexed corpus.

Example request:

```json
{
  "query": "How do I test FastAPI endpoints?",
  "k": 4,
  "mode": "similaritywithscores",
  "source": "fastapi"
}
```

Example response shape:

```json
{
  "query": "How do I test FastAPI endpoints?",
  "mode": "similaritywithscores",
  "k": 4,
  "total_hits": 4,
  "hits": [
    {
      "text": "....",
      "url": "https://fastapi.tiangolo.com/tutorial/testing/",
      "score": 0.82,
      "metadata": {
        "title": "Testing - FastAPI",
        "chunkindex": 3,
        "Sections": ["Tutorial", "Testing"]
      }
    }
  ]
}
```

### `POST /query` via LangGraph route

The repository also includes a LangGraph-backed question-answering path in `apps/api/routes/query.py`, where the graph is invoked and returns:
- question
- answer
- citations
- retrieval mode
- query type

That path is the beginning of the full agentic QA interface.

---

## Example local workflow

A typical local development loop is:

1. Download or refresh public docs
2. Build processed chunks
3. Rebuild the vector index
4. Run retrieval checks from CLI or tests
5. Start the FastAPI app
6. Query the API
7. Iterate on chunking, retrieval, and graph behavior

This makes the system easier to debug than a generation-first workflow.

---

## Testing

Run all tests:

```bash
pytest -q
```

Run API tests only:

```bash
pytest tests/test_api.py -q
```

Run retrieval tests only:

```bash
pytest tests/test_search.py -q
```

The current tests focus on:
- retrieval behavior
- API contract validation
- request handling
- search output structure

---

## Design decisions

### Retrieval-first foundation
The project was intentionally built from the bottom up: ingestion, chunking, indexing, retrieval, and only then answer generation. This makes failures easier to localize and keeps the retrieval layer reusable.

### API-first boundary
Retrieval and QA are exposed as service interfaces rather than buried inside notebooks or scripts. That makes the repo easier to extend into evaluation pipelines, frontends, and benchmark harnesses.

### Explicit graph orchestration
Using LangGraph keeps the control flow visible. Query classification, routing, retrieval, reranking, and generation are represented as graph nodes, which is a better fit for agentic AI systems than a single chain.

### Open-data reproducibility
The project is designed around public technical corpora and script-based data acquisition, so others can reproduce the same corpus and rebuild the index locally.

---

## Current limitations

This is still an actively developing backend, not yet a polished end-user product.

Current limitations include:
- hybrid retrieval is only partially implemented
- reranking logic is still evolving
- groundedness checks are not yet robust
- benchmark runners are not yet integrated
- deployment and CI hardening are still pending
- observability is still basic

These are deliberate next steps rather than missing foundations.

---

## Roadmap

### Near term
- strengthen prompt design and answer grounding
- improve reranking and retrieval diagnostics
- standardize config handling
- clean up source filtering and response schemas

### Next phase
- add benchmark runners on public RAG datasets
- compare dense vs hybrid retrieval
- log latency, retrieval quality, and answer quality
- add fallback logic and confidence annotations

### Later phase
- add graph-enhanced retrieval
- add CI / GitHub Actions
- containerize local deployment
- add evaluation dashboards and experiment reports

## References

- [FastAPI documentation](https://fastapi.tiangolo.com/)
- [FastAPI testing](https://fastapi.tiangolo.com/tutorial/testing/)
- [LangGraph workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents)
- [LangGraph Python reference](https://reference.langchain.com/python/langgraph/overview)
- [LangChain PGVector integration](https://docs.langchain.com/oss/python/integrations/vectorstores/pgvector)
- [pgvector-python](https://github.com/pgvector/pgvector-python)

