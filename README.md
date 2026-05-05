# AgentEval-RAG

A modular retrieval-first RAG backend for experimenting with document ingestion, semantic search, and API-first retrieval over open-source corpora.

> Current status: retrieval MVP complete. The project supports document ingestion, chunking, indexing, similarity search, and a FastAPI service layer for querying retrieved chunks. Full answer generation and agentic orchestration are planned next.

## Overview

AgentEval-RAG is an open-source project for building a production-style retrieval layer for RAG systems over technical documents and other public corpora. The current implementation focuses on the part that is most often under-engineered in early RAG projects: clean ingestion, reproducible indexing, search variants, and a typed API surface.

The project is designed to evolve in phases:

- Phase 1: ingestion and semantic retrieval
- Phase 2: API-first retrieval service
- Phase 3: grounded answer generation
- Phase 4: agentic orchestration and evaluation

At the current stage, this repository is intentionally retrieval-centric. It exposes the retriever as a service before adding generation so that the data and search layers are stable, testable, and reusable.

## Current Features

- Document ingestion pipeline for open-source corpora
- Chunking and metadata preservation
- Vector indexing for semantic retrieval
- Similarity search with configurable top-k
- Search variants such as score-returning search and source-constrained search
- FastAPI retrieval API with typed request and response models
- Unit and API-level test coverage for the retrieval layer
- OpenAPI docs generated automatically by FastAPI

## Project Status

### Completed

- Step 1: project scope and retrieval-first MVP definition
- Step 2: repository structure
- Step 3: open-source corpus and benchmark data selection
- Step 4: ingestion and chunking pipeline
- Step 5: baseline vector retrieval
- Step 8: FastAPI API layer for retrieval

### Not yet implemented

- Step 6: hybrid retrieval
- Step 7: citation-grounded answer generation
- Step 9+: LangGraph orchestration, benchmarking workflows, and production hardening

## Why this project

Many RAG demos jump directly to generation, but generation quality depends heavily on ingestion quality, chunking strategy, retrieval relevance, and API design. This repository starts by making retrieval robust and observable before adding generation and agents.

That design choice has a few benefits:

- It isolates search quality from generation quality
- It makes retrieval testable without model-dependent answer variability
- It creates a reusable backend that can later be called by a UI, a benchmark runner, or a LangGraph workflow
- It keeps the early system lightweight and easier to debug

## Repository Structure

```text
.
├── apps/
│   └── api/
│       ├── main.py
│       └── ...
├── packages/
│   ├── ingestion/
│   │   ├── loaders.py
│   │   ├── chunking.py
│   │   └── ...
│   ├── retrieval/
│   │   ├── service.py
│   │   ├── vector_store.py
│   │   └── ...
│   └── common/
│       └── ...
├── configs/
│   ├── chunking.yaml
│   ├── retrieval.yaml
│   └── ...
├── scripts/
│   ├── download_docs.py
│   ├── build_index.py
│   └── ...
├── tests/
│   ├── test_search.py
│   ├── test_api.py
│   └── ...
├── data/
│   ├── raw/
│   ├── processed/
│   └── ...
└── README.md
```

## Architecture

At the current stage, the system is organized into three main layers:

### 1. Ingestion layer

This layer loads raw open-source documents, normalizes them, chunks them into retrieval-friendly segments, and preserves useful metadata such as source name, section, and chunk ID.

Responsibilities:
- load public documents
- clean and normalize text
- split into chunks
- attach metadata
- write processed artifacts for indexing

### 2. Retrieval layer

This layer builds and queries the vector index. It is responsible for semantic search and retrieval variations used by tests and the API.

Responsibilities:
- embed processed chunks
- store vectors in the chosen backend
- run similarity search
- support retrieval variants such as:
  - plain similarity search
  - similarity search with scores
  - source-filtered search

### 3. API layer

This layer exposes retrieval through FastAPI. It does not generate answers yet. Instead, it provides a stable service boundary around the retriever.

Responsibilities:
- validate request bodies
- call retrieval functions
- normalize outputs into public response schemas
- expose health and query endpoints
- auto-generate OpenAPI docs

## What the API does today

The current API is a retrieval API, not a full QA API.

### Implemented endpoints

- `GET /health`  
  Health check for service status

- `POST /query`  
  Run semantic retrieval over the indexed corpus

### Typical `POST /query` request

```json
{
  "query": "What does FastAPI use for request validation?",
  "k": 3,
  "mode": "similarity"
}
```

### Example response

```json
{
  "query": "What does FastAPI use for request validation?",
  "mode": "similarity",
  "k": 3,
  "total_hits": 3,
  "hits": [
    {
      "chunk_id": "chunk_001",
      "text": "FastAPI uses Pydantic models for request body parsing and validation.",
      "source": "fastapi_docs",
      "score": null,
      "metadata": {
        "section": "request-body"
      }
    }
  ]
}
```

### Supported retrieval modes

The exact modes depend on the retriever implementation in this repository, but current testing and API design assume support for variants such as:

- `similarity`
- `similarity_with_scores`

If source filtering is implemented, the request may also support:

- `source`

## Data

This project is intended to use only open-source or publicly accessible corpora.

Examples of suitable data sources include:
- public technical documentation
- open benchmark corpora
- public machine learning or framework documentation
- openly licensed research summaries or structured QA datasets

Large raw datasets are not stored directly in the repository. Instead, scripts should be provided to download and preprocess them reproducibly.

## Getting Started

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd AgentEval-RAG
```

### 2. Create an environment

```bash
# Fill in your preferred environment setup here
```

### 3. Install dependencies

```bash
# Fill in your installation command here
```

### 4. Download and preprocess the corpus

```bash
# Fill in your corpus download command here
# Example:
# python scripts/download_docs.py
# python scripts/preprocess_docs.py
```

### 5. Build the vector index

```bash
# Fill in your indexing command here
# Example:
# python scripts/build_index.py
```

### 6. Run the FastAPI app

```bash
# Fill in your command here
# Example:
# uvicorn apps.api.main:app --reload
```

### 7. Open the docs

After the server starts, visit:

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/redoc`

## Development Workflow

A typical local workflow looks like this:

1. Download or refresh the source corpus
2. Run preprocessing and chunking
3. Rebuild the vector index if needed
4. Start the API server
5. Test `/health`
6. Test `/query`
7. Run the test suite

This makes retrieval changes easy to validate before generation is added.

## Testing

The project currently emphasizes retrieval-layer and API-layer testing.

### Run all tests

```bash
# Fill in your test command here
# Example:
# pytest -q
```

### Run retrieval tests only

```bash
# Fill in your test command here
# Example:
# pytest tests/test_search.py -q
```

### Run API tests only

```bash
# Fill in your test command here
# Example:
# pytest tests/test_api.py -q
```

### What should pass

At this stage, successful testing should confirm:

- the app imports without crashing
- `/health` returns a successful response
- `/query` accepts valid request bodies
- invalid request bodies are rejected cleanly
- the retriever returns correctly structured hits
- score-returning retrieval modes behave as expected
- fixed-source or constrained retrieval works if implemented

## Example Use Cases

This repository is currently useful for:

- testing chunking and indexing strategies
- evaluating retrieval behavior before adding generation
- exposing document search via an API
- building a retrieval backend for a later RAG assistant
- comparing search variants in a controlled setup

## Design Decisions

### Retrieval-first development

This project intentionally implements retrieval before answer generation. That keeps the system debuggable and makes failures easier to localize.

### API-first interface

Retrieval is exposed through FastAPI rather than staying hidden inside scripts or tests. This makes the backend reusable by future components such as:

- an answer generation module
- a benchmark harness
- a frontend
- a LangGraph workflow

### Typed request and response contracts

The API uses Pydantic-backed request and response models so that:
- invalid input is rejected automatically
- output shapes remain stable
- docs are generated from the code
- downstream consumers have a predictable contract

## Current Limitations

The current version is intentionally incomplete.

Not implemented yet:
- hybrid retrieval
- reranking
- answer generation
- citations over final answers
- LangGraph orchestration
- benchmark dashboards
- deployment hardening

This repository should currently be viewed as a strong retrieval service foundation rather than a complete end-user RAG assistant.

## Roadmap

### Near-term

- add hybrid retrieval
- add reranking
- standardize metadata filters
- improve retrieval diagnostics

### Next phase

- add grounded answer generation
- return citations tied to retrieved chunks
- add confidence and fallback logic

### Later phase

- convert the pipeline into a LangGraph workflow
- add benchmark runners
- compare retrieval modes on public datasets
- add deployment and CI hardening

## Suggested Benchmarks for Future Work

Once answer generation and evaluation are added, the project can be extended to benchmark:
- retrieval quality across chunking settings
- vector vs hybrid retrieval
- groundedness of generated answers
- latency and throughput tradeoffs
- graph-augmented retrieval on public benchmarks

## Contributing

Contributions are welcome, especially in the following areas:

- ingestion adapters for public corpora
- retrieval backends and filters
- evaluation scripts
- benchmark integrations
- API improvements
- documentation and examples

For substantial changes, open an issue first describing the proposed change and expected impact.

## Notes for Recruiters / Reviewers

This repository is being built in staged milestones. The current version demonstrates:

- reproducible document ingestion
- semantic retrieval over open corpora
- testable retrieval variants
- a typed FastAPI retrieval service
- a software-engineering-first approach to building toward RAG

The next major milestone is grounded answer generation on top of the existing retrieval API.

## References

- FastAPI request body docs: https://fastapi.tiangolo.com/tutorial/body/
- FastAPI testing docs: https://fastapi.tiangolo.com/tutorial/testing/
- FastAPI response model docs: https://fastapi.tiangolo.com/tutorial/response-model/
- LangGraph workflow docs: https://docs.langchain.com/oss/python/langgraph/workflows-agents

