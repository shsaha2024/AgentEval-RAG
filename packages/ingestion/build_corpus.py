# Purpose:
# Process JSON files where each record has a "text" field containing
# page text already extracted from HTML.
#
# Input:
#   data/raw/**/*.json
#
# Supports JSON shape   {"text": "...", ...}
#
# Output:
#   data/processed/chunks.jsonl
#
# Output format:
#   One JSON object per line (JSONL / newline-delimited JSON).
#   This is convenient for streaming, debugging, and later indexing.

from pathlib import Path
import json
import hashlib
import re
from typing import Dict, Any, Iterable, List
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
import yaml

# ----------------------------
# Configuration
# ----------------------------

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_FILE = OUT_DIR / "chunks.jsonl"
if OUT_FILE.exists():
    OUT_FILE.unlink()

# Only split on markdown headers.
# You can reduce or expand the list depending on your corpus structure.
HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
]
config = yaml.safe_load(open("configs/chunking.yaml", 'r'))

# Skip tiny chunks that are usually noise.
MIN_CHARS = config["MIN_CHARS"]


# ----------------------------
# Utility functions
# ----------------------------

def stable_hash(text: str) -> str:
    """
    Create a short stable hash for IDs and deduplication.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def normalize_text(text: str) -> str:
    """
    Normalize whitespace while preserving basic paragraph structure.

    This helps make chunk text more consistent before splitting.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def write_jsonl(records: Iterable[Dict[str, Any]], out_path: Path) -> None:
    """
    Write newline-delimited JSON.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ----------------------------
# JSON loading
# ----------------------------

def load_json_records(path: Path) -> List[Dict[str, Any]]:
    """
    Load a JSON file and return a list of records.

    Supported formats:
    - one dict
    - a list of dicts

    Every returned record should be a dict.
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        return [data]

    # if isinstance(data, list):
    #     return [x for x in data if isinstance(x, dict)]

    return []


def iter_raw_documents(raw_dir: Path) -> Iterable[Dict[str, Any]]:
    """
    Iterate through all JSON files in data/raw and extract records
    that contain a 'text' field.

    Expected raw record example:
    {
        "url": "...",
        "title": "...",
        "text": "# Heading\\n\\nSome markdown-like content..."
    }
    """
    for path in raw_dir.rglob("*.json"):
        try:
            records = load_json_records(path)
        except Exception as e:
            print(f"[WARN] Failed to read {path}: {e}")
            continue

        for i, record in enumerate(records):
            text = record.get("text", "")
            text = re.sub(r'<.*?>', '', text)
            text.strip()
            if not isinstance(text, str):
                continue

            text = normalize_text(text)
            if not text:
                continue

            # Create a stable document id from file path + record index.
            doc_id = stable_hash(f"{path}|{i}")

            yield {
                "doc_id": doc_id,
                "record_index": i,
                "source_path": str(path),
                "title": record.get("title"),
                "url": record.get("url"),
                "text": text,
            }


# ----------------------------
# Markdown section chunking
# ----------------------------

def split_markdown(text: str) -> List[Dict[str, Any]]:
    """
    Split markdown text into section chunks using markdown headings.

    LangChain's MarkdownHeaderTextSplitter returns documents with:
    - page_content
    - metadata containing matched header values

    Example metadata:
    {
        "h1": "Introduction",
        "h2": "Installation"
    }
    """
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,  # keep headers inside the chunk text for better retrieval context
    )
    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunk_size"],
        chunk_overlap=  config["chunk_overlap"],
        separators=["\n\n", "\n", ". ", " ", ""],
)
    docs = header_splitter.split_text(text)
    docs = size_splitter.split_documents(docs)

    chunks = []
    for d in docs:
        chunk_text = normalize_text(d.page_content)
        if len(chunk_text) < MIN_CHARS:
            continue

        metadata = d.metadata if d.metadata else {}

        chunks.append({
            "text": chunk_text,
            "section_path": metadata,  # e.g. {"h1": "...", "h2": "..."}
        })

    return chunks


# ----------------------------
# Main processing logic
# ----------------------------

def process_documents() -> List[Dict[str, Any]]:
    """
    Process all raw JSON documents into section-based chunks.

    Output schema per chunk:
    {
        "chunk_id": ...,
        "doc_id": ...,
        "source_path": ...,
        "title": ...,
        "url": ...,
        "chunk_index": ...,
        "section_h1": ...,
        "section_h2": ...,
        "section_h3": ...,
        "section_h4": ...,
        "text": ...,
        "char_count": ...
    }
    """
    all_chunks = []
    seen_text_hashes = set()

    for doc in iter_raw_documents(RAW_DIR):
        section_chunks = split_markdown(doc["text"])

        for idx, chunk in enumerate(section_chunks):
            chunk_text = chunk["text"]

            # Exact deduplication across the corpus.
            text_hash = stable_hash(chunk_text)
            if text_hash in seen_text_hashes:
                continue
            seen_text_hashes.add(text_hash)

            section_path = chunk["section_path"]

            chunk_record = {
                "chunk_id": f"{doc['doc_id']}__{idx:05d}",
                "doc_id": doc["doc_id"],
                "source_path": doc["source_path"],
                "title": doc["title"],
                "url": doc["url"],
                "chunk_index": idx,
                "section_h1": section_path.get("h1"),
                "section_h2": section_path.get("h2"),
                "section_h3": section_path.get("h3"),
                "section_h4": section_path.get("h4"),
                "text": chunk_text,
                "char_count": len(chunk_text),
            }

            all_chunks.append(chunk_record)

    return all_chunks


# ----------------------------
# Entry point
# ----------------------------

def main() -> None:
    print("[INFO] Processing JSON documents with markdown section chunking...")
    chunks = process_documents()
    print(f"[INFO] Created {len(chunks)} chunks")

    write_jsonl(chunks, OUT_FILE)
    print(f"[INFO] Wrote output to {OUT_FILE}")


if __name__ == "__main__":
    main()