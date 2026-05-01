# scripts/download_langgraph_docs.py

# Standard library imports
# json: save each downloaded page as structured JSON
# re: create safe file names / IDs from URLs
# time: pause between requests so we don't hit the server too aggressively
from pathlib import Path
from datetime import datetime, timezone
import json
import re
import time

# Third-party libraries
# requests: download web pages
# BeautifulSoup: parse HTML and extract readable text
import requests
from bs4 import BeautifulSoup


# Folder where raw LangGraph pages will be saved
OUT_DIR = Path("data/raw/langgraph")
if not OUT_DIR.exists():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

# LangChain docs publish an llms.txt index, which is an easy way to discover docs pages
LLMS_TXT_URL = "https://docs.langchain.com/llms.txt"

# A basic user-agent helps avoid being treated like a suspicious bot
HEADERS = {"User-Agent": "Mozilla/5.0"}


def slugify(text: str) -> str:
    """
    Convert a string into a safe file-friendly ID.

    Example:
    'overview/get-started' -> 'overview_get_started'
    """
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def get_langgraph_urls() -> list[str]:
    """
    Download llms.txt and keep only URLs that belong to LangGraph docs.

    Why this function exists:
    We want URL discovery to be separate from page downloading.
    That makes the script easier to debug.
    """
    response = requests.get(LLMS_TXT_URL, headers=HEADERS, timeout=30)
    response.raise_for_status()  # Stop immediately if the request failed

    urls = []

    # llms.txt is plain text, so we scan it line by line
    for line in response.text.splitlines():
        line = line.strip()

        # Keep only actual URLs that point to LangGraph pages
        if line.startswith("http") and "/langgraph/" in line:
            urls.append(line)

    # Remove duplicates and sort for stable output
    return sorted(set(urls))


def extract_text(html: str) -> tuple[str, str]:
    """
    Extract a page title and main readable text from HTML.

    We remove obvious non-content tags like script, style, footer, etc.
    Then we prefer <main> content if it exists.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove elements that usually add noise instead of useful documentation text
    for tag in soup(["script", "style", "nav", "footer", "aside"]):
        tag.decompose()

    # Read the page title if available
    title = soup.title.get_text(" ", strip=True) if soup.title else ""

    # Many docs pages place the real content inside <main>
    main = soup.find("main")

    if main:
        # separator="\n" keeps sections separated instead of smashing text together
        text = main.get_text("\n", strip=True)
    else:
        # Fallback: take all visible page text
        text = soup.get_text("\n", strip=True)

    # Collapse very large runs of blank lines into cleaner spacing
    text = re.sub(r"\n{2,}", "\n\n", text)

    return title, text


def save_record(url: str, title: str, text: str):
    """
    Save one documentation page as one JSON file.

    Keeping one file per page is useful because:
    1. it is easy to inspect manually,
    2. failures affect only one page,
    3. later we can merge everything into JSONL.
    """
    # Build a stable ID from the part of the URL after /langgraph/
    tail = url.split("/langgraph/")[-1] or "index"
    doc_id = slugify(tail)

    record = {
        "doc_id": doc_id,
        "source": "langgraph",
        "url": url,
        "title": title,
        "text": text,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
    }

    out_path = OUT_DIR / f"{doc_id}.json"

    # ensure_ascii=False keeps normal Unicode characters readable
    # indent=2 makes the JSON easy to inspect by eye
    out_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )




OUT_DIR_fast = Path("data/raw/fastapi")
if not OUT_DIR_fast.exists():
    OUT_DIR_fast.mkdir(parents=True, exist_ok=True)

HEADERS_fast = {"User-Agent": "Mozilla/5.0"}

URLS = [
    "https://fastapi.tiangolo.com/",
    "https://fastapi.tiangolo.com/tutorial/",
    "https://fastapi.tiangolo.com/tutorial/metadata/",
    "https://fastapi.tiangolo.com/tutorial/testing/",
    "https://fastapi.tiangolo.com/deployment/",
    "https://fastapi.tiangolo.com/reference/openapi/docs/",
]

def save_record_fastapi(url: str, title: str, text: str):
    tail = url.replace("https://fastapi.tiangolo.com/", "").strip("/") or "home"
    doc_id = "fastapi_" + slugify(tail)
    record = {
        "doc_id": doc_id,
        "source": "fastapi",
        "url": url,
        "title": title,
        "text": text,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
    }
    out_path = OUT_DIR_fast / f"{doc_id}.json"
    out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

def main():
    """
    Main program flow:
    1. Discover LangGraph URLs
    2. Download each page
    3. Extract text
    4. Save JSON
    """
    urls = get_langgraph_urls()
    print(f"Found {len(urls)} LangGraph URLs")

    for i, url in enumerate(urls, start=1):
        try:
            # Download one docs page
            response = requests.get(url, headers=HEADERS, timeout=30)
            response.raise_for_status()

            # Parse HTML into title + cleaned text
            title, text = extract_text(response.text)

            # Skip pages that are too short to be useful
            # This filters out broken pages or pages with mostly boilerplate
            if len(text) < 200:
                print(f"Skipping short page: {url}")
                continue

            # Save the cleaned result
            save_record(url, title, text)
            print(f"[{i}/{len(urls)}] Saved: {url}")

            # Small pause to avoid hammering the site
            time.sleep(0.5)

        except Exception as e:
            # We log the failure and continue, instead of crashing the whole run
            print(f"Failed {url}: {e}")

    for i, url in enumerate(URLS, 1):
        try:
            r = requests.get(url, headers=HEADERS_fast, timeout=30)
            r.raise_for_status()
            title, text = extract_text(r.text)
            if len(text) < 200:
                print(f"Skipping short page: {url}")
                continue
            save_record_fastapi(url, title, text)
            print(f"[{i}/{len(URLS)}] Saved: {url}")
            time.sleep(0.5)
        except Exception as e:
            print(f"Failed {url}: {e}")


# Python runs this block only when the file is executed directly
if __name__ == "__main__":
    main()