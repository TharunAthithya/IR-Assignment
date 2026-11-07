"""Tamil Wikipedia Dump Extraction Module.

Extracts articles from the official Tamil Wikipedia XML dump into clean
JSON-lines files suitable for indexing.

Uses the `wikiextractor` library to strip MediaWiki markup, then applies
additional filtering to remove redirects, disambiguation stubs, and
empty/unusable articles.

Output format (one JSON object per line in each output file):
    {"id": 123, "title": "தமிழ்நாடு", "text": "தமிழ்நாடு இந்தியாவின்..."}
"""

import json
import os
import re
import sys
import time
import subprocess
import unicodedata
from typing import Generator, Dict, Any

# UTF-8 stdout
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Default paths
DEFAULT_DUMP_DIR = os.path.join("data", "wikipedia")
DEFAULT_OUTPUT_DIR = os.path.join("data", "extracted")

# Known dump filenames to look for (in order of preference)
KNOWN_DUMP_FILENAMES = [
    "tawiki-latest-pages-articles-multistream.xml.bz2",
    "tawiki-latest-pages-articles-multistream.xml",
    "tawiki-latest-pages-articles.xml.bz2",
    "tawiki-latest-pages-articles.xml",
]

# Patterns indicating non-article or redirect content
REDIRECT_PATTERN = re.compile(
    r"^\s*#(?:REDIRECT|redirect|வழிமாற்று|TAMA|தாமா)\s*\[\[",
    re.IGNORECASE | re.MULTILINE,
)
DISAMBIGUATION_PATTERNS = [
    "{{பக்கவழி நெறிப்படுத்தல்}}",
    "{{disambig",
    "{{Disambig",
    "{{சொற்பொருள் வேறுபாடு",
]


def find_dump_file(dump_dir: str) -> str:
    """Locates the Wikipedia dump file in the given directory.

    Args:
        dump_dir: Directory to search.

    Returns:
        Absolute path to the dump file.

    Raises:
        FileNotFoundError: If no recognized dump file is found.
    """
    for filename in KNOWN_DUMP_FILENAMES:
        path = os.path.join(dump_dir, filename)
        if os.path.exists(path):
            return os.path.abspath(path)

    # Fallback: look for any XML or XML.bz2 file
    for entry in os.listdir(dump_dir):
        if entry.endswith(".xml") or entry.endswith(".xml.bz2"):
            return os.path.abspath(os.path.join(dump_dir, entry))

    raise FileNotFoundError(
        f"No Wikipedia dump file found in '{dump_dir}'. "
        f"Expected one of: {KNOWN_DUMP_FILENAMES}"
    )


def extract_with_wikiextractor(dump_path: str, output_dir: str) -> str:
    """Runs WikiExtractor on the dump file to produce JSON output.

    Args:
        dump_path: Path to the XML (or .xml.bz2) dump.
        output_dir: Target directory for extracted files.

    Returns:
        Path to the extraction output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    extract_target = os.path.join(output_dir, "raw")

    print(f"Running WikiExtractor on: {dump_path}")
    print(f"Output directory: {extract_target}")
    print("(This may take several minutes for a large dump...)")

    cmd = [
        sys.executable, "-m", "wikiextractor.WikiExtractor",
        dump_path,
        "--json",
        "--output", extract_target,
        "--no-templates",
        "--processes", "1",  # Safe single-process for Windows compatibility
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if result.returncode != 0:
        print(f"WikiExtractor stderr:\n{result.stderr}")
        raise RuntimeError(
            f"WikiExtractor failed with return code {result.returncode}. "
            f"stderr: {result.stderr[:500]}"
        )

    print("WikiExtractor completed successfully.")
    return extract_target


def iter_extracted_articles(raw_dir: str) -> Generator[Dict[str, Any], None, None]:
    """Iterates over all articles produced by WikiExtractor.

    WikiExtractor creates subdirectories (AA, AB, ...) each containing
    files (wiki_00, wiki_01, ...) with one JSON object per line.

    Yields:
        Dict with keys: id, title, text
    """
    if not os.path.exists(raw_dir):
        raise FileNotFoundError(f"Extraction output not found at '{raw_dir}'")

    for dirpath, _dirnames, filenames in sorted(os.walk(raw_dir)):
        for filename in sorted(filenames):
            if not filename.startswith("wiki_"):
                continue
            filepath = os.path.join(dirpath, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        article = json.loads(line)
                        yield article
                    except json.JSONDecodeError:
                        continue


def is_valid_article(article: Dict[str, Any]) -> bool:
    """Checks whether an extracted article is suitable for indexing.

    Filters out:
    - Articles with no title or no text
    - Redirects
    - Disambiguation pages
    - Very short articles (< 50 characters of text)
    - Articles whose titles start with Wikipedia/Template/Category namespaces

    Args:
        article: Dict with 'title' and 'text' keys.

    Returns:
        True if the article should be included in the index.
    """
    title = (article.get("title") or "").strip()
    text = (article.get("text") or "").strip()

    if not title or not text:
        return False

    # Filter Wikipedia internal namespaces
    skip_prefixes = [
        "Wikipedia:", "விக்கிப்பீடியா:",
        "Template:", "வார்ப்புரு:",
        "Category:", "பகுப்பு:",
        "Module:", "கூறு:",
        "Help:", "உதவி:",
        "Portal:", "வலைவாசல்:",
        "MediaWiki:",
        "File:", "படிமம்:",
        "Draft:",
    ]
    for prefix in skip_prefixes:
        if title.startswith(prefix):
            return False

    # Filter redirects
    if REDIRECT_PATTERN.search(text):
        return False

    # Filter disambiguation pages
    text_lower = text.lower()
    for pattern in DISAMBIGUATION_PATTERNS:
        if pattern.lower() in text_lower:
            return False

    # Filter very short articles
    if len(text) < 50:
        return False

    return True


def clean_article_text(text: str) -> str:
    """Applies light cleaning to extracted article text.

    WikiExtractor already removes most markup. This handles residual artifacts.

    Args:
        text: Raw text from WikiExtractor.

    Returns:
        Cleaned text string.
    """
    # Apply Unicode NFC normalization
    text = unicodedata.normalize("NFC", text)

    # Remove residual HTML entities
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&quot;", '"')
    text = text.replace("&nbsp;", " ")

    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def extract_and_save(
    dump_dir: str = DEFAULT_DUMP_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    force: bool = False,
    max_articles: int = None,
) -> Dict[str, Any]:
    """Complete extraction pipeline: find dump → extract → filter → save.

    Args:
        dump_dir: Directory containing the Wikipedia dump file.
        output_dir: Directory to write extracted articles.
        force: If True, re-extract even if output already exists.
        max_articles: Maximum number of valid articles to extract.

    Returns:
        Dict with extraction statistics.
    """
    articles_file = os.path.join(output_dir, "articles.jsonl")

    # Check if extraction was already done
    if os.path.exists(articles_file) and not force:
        # Count existing articles
        count = 0
        with open(articles_file, "r", encoding="utf-8") as f:
            for _ in f:
                count += 1
        print(f"Extraction already completed. Found {count:,} articles in '{articles_file}'.")
        print("Use --force to re-extract.")
        return {"articles_file": articles_file, "total_articles": count, "skipped": True}

    start_time = time.perf_counter()

    # Step 1: Find the dump file
    dump_path = find_dump_file(dump_dir)
    print(f"Found Wikipedia dump: {dump_path}")
    file_size_mb = os.path.getsize(dump_path) / (1024 * 1024)
    print(f"Dump file size: {file_size_mb:.1f} MB")

    # Step 2: Run WikiExtractor
    raw_dir = extract_with_wikiextractor(dump_path, output_dir)

    # Step 3: Filter and consolidate into a single JSONL file
    print("\nFiltering and consolidating articles...")
    os.makedirs(output_dir, exist_ok=True)

    total_raw = 0
    total_valid = 0
    total_rejected = 0
    seen_titles = set()

    with open(articles_file, "w", encoding="utf-8") as out_f:
        for article in iter_extracted_articles(raw_dir):
            total_raw += 1

            if not is_valid_article(article):
                total_rejected += 1
                continue

            title = article["title"].strip()

            # Deduplicate by title
            title_key = unicodedata.normalize("NFC", title).lower()
            if title_key in seen_titles:
                total_rejected += 1
                continue
            seen_titles.add(title_key)

            # Clean text
            text = clean_article_text(article.get("text", ""))
            if len(text) < 50:
                total_rejected += 1
                continue

            # Write as JSONL
            record = {
                "id": total_valid,
                "title": title,
                "text": text,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            total_valid += 1

            if max_articles is not None and total_valid >= max_articles:
                print(f"Reached maximum articles limit ({max_articles:,}). Stopping extraction.")
                break

            if total_valid % 10000 == 0:
                print(f"  Processed {total_valid:,} valid articles...")




def load_articles(articles_file: str = None) -> Generator[Dict[str, Any], None, None]:
    """Loads extracted articles from the JSONL file.

    Args:
        articles_file: Path to articles.jsonl. Defaults to data/extracted/articles.jsonl.

    Yields:
        Dict with keys: id, title, text
    """
    if articles_file is None:
        articles_file = os.path.join(DEFAULT_OUTPUT_DIR, "articles.jsonl")

    if not os.path.exists(articles_file):
        raise FileNotFoundError(
            f"Extracted articles not found at '{articles_file}'. "
            f"Run 'python main.py extract' first."
        )

    with open(articles_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


if __name__ == "__main__":
    extract_and_save()
