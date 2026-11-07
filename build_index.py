"""Disk-Based Positional Inverted Index Construction Module.

This module constructs a positional inverted index from extracted Tamil Wikipedia
articles, using a disk-based storage architecture that satisfies the IR assignment
requirement of keeping vocabulary in RAM and postings on secondary storage.

Architecture:
    1. vocabulary.json — Loaded into RAM at query time. Contains each term's
       Document Frequency (df), Collection Frequency (cf), byte offset into
       postings.dat, and byte length of the serialized postings.

    2. postings.dat — Stored on disk (secondary storage). Contains the serialized
       positional postings for every term. Each term's postings are a JSON object
       mapping document ID strings to arrays of zero-based positions. Access is
       via random file seek using offset+length from the vocabulary.

    3. doc_meta.json — Document metadata mapping doc_id to article title.
       Loaded into RAM at query time for result display.

    4. metadata.json — Index construction statistics and provenance information.

Construction Algorithm:
    Phase 1: Single-pass document scan — accumulate positional postings in memory.
    Phase 2: Sort vocabulary and write each term's postings to postings.dat,
             recording byte offsets.
    Phase 3: Write vocabulary.json with df, cf, offset, length.
    Phase 4: Write doc_meta.json and metadata.json.

Zero-based positions are used throughout. Combined document text is
"{title} {text}", so title tokens occupy the first positions.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple

# Reconfigure standard streams to UTF-8
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from preprocess import tokenize
from extract_wiki import load_articles, DEFAULT_OUTPUT_DIR


def build_and_save(
    articles_path: str = None,
    output_dir: str = "indexes",
) -> Dict[str, Any]:
    """High-level pipeline: load articles → build positional index → write to disk.

    Args:
        articles_path: Path to extracted articles JSONL file.
                       Defaults to data/extracted/articles.jsonl.
        output_dir: Directory to write index files. Defaults to 'indexes'.

    Returns:
        Dict of build statistics.
    """
    if articles_path is None:
        articles_path = os.path.join(DEFAULT_OUTPUT_DIR, "articles.jsonl")

    if not os.path.exists(articles_path):
        raise FileNotFoundError(
            f"Extracted articles not found at '{articles_path}'. "
            f"Run 'python main.py extract' first."
        )

    print("=" * 60)
    print("POSITIONAL INVERTED INDEX — DISK-BASED CONSTRUCTION")
    print("=" * 60)

    start_time = time.perf_counter()

    # -------------------------------------------------------------------------
    # Phase 1: Single-pass document scan — accumulate positional postings
    # -------------------------------------------------------------------------
    print(f"Phase 1: Loading and tokenizing articles from '{articles_path}'...")

    # In-memory structures during build
    positional_index: Dict[str, Dict[str, List[int]]] = {}
    doc_meta: Dict[str, str] = {}  # doc_id_str -> title
    total_tokens = 0
    total_docs = 0
    total_positional_entries = 0  # sum of all position list lengths

    for article in load_articles(articles_path):
        doc_id = article["id"]
        doc_id_str = str(doc_id)
        title = article.get("title", "")
        text = article.get("text", "")

        # Store document metadata
        doc_meta[doc_id_str] = title

        # Combine title and text — title tokens come first
        combined_text = f"{title} {text}"
        tokens = tokenize(combined_text)
        total_tokens += len(tokens)
        total_docs += 1

        # Accumulate positional postings
        for pos, term in enumerate(tokens):
            if term not in positional_index:
                positional_index[term] = {}
            doc_postings = positional_index[term]
            if doc_id_str not in doc_postings:
                doc_postings[doc_id_str] = [pos]
            else:
                doc_postings[doc_id_str].append(pos)

        if total_docs % 20000 == 0:
            print(f"  Processed {total_docs:,} documents, "
                  f"{len(positional_index):,} unique terms so far...")

    vocab_size = len(positional_index)
    print(f"\n  Phase 1 complete:")
    print(f"    Total documents:    {total_docs:,}")
    print(f"    Total tokens:       {total_tokens:,}")
    print(f"    Vocabulary size:    {vocab_size:,}")

    # -------------------------------------------------------------------------
    # Phase 2: Write postings.dat — serialize each term's postings to disk
    # -------------------------------------------------------------------------
    print(f"\nPhase 2: Writing positional postings to disk...")

    os.makedirs(output_dir, exist_ok=True)
    postings_path = os.path.join(output_dir, "postings.dat")

    # Vocabulary structure: term -> {df, cf, offset, length}
    vocabulary: Dict[str, Dict[str, int]] = {}

    # Sort terms for deterministic output
    sorted_terms = sorted(positional_index.keys())

    with open(postings_path, "wb") as postings_file:
        for term in sorted_terms:
            doc_map = positional_index[term]

            # Calculate df and cf
            df = len(doc_map)
            cf = sum(len(positions) for positions in doc_map.values())
            total_positional_entries += cf

            # Serialize this term's postings as JSON bytes (UTF-8)
            # Format: {"doc_id": [pos0, pos1, ...], ...}
            postings_json = json.dumps(doc_map, ensure_ascii=False)
            postings_bytes = postings_json.encode("utf-8")

            # Record byte offset and length
            offset = postings_file.tell()
            length = len(postings_bytes)

            # Write to disk
            postings_file.write(postings_bytes)

            # Store in vocabulary
            vocabulary[term] = {
                "df": df,
                "cf": cf,
                "offset": offset,
                "length": length,
            }

    postings_file_size = os.path.getsize(postings_path)
    print(f"  Written {len(sorted_terms):,} terms to '{postings_path}'")
    print(f"  Postings file size: {postings_file_size / (1024*1024):.2f} MB")

    # Free the large in-memory positional index
    del positional_index
    del sorted_terms

    # -------------------------------------------------------------------------
    # Phase 3: Write vocabulary.json
    # -------------------------------------------------------------------------
    print(f"\nPhase 3: Writing vocabulary index...")

    vocab_path = os.path.join(output_dir, "vocabulary.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocabulary, f, ensure_ascii=False)
    vocab_file_size = os.path.getsize(vocab_path)
    print(f"  Vocabulary file size: {vocab_file_size / (1024*1024):.2f} MB")

    # -------------------------------------------------------------------------
    # Phase 4: Write document metadata
    # -------------------------------------------------------------------------
    print(f"\nPhase 4: Writing document metadata...")

    doc_meta_path = os.path.join(output_dir, "doc_meta.json")
    with open(doc_meta_path, "w", encoding="utf-8") as f:
        json.dump(doc_meta, f, ensure_ascii=False)
    doc_meta_size = os.path.getsize(doc_meta_path)
    print(f"  Document metadata size: {doc_meta_size / (1024*1024):.2f} MB")

    # -------------------------------------------------------------------------
    # Phase 5: Write metadata.json
    # -------------------------------------------------------------------------
    duration = time.perf_counter() - start_time

    # Compute total term-document postings (number of (term, doc) pairs)
    total_term_doc_postings = sum(v["df"] for v in vocabulary.values())

    stats = {
        "dataset_name": "Tamil Wikipedia",
        "dataset_source": "https://dumps.wikimedia.org/tawiki/",
        "articles_file": articles_path,
        "indexing_timestamp": datetime.now().isoformat(),
        "total_documents": total_docs,
        "vocabulary_size": vocab_size,
        "total_tokens": total_tokens,
        "total_term_doc_postings": total_term_doc_postings,
        "total_positional_entries": total_positional_entries,
        "average_doc_length": round(total_tokens / total_docs, 2) if total_docs else 0,
        "index_construction_seconds": round(duration, 3),
        "preprocessing": {
            "normalization": "Unicode NFC",
            "case_folding": "Lowercase (English only; Tamil unaffected)",
            "token_pattern": r"[\p{L}\p{N}][\p{L}\p{M}\p{N}]*",
            "stopwords_removed": False,
            "stemming_applied": False,
            "combining_marks_preserved": True,
            "positions": "Zero-based",
        },
        "storage": {
            "vocabulary_file": "vocabulary.json",
            "vocabulary_size_bytes": vocab_file_size,
            "vocabulary_size_mb": round(vocab_file_size / (1024*1024), 2),
            "postings_file": "postings.dat",
            "postings_size_bytes": postings_file_size,
            "postings_size_mb": round(postings_file_size / (1024*1024), 2),
            "doc_meta_file": "doc_meta.json",
            "doc_meta_size_bytes": doc_meta_size,
            "doc_meta_size_mb": round(doc_meta_size / (1024*1024), 2),
            "architecture": "Vocabulary in RAM, postings on secondary storage (disk)",
            "postings_format": "JSON-encoded UTF-8 bytes, accessed via byte offset + length",
        },
    }

    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\nPhase 5: Metadata written to '{metadata_path}'")

    print("\n" + "=" * 60)
    print("INDEX CONSTRUCTION COMPLETE")
    print("=" * 60)
    print(f"  Total Documents:           {total_docs:,}")
    print(f"  Vocabulary Size:           {vocab_size:,} unique terms")
    print(f"  Total Tokens:              {total_tokens:,}")
    print(f"  Term-Document Postings:    {total_term_doc_postings:,}")
    print(f"  Positional Entries:        {total_positional_entries:,}")
    print(f"  Average Doc Length:        {stats['average_doc_length']:.1f} tokens")
    print(f"  Construction Time:         {duration:.2f} seconds")
    print(f"  Vocabulary File:           {vocab_file_size/(1024*1024):.2f} MB")
    print(f"  Postings File:             {postings_file_size/(1024*1024):.2f} MB")
    print(f"  Doc Metadata File:         {doc_meta_size/(1024*1024):.2f} MB")
    print("=" * 60)

    return stats


def main():
    """Command-line interface for index building."""
    parser = argparse.ArgumentParser(
        description="Build Disk-Based Positional Inverted Index from Tamil Wikipedia"
    )
    parser.add_argument(
        "--articles",
        type=str,
        default=None,
        help="Path to extracted articles JSONL file (default: data/extracted/articles.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="indexes",
        help="Directory to store index files (default: indexes)",
    )

    args = parser.parse_args()
    build_and_save(articles_path=args.articles, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
