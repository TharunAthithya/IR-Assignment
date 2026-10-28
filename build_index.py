"""Index Construction Module for Tamil Information Retrieval System.

This module is responsible for:
1. Loading the Tamil news dataset from a Parquet file (using PyArrow / Pandas).
2. Applying Unicode-aware preprocessing (NFC normalization, lowercasing, tokenization).
3. Constructing the Positional Inverted Index:
   - Maps each normalized term to a dictionary of {doc_id: [pos0, pos1, ...]}.
   - Stores zero-based positions in ascending order.
   - Tracks term occurrences and repeated words within each document.
4. Deriving the Regular Inverted Index:
   - Postings list of sorted document IDs: {term: [doc_id0, doc_id1, ...]}.
   - Derived directly and efficiently from the positional index without rescanning the corpus.
5. Computing collection statistics (Vocabulary size, Document Frequency `df`, Collection Frequency `cf`).
6. Persisting all indexes and metadata to JSON files in the `indexes/` directory:
   - `indexes/inverted_index.json`
   - `indexes/positional_index.json`
   - `indexes/dictionary.json`
   - `indexes/documents.json`
   - `indexes/metadata.json`
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


def load_dataset(parquet_path: str) -> List[Dict[str, Any]]:
    """Loads records from the Parquet dataset file using PyArrow or Pandas.

    Args:
        parquet_path: Path to the Parquet file.

    Returns:
        List of dictionaries containing document fields ('Title', 'Author', 'City', 'Published', 'Text').

    Raises:
        FileNotFoundError: If the Parquet file does not exist.
        RuntimeError: If neither PyArrow nor Pandas can read the file.
    """
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Dataset file not found at: {parquet_path}")

    # Primary loader: PyArrow (fast, native, reliable on all environments)
    try:
        import pyarrow.parquet as pq
        table = pq.read_table(parquet_path)
        records = table.to_pylist()
        return records
    except Exception as pa_err:
        # Fallback loader: Pandas
        try:
            import pandas as pd
            df = pd.read_parquet(parquet_path)
            records = df.to_dict(orient="records")
            return records
        except Exception as pd_err:
            raise RuntimeError(
                f"Failed to read Parquet dataset using both PyArrow ({pa_err}) and Pandas ({pd_err})."
            )


def build_indexes(
    records: List[Dict[str, Any]]
) -> Tuple[
    Dict[str, Dict[str, List[int]]],  # positional_index
    Dict[str, List[int]],              # inverted_index
    Dict[str, Dict[str, int]],         # dictionary (df, cf)
    Dict[str, Dict[str, Any]],         # documents store
    Dict[str, Any]                     # indexing stats
]:
    """Constructs positional and regular inverted indexes from the loaded records.

    Algorithm:
    1. Single-pass over documents:
       - Combine Title and Text into a unified document text stream (`f"{Title} {Text}"`).
       - Tokenize using Unicode-aware Tamil tokenizer.
       - Accumulate positional postings: term -> doc_id -> list of positions.
       - Store document metadata in the documents dictionary.
    2. Academic Derivation of Regular Inverted Index:
       - A positional inverted index already captures all document-term associations.
       - Thus, the regular postings list {term: [doc_ids]} is derived directly from
         the positional postings keys, eliminating redundant corpus rescanning.
    3. Dictionary Calculation:
       - Computes Document Frequency (df = number of documents containing term).
       - Computes Collection Frequency (cf = total token occurrences across collection).

    Args:
        records: List of document dicts.

    Returns:
        Tuple of (positional_index, inverted_index, dictionary, documents, stats).
    """
    start_time = time.perf_counter()

    positional_index: Dict[str, Dict[str, List[int]]] = {}
    documents: Dict[str, Dict[str, Any]] = {}
    total_tokens = 0

    for doc_id, record in enumerate(records):
        doc_id_str = str(doc_id)
        title = record.get("Title") or ""
        text = record.get("Text") or ""
        author = record.get("Author") or ""
        city = record.get("City") or ""
        published = record.get("Published") or ""

        # Store document metadata
        documents[doc_id_str] = {
            "doc_id": doc_id,
            "title": title,
            "author": author,
            "city": city,
            "published": published,
            "text": text,
        }

        # Combine Title and Text with a separating space
        # Token positions: title tokens come first, followed by text tokens
        combined_text = f"{title} {text}"
        tokens = tokenize(combined_text)
        total_tokens += len(tokens)

        # Single-pass positional posting accumulation
        for pos, term in enumerate(tokens):
            if term not in positional_index:
                positional_index[term] = {}

            doc_postings = positional_index[term]
            if doc_id_str not in doc_postings:
                doc_postings[doc_id_str] = [pos]
            else:
                doc_postings[doc_id_str].append(pos)

    # Academic Derivation: Derive regular inverted index and dictionary
    inverted_index: Dict[str, List[int]] = {}
    dictionary: Dict[str, Dict[str, int]] = {}

    for term, doc_map in positional_index.items():
        # Sorted list of integer document IDs
        sorted_doc_ids = sorted([int(did) for did in doc_map.keys()])
        inverted_index[term] = sorted_doc_ids

        # Collection frequency = sum of position occurrences across all matching documents
        cf = sum(len(positions) for positions in doc_map.values())
        dictionary[term] = {
            "df": len(sorted_doc_ids),
            "cf": cf,
        }

    duration = time.perf_counter() - start_time
    stats = {
        "total_documents": len(records),
        "total_tokens": total_tokens,
        "vocabulary_size": len(positional_index),
        "avg_tokens_per_doc": round(total_tokens / len(records), 2) if records else 0,
        "build_duration_seconds": round(duration, 3),
    }

    return positional_index, inverted_index, dictionary, documents, stats


def save_indexes(
    output_dir: str,
    positional_index: Dict[str, Dict[str, List[int]]],
    inverted_index: Dict[str, List[int]],
    dictionary: Dict[str, Dict[str, int]],
    documents: Dict[str, Dict[str, Any]],
    stats: Dict[str, Any],
    dataset_path: str = "data/0000.parquet",
) -> Dict[str, str]:
    """Persists indexes and metadata into UTF-8 formatted JSON files.

    Args:
        output_dir: Directory where JSON files will be written.
        positional_index: Positional index mapping.
        inverted_index: Regular inverted index mapping.
        dictionary: Vocabulary dictionary with df and cf.
        documents: Document metadata store.
        stats: Indexing statistics.
        dataset_path: Original Parquet file path for provenance metadata.

    Returns:
        Dict mapping filename to absolute filepath.
    """
    os.makedirs(output_dir, exist_ok=True)

    files_to_write = [
        ("inverted_index.json", inverted_index),
        ("positional_index.json", positional_index),
        ("dictionary.json", dictionary),
        ("documents.json", documents),
    ]

    saved_paths = {}
    for filename, data in files_to_write:
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        saved_paths[filename] = filepath

    # Measure actual generated file sizes
    file_sizes = {}
    for filename in ["inverted_index.json", "positional_index.json", "dictionary.json", "documents.json"]:
        fpath = os.path.join(output_dir, filename)
        bytes_size = os.path.getsize(fpath)
        file_sizes[filename] = {
            "bytes": bytes_size,
            "mb": round(bytes_size / (1024 * 1024), 2),
        }

    # Construct comprehensive metadata
    metadata = {
        "dataset_name": "Hindu Tamil News Articles Dataset",
        "dataset_source": "https://huggingface.co/datasets/Shwetasss/HinduTamil-News-Articles-Dataset",
        "local_dataset_file": dataset_path,
        "indexing_timestamp": datetime.now().isoformat(),
        "total_original_records": stats["total_documents"],
        "total_indexed_documents": stats["total_documents"],
        "total_unique_terms": stats["vocabulary_size"],
        "total_indexed_tokens": stats["total_tokens"],
        "average_tokens_per_document": stats["avg_tokens_per_doc"],
        "index_construction_duration_seconds": stats["build_duration_seconds"],
        "preprocessing_settings": {
            "normalization": "Unicode NFC",
            "case_folding": "Lowercase for English, preserved for Tamil",
            "token_pattern": r"[\p{L}\p{N}][\p{L}\p{M}\p{N}]*",
            "stopwords_removed": False,
            "stemming_applied": False,
            "combining_marks_preserved": True,
        },
        "file_sizes": file_sizes,
    }

    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    saved_paths["metadata.json"] = metadata_path

    return saved_paths


def build_and_save(
    data_path: str = "data/0000.parquet",
    output_dir: str = "indexes",
) -> Dict[str, Any]:
    """High-level pipeline function to load data, build indexes, and persist them."""
    print("=" * 60)
    print("TAMIL INFORMATION RETRIEVAL SYSTEM — INDEX CONSTRUCTION")
    print("=" * 60)
    print(f"Loading dataset from: {data_path} ...")
    records = load_dataset(data_path)
    print(f"Loaded {len(records):,} records.")

    print("Building positional and regular inverted indexes...")
    pos_idx, inv_idx, dictionary, docs, stats = build_indexes(records)

    print(f"Indexing completed in {stats['build_duration_seconds']:.2f} seconds.")
    print(f"  Total Indexed Documents: {stats['total_documents']:,}")
    print(f"  Vocabulary Size:         {stats['vocabulary_size']:,} unique terms")
    print(f"  Total Indexed Tokens:    {stats['total_tokens']:,}")
    print(f"  Average Tokens/Doc:      {stats['avg_tokens_per_doc']}")

    print(f"\nPersisting indexes to '{output_dir}/' ...")
    saved_paths = save_indexes(output_dir, pos_idx, inv_idx, dictionary, docs, stats, data_path)

    print("Successfully saved index files:")
    for fname, path in saved_paths.items():
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  - {fname:24s} ({size_mb:6.2f} MB) -> {path}")

    print("=" * 60)
    return stats


def main():
    """Command-line interface for index building."""
    parser = argparse.ArgumentParser(
        description="Build Inverted and Positional Indexes for Tamil News Dataset"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/0000.parquet",
        help="Path to the input Parquet dataset (default: data/0000.parquet)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="indexes",
        help="Directory to store generated JSON index files (default: indexes)",
    )

    args = parser.parse_args()
    build_and_save(data_path=args.data, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
