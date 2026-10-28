"""Interactive Command-Line Application for Tamil Information Retrieval System.

This module provides a rich, robust, and interactive terminal interface for:
1. Building or rebuilding Inverted and Positional indexes.
2. Executing Single-Term search queries.
3. Executing Boolean AND search queries.
4. Executing Boolean OR search queries.
5. Executing Exact Phrase searches with positional verification.
6. Viewing comprehensive corpus and index statistics.
7. Inspecting index postings and positions for a specific Tamil term.
8. Graceful exit and error recovery.

Also supports non-interactive command-line arguments for automated testing and scripting.
"""

import argparse
import os
import sys
from typing import Optional

# Reconfigure stdout and stdin for full Unicode support in Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
        sys.stdin.reconfigure(encoding="utf-8")
    except Exception:
        pass

from build_index import build_and_save
from preprocess import normalize_text, tokenize
from search import TamilSearchEngine


def print_banner():
    """Prints the system header banner."""
    print("=" * 66)
    print("      TAMIL INFORMATION RETRIEVAL SEARCH ENGINE (IR ASSIGNMENT)    ")
    print("=" * 66)


def print_menu():
    """Displays the interactive menu options."""
    print("\n" + "=" * 46)
    print("  TAMIL INFORMATION RETRIEVAL SEARCH ENGINE  ")
    print("=" * 46)
    print(" 1. Build Indexes")
    print(" 2. Single-Term Search")
    print(" 3. Boolean AND Search")
    print(" 4. Boolean OR Search")
    print(" 5. Exact Phrase Search")
    print(" 6. View Index Statistics")
    print(" 7. Display Index for a Term")
    print(" 8. Exit")
    print("=" * 46)


def load_or_init_engine(index_dir: str = "indexes") -> Optional[TamilSearchEngine]:
    """Safely loads the search engine, warning user if indexes are missing."""
    try:
        engine = TamilSearchEngine(index_dir=index_dir)
        return engine
    except FileNotFoundError:
        print(f"\n[!] Warning: Index files not found in '{index_dir}'.")
        print("    Please select Option 1 ('Build Indexes') from the menu first.\n")
        return None
    except Exception as e:
        print(f"\n[!] Error loading indexes: {e}\n")
        return None


def handle_search(engine: TamilSearchEngine, mode: str):
    """Handles interactive query entry, execution, and pagination of results."""
    mode_names = {
        "term": ("Single-Term Search", "Enter a Tamil word (e.g. சென்னை): "),
        "and": ("Boolean AND Search", "Enter terms separated by AND (e.g. சென்னை AND மழை): "),
        "or": ("Boolean OR Search", "Enter terms separated by OR (e.g. தமிழ்நாடு OR சென்னை): "),
        "phrase": ("Exact Phrase Search", "Enter exact Tamil phrase (e.g. தமிழக அரசு): "),
    }

    title, prompt = mode_names.get(mode, ("Search", "Enter query: "))
    print(f"\n--- {title} ---")
    query = input(prompt).strip()

    if not query:
        print("[!] Empty query entered. Returning to menu.")
        return

    # Execute search according to mode
    if mode == "term":
        res = engine.single_term_search(query)
    elif mode == "and":
        res = engine.boolean_and_search(query)
    elif mode == "or":
        res = engine.boolean_or_search(query)
    elif mode == "phrase":
        res = engine.exact_phrase_search(query)
    else:
        print("[!] Invalid search mode.")
        return

    # Display initial batch of results
    print("\n" + engine.format_output(res, max_display=10))

    # Pagination: allow viewing additional matches if total > 10
    total = res.total_matches
    current_offset = 10

    while current_offset < total:
        print(f"\nMatches {current_offset + 1} to {min(current_offset + 10, total)} of {total} available.")
        user_choice = input("Enter 'n' or press Enter to view next 10 results, or 'm' for main menu: ").strip().lower()
        if user_choice == "m":
            break

        next_ids = res.matching_doc_ids[current_offset : current_offset + 10]
        pos_map = res.matching_positions if mode == "phrase" else None
        more_items = engine.get_result_items(
            next_ids,
            res.query_terms,
            positions_map=pos_map,
            phrase_match=(mode == "phrase"),
        )

        for idx, item in enumerate(more_items, start=current_offset + 1):
            print("-" * 60)
            print(f"Result #{idx}")
            print(f"Document ID: {item.doc_id}")
            print(f"Title: {item.title}")
            if item.matched_positions is not None:
                print(f"Matched Positions: {item.matched_positions}")
            print(f"Snippet: {item.snippet}")
            if item.city or item.published:
                print(f"Metadata: City={item.city} | Published={item.published}")

        current_offset += len(more_items)


def display_index_statistics(engine: Optional[TamilSearchEngine], index_dir: str = "indexes"):
    """Displays corpus and index statistics."""
    print("\n" + "=" * 60)
    print("            COLLECTION & INDEX STATISTICS            ")
    print("=" * 60)

    if engine is None or not engine.is_loaded:
        print(f"[!] Indexes are not loaded from '{index_dir}'.")
        return

    meta = engine.metadata
    print(f"Dataset Name:          {meta.get('dataset_name', 'N/A')}")
    print(f"Dataset Source:        {meta.get('dataset_source', 'N/A')}")
    print(f"Local Parquet File:    {meta.get('local_dataset_file', 'N/A')}")
    print(f"Indexing Timestamp:    {meta.get('indexing_timestamp', 'N/A')}")
    print("-" * 60)
    print(f"Total Documents:       {meta.get('total_indexed_documents', 0):,}")
    print(f"Total Unique Terms:    {meta.get('total_unique_terms', 0):,} terms (Vocabulary Size)")
    print(f"Total Indexed Tokens:  {meta.get('total_indexed_tokens', 0):,} tokens")
    print(f"Average Tokens/Doc:    {meta.get('average_tokens_per_document', 0)}")
    print(f"Build Duration:        {meta.get('index_construction_duration_seconds', 0):.2f} seconds")
    print("-" * 60)
    print("Storage File Sizes on Disk:")
    for fname, size_info in meta.get("file_sizes", {}).items():
        mb = size_info.get("mb", 0)
        bytes_val = size_info.get("bytes", 0)
        print(f"  - {fname:24s}: {mb:6.2f} MB ({bytes_val:,} bytes)")

    print("-" * 60)
    print("Preprocessing Settings:")
    for k, v in meta.get("preprocessing_settings", {}).items():
        print(f"  - {k}: {v}")
    print("=" * 60)


def display_term_index(engine: Optional[TamilSearchEngine]):
    """Displays posting details for a user-specified Tamil term."""
    print("\n--- Display Index for a Term ---")
    if engine is None or not engine.is_loaded:
        print("[!] Indexes must be built and loaded first.")
        return

    term_input = input("Enter a Tamil term to inspect (e.g. தமிழ்நாடு or சென்னை): ").strip()
    if not term_input:
        print("[!] No term entered.")
        return

    tokens = tokenize(term_input)
    if not tokens:
        print("[!] Input produced no valid tokens.")
        return

    term = tokens[0]
    print(f"\nNormalized Term: '{term}'")

    dict_entry = engine.dictionary.get(term)
    if not dict_entry:
        print(f"[!] Term '{term}' was NOT found in the vocabulary.")
        return

    df = dict_entry.get("df", 0)
    cf = dict_entry.get("cf", 0)
    print(f"Document Frequency (df):    {df:,} documents")
    print(f"Collection Frequency (cf):  {cf:,} occurrences")

    # Regular Inverted Index Posting List
    inv_postings = engine.inverted_index.get(term, [])
    preview_inv = inv_postings[:25]
    print(f"\nRegular Inverted Index Postings List ({len(inv_postings):,} total docs):")
    print(f"  First 25 Document IDs: {preview_inv}")
    if len(inv_postings) > 25:
        print(f"  ... ({len(inv_postings) - 25} more document IDs)")

    # Positional Inverted Index Posting Sample
    pos_postings = engine.positional_index.get(term, {})
    sample_docs = list(pos_postings.keys())[:5]
    print(f"\nPositional Inverted Index Sample (First 5 docs):")
    for doc_id_str in sample_docs:
        positions = pos_postings[doc_id_str]
        doc_title = engine.documents.get(doc_id_str, {}).get("title", "")
        print(f"  Doc ID {doc_id_str} ({len(positions)} occurrences):")
        print(f"    Positions: {positions}")
        print(f"    Title:     {doc_title[:75]}...")


def interactive_loop():
    """Main application loop."""
    print_banner()
    index_dir = "indexes"
    engine = load_or_init_engine(index_dir=index_dir)

    while True:
        print_menu()
        choice = input("Enter your choice (1-8): ").strip()

        if choice == "1":
            print("\n[+] Rebuilding indexes from dataset...")
            build_and_save(data_path="data/0000.parquet", output_dir=index_dir)
            print("[+] Reloading search engine with newly generated indexes...")
            engine = load_or_init_engine(index_dir=index_dir)
            print("[+] Indexes successfully built and loaded!")

        elif choice in ("2", "3", "4", "5"):
            if engine is None:
                print("\n[!] Indexes not loaded. Please choose Option 1 to build indexes first.")
                continue

            mode_map = {
                "2": "term",
                "3": "and",
                "4": "or",
                "5": "phrase",
            }
            handle_search(engine, mode_map[choice])

        elif choice == "6":
            display_index_statistics(engine, index_dir=index_dir)

        elif choice == "7":
            display_term_index(engine)

        elif choice == "8":
            print("\nExiting Tamil Information Retrieval System. Goodbye!")
            sys.exit(0)

        else:
            print("\n[!] Invalid selection. Please enter a number between 1 and 8.")


def main():
    """Entry point supporting both interactive CLI and non-interactive arguments."""
    parser = argparse.ArgumentParser(description="Tamil Information Retrieval Search Engine")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive menu loop")
    parser.add_argument("--build", action="store_true", help="Build indexes non-interactively")
    parser.add_argument("--stats", action="store_true", help="Print index statistics and exit")
    parser.add_argument("--inspect-term", type=str, help="Inspect index details for a given term")
    parser.add_argument("--data", type=str, default="data/0000.parquet", help="Parquet dataset path")
    parser.add_argument("--index-dir", type=str, default="indexes", help="Indexes directory")

    args = parser.parse_args()

    # Non-interactive triggers
    if args.build:
        build_and_save(data_path=args.data, output_dir=args.index_dir)
        return

    if args.stats:
        engine = load_or_init_engine(index_dir=args.index_dir)
        display_index_statistics(engine, index_dir=args.index_dir)
        return

    if args.inspect_term:
        engine = load_or_init_engine(index_dir=args.index_dir)
        tokens = tokenize(args.inspect_term)
        term = tokens[0] if tokens else args.inspect_term
        print(f"\n--- Inspection for Term: '{term}' ---")
        if engine and term in engine.dictionary:
            dict_info = engine.dictionary[term]
            print(f"df: {dict_info['df']}, cf: {dict_info['cf']}")
            print(f"Inverted postings (first 10): {engine.inverted_index.get(term, [])[:10]}")
            pos_dict = engine.positional_index.get(term, {})
            first_docs = list(pos_dict.keys())[:3]
            for did in first_docs:
                print(f"  Doc {did}: positions={pos_dict[did]}")
        else:
            print("Term not found in dictionary.")
        return

    # Default: launch interactive loop
    interactive_loop()


if __name__ == "__main__":
    main()
