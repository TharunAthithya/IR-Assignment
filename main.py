"""Command-Line Interface for Tamil Wikipedia Information Retrieval System.

Provides subcommands for the complete IR workflow:
    extract   — Extract articles from Wikipedia XML dump
    build     — Build disk-based positional inverted index
    search    — Search for a term or multiple terms
    phrase    — Exact phrase search with positional verification
    inspect   — Inspect a term's positional postings
    stats     — Display index statistics
    generate  — Generate assignment output files (test cases + rare terms)
    package   — Create submission ZIP archives
    interactive — Launch interactive menu

Usage examples:
    python main.py extract
    python main.py build
    python main.py search "தமிழ்நாடு"
    python main.py phrase "தமிழ்நாடு அரசு"
    python main.py inspect "தமிழ்நாடு"
    python main.py stats
    python main.py generate
    python main.py package --roll ROLLNO
"""

import argparse
import os
import sys
import zipfile
from typing import Optional

# Reconfigure stdout and stdin for full Unicode support
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
        sys.stdin.reconfigure(encoding="utf-8")
    except Exception:
        pass


def cmd_extract(args):
    """Extract articles from Tamil Wikipedia XML dump."""
    from extract_wiki import extract_and_save
    extract_and_save(
        dump_dir=args.dump_dir,
        output_dir=args.extracted_dir,
        force=args.force,
    )


def cmd_build(args):
    """Build disk-based positional inverted index."""
    from build_index import build_and_save
    build_and_save(
        articles_path=args.articles,
        output_dir=args.index_dir,
    )


def cmd_search(args):
    """Execute a single-term or multi-term search."""
    from search import TamilSearchEngine
    engine = TamilSearchEngine(index_dir=args.index_dir)

    if len(args.query.split()) > 1:
        res = engine.multi_term_search(args.query, max_results=args.limit)
    else:
        res = engine.single_term_search(args.query, max_results=args.limit)

    print(engine.format_output(res, max_display=args.limit))


def cmd_phrase(args):
    """Execute an exact phrase search."""
    from search import TamilSearchEngine
    engine = TamilSearchEngine(index_dir=args.index_dir)
    res = engine.phrase_search(args.query, max_results=args.limit)
    print(engine.format_output(res, max_display=args.limit))


def cmd_inspect(args):
    """Inspect a term's positional postings."""
    from search import TamilSearchEngine
    engine = TamilSearchEngine(index_dir=args.index_dir)
    info = engine.get_term_info(args.term)

    if info is None:
        print(f"Term '{args.term}' was NOT found in the vocabulary.")
        return

    print("\n" + "=" * 60)
    print(f"TERM INSPECTION: '{info['term']}'")
    print("=" * 60)
    print(f"Document Frequency (df):  {info['df']:,} documents")
    print(f"Collection Frequency (cf): {info['cf']:,} total occurrences")
    print(f"Postings offset:          {info['offset']:,} bytes")
    print(f"Postings length:          {info['length']:,} bytes")
    print("-" * 60)
    print(f"Positional Postings Sample (first {len(info['sample_docs'])} documents):")
    for did_str, doc_info in info["sample_docs"].items():
        print(f"  Doc ID {did_str} ({doc_info['occurrences']} occurrences):")
        print(f"    Title:     {doc_info['title'][:80]}")
        print(f"    Positions: {doc_info['positions']}")
    print("=" * 60)


def cmd_stats(args):
    """Display index statistics."""
    from search import TamilSearchEngine
    engine = TamilSearchEngine(index_dir=args.index_dir)
    meta = engine.metadata

    print("\n" + "=" * 60)
    print("         INDEX & COLLECTION STATISTICS")
    print("=" * 60)
    print(f"Dataset:                 {meta.get('dataset_name', 'N/A')}")
    print(f"Source:                  {meta.get('dataset_source', 'N/A')}")
    print(f"Indexing Timestamp:      {meta.get('indexing_timestamp', 'N/A')}")
    print("-" * 60)
    print(f"Total Documents:         {meta.get('total_documents', 0):,}")
    print(f"Vocabulary Size:         {meta.get('vocabulary_size', 0):,} unique terms")
    print(f"Total Tokens:            {meta.get('total_tokens', 0):,}")
    print(f"Term-Doc Postings:       {meta.get('total_term_doc_postings', 0):,}")
    print(f"Positional Entries:      {meta.get('total_positional_entries', 0):,}")
    print(f"Average Doc Length:      {meta.get('average_doc_length', 0)} tokens")
    print(f"Build Duration:          {meta.get('index_construction_seconds', 0):.2f} seconds")
    print("-" * 60)
    storage = meta.get("storage", {})
    print("Storage Architecture: Vocabulary in RAM, Postings on Disk")
    print(f"  Vocabulary file:       {storage.get('vocabulary_size_mb', 0):.2f} MB")
    print(f"  Postings file:         {storage.get('postings_size_mb', 0):.2f} MB")
    print(f"  Doc metadata file:     {storage.get('doc_meta_size_mb', 0):.2f} MB")
    print("-" * 60)
    preprocessing = meta.get("preprocessing", {})
    print("Preprocessing Settings:")
    for k, v in preprocessing.items():
        print(f"  {k}: {v}")
    print("=" * 60)


def cmd_generate(args):
    """Generate assignment output files."""
    from generate_outputs import generate_all_outputs
    generate_all_outputs(
        output_dir=args.output_dir,
        index_dir=args.index_dir,
    )


def cmd_package(args):
    """Create submission ZIP archives."""
    roll = args.roll

    # Source files to include in src.zip
    src_files = [
        "preprocess.py",
        "extract_wiki.py",
        "build_index.py",
        "search.py",
        "main.py",
        "generate_outputs.py",
        "requirements.txt",
        "tests/__init__.py",
        "tests/test_preprocessing.py",
        "tests/test_indexing.py",
        "tests/test_search.py",
        "tests/test_integration.py",
    ]

    # Create source ZIP
    src_zip_name = f"{roll}-src.zip"
    src_zip_path = os.path.join(".", src_zip_name)
    print(f"Creating {src_zip_name}...")
    with zipfile.ZipFile(src_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fpath in src_files:
            if os.path.exists(fpath):
                zf.write(fpath, fpath)
                print(f"  Added: {fpath}")
            else:
                print(f"  MISSING: {fpath}")
    print(f"Created: {src_zip_path}")

    # Create README.txt (copy of README.md as plain text)
    readme_txt_path = os.path.join(".", f"{roll}-README.txt")
    if os.path.exists("README.md"):
        with open("README.md", "r", encoding="utf-8") as fin:
            content = fin.read()
        with open(readme_txt_path, "w", encoding="utf-8") as fout:
            fout.write(content)
        print(f"Created: {readme_txt_path}")

    # Check output file
    output_file = os.path.join("outputs", "assignment_output.txt")
    if not os.path.exists(output_file):
        print(f"\n[!] Output file '{output_file}' not found.")
        print("    Run 'python main.py generate' first.")
        return

    output_txt_name = f"{roll}-output.txt"
    output_txt_path = os.path.join(".", output_txt_name)
    with open(output_file, "r", encoding="utf-8") as fin:
        content = fin.read()
    with open(output_txt_path, "w", encoding="utf-8") as fout:
        fout.write(content)
    print(f"Created: {output_txt_path}")

    # Create final submission ZIP
    final_zip_name = f"{roll}-A1-PostIndex-IR-M2026.zip"
    final_zip_path = os.path.join(".", final_zip_name)
    print(f"\nCreating final submission: {final_zip_name}...")
    with zipfile.ZipFile(final_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(readme_txt_path, f"{roll}-README.txt")
        zf.write(src_zip_path, src_zip_name)
        zf.write(output_txt_path, output_txt_name)
    print(f"Created: {final_zip_path}")

    print("\n" + "=" * 60)
    print("SUBMISSION PACKAGING COMPLETE")
    print("=" * 60)
    print(f"Final archive: {final_zip_path}")
    print("Contents:")
    print(f"  1. {roll}-README.txt")
    print(f"  2. {src_zip_name}")
    print(f"  3. {output_txt_name}")
    print("=" * 60)


def cmd_test(args):
    """Run the test suite."""
    import subprocess
    cmd = [
        sys.executable, "-X", "utf8",
        "-m", "unittest", "discover",
        "-s", "tests", "-p", "test_*.py", "-v",
    ]
    subprocess.run(cmd, cwd=os.getcwd())


def cmd_interactive(args):
    """Launch interactive search menu."""
    from search import TamilSearchEngine
    from preprocess import tokenize

    print("=" * 66)
    print("   TAMIL WIKIPEDIA IR SYSTEM — INTERACTIVE MODE")
    print("=" * 66)

    try:
        engine = TamilSearchEngine(index_dir=args.index_dir)
    except FileNotFoundError:
        print("[!] Index files not found. Run 'python main.py build' first.")
        return

    while True:
        print("\n" + "=" * 46)
        print("  SEARCH MENU")
        print("=" * 46)
        print(" 1. Single-Term Search")
        print(" 2. Multi-Term Search")
        print(" 3. Phrase Search")
        print(" 4. Inspect Term")
        print(" 5. Index Statistics")
        print(" 6. Exit")
        print("=" * 46)

        choice = input("Enter choice (1-6): ").strip()

        if choice == "1":
            query = input("Enter a Tamil term: ").strip()
            if query:
                res = engine.single_term_search(query)
                print("\n" + engine.format_output(res, max_display=10))

        elif choice == "2":
            query = input("Enter multiple Tamil terms: ").strip()
            if query:
                res = engine.multi_term_search(query)
                print("\n" + engine.format_output(res, max_display=10))

        elif choice == "3":
            query = input("Enter Tamil phrase: ").strip()
            if query:
                res = engine.phrase_search(query)
                print("\n" + engine.format_output(res, max_display=10))

        elif choice == "4":
            term = input("Enter term to inspect: ").strip()
            if term:
                info = engine.get_term_info(term)
                if info:
                    print(f"\nTerm: {info['term']}")
                    print(f"df: {info['df']:,}, cf: {info['cf']:,}")
                    print(f"Offset: {info['offset']}, Length: {info['length']}")
                    for did, dinfo in info["sample_docs"].items():
                        print(f"  Doc {did}: positions={dinfo['positions']}  title={dinfo['title'][:60]}")
                else:
                    print("Term not found in vocabulary.")

        elif choice == "5":
            # Reuse stats display
            class _Args:
                index_dir = args.index_dir
            cmd_stats(_Args())

        elif choice == "6":
            print("Goodbye!")
            break

        else:
            print("[!] Invalid choice.")


def main():
    """Entry point with subcommand routing."""
    parser = argparse.ArgumentParser(
        description="Tamil Wikipedia Information Retrieval System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- extract ---
    p_extract = subparsers.add_parser("extract", help="Extract Wikipedia dump")
    p_extract.add_argument("--dump-dir", default="data/wikipedia", help="Directory containing dump file")
    p_extract.add_argument("--extracted-dir", default="data/extracted", help="Output directory for extracted articles")
    p_extract.add_argument("--force", action="store_true", help="Force re-extraction")
    p_extract.set_defaults(func=cmd_extract)

    # --- build ---
    p_build = subparsers.add_parser("build", help="Build positional inverted index")
    p_build.add_argument("--articles", default=None, help="Path to articles.jsonl")
    p_build.add_argument("--index-dir", default="indexes", help="Output directory for index files")
    p_build.set_defaults(func=cmd_build)

    # --- search ---
    p_search = subparsers.add_parser("search", help="Search for terms")
    p_search.add_argument("query", type=str, help="Search query")
    p_search.add_argument("--index-dir", default="indexes", help="Index directory")
    p_search.add_argument("--limit", type=int, default=10, help="Max results")
    p_search.set_defaults(func=cmd_search)

    # --- phrase ---
    p_phrase = subparsers.add_parser("phrase", help="Exact phrase search")
    p_phrase.add_argument("query", type=str, help="Phrase query")
    p_phrase.add_argument("--index-dir", default="indexes", help="Index directory")
    p_phrase.add_argument("--limit", type=int, default=10, help="Max results")
    p_phrase.set_defaults(func=cmd_phrase)

    # --- inspect ---
    p_inspect = subparsers.add_parser("inspect", help="Inspect a term's postings")
    p_inspect.add_argument("term", type=str, help="Term to inspect")
    p_inspect.add_argument("--index-dir", default="indexes", help="Index directory")
    p_inspect.set_defaults(func=cmd_inspect)

    # --- stats ---
    p_stats = subparsers.add_parser("stats", help="Show index statistics")
    p_stats.add_argument("--index-dir", default="indexes", help="Index directory")
    p_stats.set_defaults(func=cmd_stats)

    # --- generate ---
    p_gen = subparsers.add_parser("generate", help="Generate assignment outputs")
    p_gen.add_argument("--output-dir", default="outputs", help="Output directory")
    p_gen.add_argument("--index-dir", default="indexes", help="Index directory")
    p_gen.set_defaults(func=cmd_generate)

    # --- package ---
    p_pkg = subparsers.add_parser("package", help="Create submission ZIPs")
    p_pkg.add_argument("--roll", default="ROLLNO", help="Your roll number")
    p_pkg.set_defaults(func=cmd_package)

    # --- test ---
    p_test = subparsers.add_parser("test", help="Run test suite")
    p_test.set_defaults(func=cmd_test)

    # --- interactive ---
    p_inter = subparsers.add_parser("interactive", help="Interactive search menu")
    p_inter.add_argument("--index-dir", default="indexes", help="Index directory")
    p_inter.set_defaults(func=cmd_interactive)

    args = parser.parse_args()

    if args.command is None:
        # Default: show help
        parser.print_help()
        print("\nQuick start:")
        print("  python main.py extract       # Extract Wikipedia dump")
        print("  python main.py build         # Build index")
        print("  python main.py search \"term\" # Search")
        print("  python main.py interactive   # Interactive mode")
        return

    args.func(args)


if __name__ == "__main__":
    main()
