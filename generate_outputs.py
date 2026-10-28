"""Generate real sample output files from actual program executions."""

import os
import sys
import io

# Ensure UTF-8 output encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from search import TamilSearchEngine
from preprocess import tokenize


def generate_all_outputs(output_dir: str = "outputs", index_dir: str = "indexes"):
    os.makedirs(output_dir, exist_ok=True)
    engine = TamilSearchEngine(index_dir=index_dir)

    print(f"Generating sample output files in '{output_dir}/' ...")

    # 1. index_statistics.txt
    stats_file = os.path.join(output_dir, "index_statistics.txt")
    meta = engine.metadata
    with open(stats_file, "w", encoding="utf-8") as f:
        f.write("=" * 66 + "\n")
        f.write("      TAMIL INFORMATION RETRIEVAL SYSTEM — INDEX STATISTICS       \n")
        f.write("=" * 66 + "\n")
        f.write(f"Dataset Name:          {meta.get('dataset_name')}\n")
        f.write(f"Dataset Source:        {meta.get('dataset_source')}\n")
        f.write(f"Local Parquet Path:    {meta.get('local_dataset_file')}\n")
        f.write(f"Indexing Timestamp:    {meta.get('indexing_timestamp')}\n")
        f.write("-" * 66 + "\n")
        f.write(f"Total Original Docs:   {meta.get('total_original_records'):,}\n")
        f.write(f"Total Indexed Docs:    {meta.get('total_indexed_documents'):,}\n")
        f.write(f"Vocabulary Size:       {meta.get('total_unique_terms'):,} unique terms\n")
        f.write(f"Total Indexed Tokens:  {meta.get('total_indexed_tokens'):,} tokens\n")
        f.write(f"Average Tokens/Doc:    {meta.get('average_tokens_per_document')}\n")
        f.write(f"Build Duration:        {meta.get('index_construction_duration_seconds'):.3f} seconds\n")
        f.write("-" * 66 + "\n")
        f.write("Generated Index Files on Disk:\n")
        for fname, sinfo in meta.get("file_sizes", {}).items():
            f.write(f"  - {fname:24s}: {sinfo.get('mb'):6.2f} MB ({sinfo.get('bytes'):,} bytes)\n")
        f.write("-" * 66 + "\n")
        f.write("Preprocessing Configuration:\n")
        for k, v in meta.get("preprocessing_settings", {}).items():
            f.write(f"  - {k:28s}: {v}\n")
        f.write("=" * 66 + "\n")
    print(f"Created {stats_file}")

    # 2. term_search.txt
    term_file = os.path.join(output_dir, "term_search.txt")
    with open(term_file, "w", encoding="utf-8") as f:
        f.write("=" * 66 + "\n")
        f.write("            SINGLE-TERM SEARCH EXECUTION EXAMPLES                \n")
        f.write("=" * 66 + "\n\n")

        # Example 1: High frequency term
        res1 = engine.single_term_search("சென்னை")
        f.write("### Example 1: Frequent City Term 'சென்னை' ###\n")
        f.write(engine.format_output(res1, max_display=5) + "\n\n")

        # Example 2: Topical term
        res2 = engine.single_term_search("விவசாயிகள்")
        f.write("### Example 2: Topical Term 'விவசாயிகள்' (Farmers) ###\n")
        f.write(engine.format_output(res2, max_display=5) + "\n\n")

        # Example 3: Non-existent term
        res3 = engine.single_term_search("இமயமலைரோபோ2099")
        f.write("### Example 3: Absent Term 'இமயமலைரோபோ2099' (No-match) ###\n")
        f.write(engine.format_output(res3, max_display=5) + "\n\n")
    print(f"Created {term_file}")

    # 3. boolean_and_search.txt
    and_file = os.path.join(output_dir, "boolean_and_search.txt")
    with open(and_file, "w", encoding="utf-8") as f:
        f.write("=" * 66 + "\n")
        f.write("            BOOLEAN AND SEARCH EXECUTION EXAMPLES                \n")
        f.write("=" * 66 + "\n\n")

        # Example 1: Two terms
        res1 = engine.boolean_and_search("சென்னை AND மழை")
        f.write("### Example 1: Two Terms 'சென்னை AND மழை' ###\n")
        f.write(engine.format_output(res1, max_display=5) + "\n\n")

        # Example 2: Three terms
        res2 = engine.boolean_and_search("தமிழ்நாடு AND காங்கிரஸ் AND கமிட்டி")
        f.write("### Example 2: Three Terms 'தமிழ்நாடு AND காங்கிரஸ் AND கமிட்டி' ###\n")
        f.write(engine.format_output(res2, max_display=5) + "\n\n")

        # Example 3: No-match intersection
        res3 = engine.boolean_and_search("சென்னை AND மழை AND விண்கலப்பயணம்99")
        f.write("### Example 3: Intersection with Absent Term (No-match) ###\n")
        f.write(engine.format_output(res3, max_display=5) + "\n\n")
    print(f"Created {and_file}")

    # 4. boolean_or_search.txt
    or_file = os.path.join(output_dir, "boolean_or_search.txt")
    with open(or_file, "w", encoding="utf-8") as f:
        f.write("=" * 66 + "\n")
        f.write("            BOOLEAN OR SEARCH EXECUTION EXAMPLES                 \n")
        f.write("=" * 66 + "\n\n")

        # Example 1: Two major terms
        res1 = engine.boolean_or_search("தமிழ்நாடு OR சென்னை")
        f.write("### Example 1: Two Major Entities 'தமிழ்நாடு OR சென்னை' ###\n")
        f.write(engine.format_output(res1, max_display=5) + "\n\n")

        # Example 2: Three cities
        res2 = engine.boolean_or_search("மதுரை OR கோவை OR திருச்சி")
        f.write("### Example 2: Multi-term Union 'மதுரை OR கோவை OR திருச்சி' ###\n")
        f.write(engine.format_output(res2, max_display=5) + "\n\n")

        # Example 3: Partial match (one term exists, one does not)
        res3 = engine.boolean_or_search("சென்னை OR செவ்வாய்க்கோள்99")
        f.write("### Example 3: Partial Match 'சென்னை OR செவ்வாய்க்கோள்99' ###\n")
        f.write(engine.format_output(res3, max_display=5) + "\n\n")
    print(f"Created {or_file}")

    # 5. phrase_search.txt
    phrase_file = os.path.join(output_dir, "phrase_search.txt")
    with open(phrase_file, "w", encoding="utf-8") as f:
        f.write("=" * 66 + "\n")
        f.write("            EXACT PHRASE SEARCH EXECUTION EXAMPLES               \n")
        f.write("=" * 66 + "\n\n")

        # Example 1: Standard 2-word phrase
        res1 = engine.exact_phrase_search("தமிழக அரசு")
        f.write("### Example 1: Two-Word Exact Phrase 'தமிழக அரசு' ###\n")
        f.write(engine.format_output(res1, max_display=5) + "\n\n")

        # Example 2: Three-word phrase
        res2 = engine.exact_phrase_search("மாநகர் போக்குவரத்துக் கழகம்")
        f.write("### Example 2: Multi-Word Phrase 'மாநகர் போக்குவரத்துக் கழகம்' ###\n")
        f.write(engine.format_output(res2, max_display=5) + "\n\n")

        # Example 3: Reversed order comparison (demonstrating position sensitivity)
        res3 = engine.exact_phrase_search("அரசு தமிழக")
        f.write("### Example 3: Word-Order Sensitivity: 'அரசு தமிழக' vs 'தமிழக அரசு' ###\n")
        f.write(f"Note: 'தமிழக அரசு' yielded {res1.total_matches} documents, while reversed 'அரசு தமிழக' yielded only {res3.total_matches} documents.\n\n")
        f.write(engine.format_output(res3, max_display=5) + "\n\n")

        # Example 4: No-match phrase
        res4 = engine.exact_phrase_search("செவ்வாய் கிரகத்தில் தக்காளி விவசாயம்")
        f.write("### Example 4: Absent Phrase (No-match) ###\n")
        f.write(engine.format_output(res4, max_display=5) + "\n\n")
    print(f"Created {phrase_file}")

    # 6. index_example.txt
    example_file = os.path.join(output_dir, "index_example.txt")
    with open(example_file, "w", encoding="utf-8") as f:
        f.write("=" * 66 + "\n")
        f.write("      INDEX POSTINGS & POSITIONAL POSTINGS DETAILED EXAMPLES     \n")
        f.write("=" * 66 + "\n\n")

        terms_to_inspect = ["தமிழ்நாடு", "சென்னை", "அரசு"]
        for term in terms_to_inspect:
            dict_entry = engine.dictionary.get(term, {})
            inv_postings = engine.inverted_index.get(term, [])
            pos_postings = engine.positional_index.get(term, {})

            f.write("-" * 66 + "\n")
            f.write(f"TERM: '{term}'\n")
            f.write("-" * 66 + "\n")
            f.write(f"Document Frequency (df):    {dict_entry.get('df', 0):,} documents\n")
            f.write(f"Collection Frequency (cf):  {dict_entry.get('cf', 0):,} total occurrences\n\n")

            f.write(f"Regular Inverted Index Postings List ({len(inv_postings):,} document IDs):\n")
            f.write(f"  First 30 Document IDs: {inv_postings[:30]}\n")
            if len(inv_postings) > 30:
                f.write(f"  ... ({len(inv_postings) - 30} more documents omitted)\n\n")

            f.write(f"Positional Inverted Index Postings Sample (First 5 documents):\n")
            sample_docs = list(pos_postings.keys())[:5]
            for did in sample_docs:
                positions = pos_postings[did]
                doc_title = engine.documents.get(did, {}).get("title", "")
                f.write(f"  Doc ID {did} (Occurrences={len(positions)}):\n")
                f.write(f"    Positions: {positions}\n")
                f.write(f"    Title:     {doc_title}\n")
            f.write("\n")
        f.write("=" * 66 + "\n")
    print(f"Created {example_file}")


if __name__ == "__main__":
    generate_all_outputs()
