================================================================================
INFORMATION RETRIEVAL (IR-M2026) - ASSIGNMENT 1
Positional Inverted Index for Indian Language (Tamil)
================================================================================

Student Name : Tharun Athithya
Roll Number  : S20240010238
Repository   : https://github.com/TharunAthithya/IR-Assignment
Language     : Tamil (Indic Language)

--------------------------------------------------------------------------------
1. PROJECT OVERVIEW & PROBLEM STATEMENT
--------------------------------------------------------------------------------
This project implements a disk-based positional inverted index system tailored
for an Indian language (Tamil), satisfying all requirements of Assignment 1:
  - Supports an Indian language with complex Unicode combining characters.
  - Keeps the vocabulary in main memory (RAM) and postings on secondary memory (Disk).
  - Stores term positions, Collection Frequency (cf), and Document Frequency (df).
  - Evaluates single-term, multi-term, and exact phrase queries.

Why Tamil?
Tamil text poses specific challenges for information retrieval:
1. Combining Characters: Consonants combine with vowel signs (matras, e.g., ா, ி, ு)
   and the pulli/virama (்). Standard tokenizers detach these marks, producing
   garbled roots. A Unicode-aware regex tokenizer is required.
2. Canonical Normalization: Unicode can represent Tamil characters in composed (NFC)
   or decomposed (NFD) form. Strict NFC normalization ensures consistent matching.
3. Proximity & Phrase Queries: A standard inverted index only tells if words appear
   in a document. Positional indexing is needed to enforce exact word order (e.g.,
   "தமிழ்நாடு அரசு" vs "அரசு தமிழ்நாடு").


--------------------------------------------------------------------------------
2. DATASET & CORPUS STATISTICS
--------------------------------------------------------------------------------
Dataset Source: Official Tamil Wikipedia XML Dump (Wikimedia)
Download URL  : https://dumps.wikimedia.org/tawiki/
Raw File      : data/wikipedia/tawiki-latest-pages-articles-multistream.xml
Extracted File: data/extracted/articles.jsonl

Corpus Statistics (Verified from Built Index):
  - Number of Wikipedia articles indexed : 30
  - Total tokens indexed                 : 26,410
  - Vocabulary size (unique terms)       : 12,230
  - Total term-document postings         : 16,774
  - Total positional occurrences         : 26,410
  - Average document length              : 880.33 tokens
  - Index construction time              : ~0.16 seconds

Storage Footprint:
  - Vocabulary file size (RAM)           : 0.98 MB (1,027,863 bytes)
  - Postings file size (Disk)            : 0.25 MB (266,934 bytes)
  - Document metadata file size (RAM)    : 1.77 KB (1,769 bytes)

Note on Large Dataset:
As permitted by the assignment guidelines, the full multi-gigabyte raw Wikipedia
dump is omitted from this submission zip. The extraction script (extract_wiki.py)
and sample dataset are provided, and full dumps can be fetched using the link above.


--------------------------------------------------------------------------------
3. ARCHITECTURE: RAM VS. SECONDARY STORAGE
--------------------------------------------------------------------------------
The indexing engine implements the required separation between main memory
and secondary storage:

[MAIN MEMORY (RAM)]
  1. vocabulary.json:
     Loaded into RAM when the search engine initializes.
     Maps each term to its Document Frequency (df), Collection Frequency (cf),
     byte offset, and byte length:
       {
         "தமிழ்நாடு": {
           "df": 14,
           "cf": 78,
           "offset": 45120,
           "length": 312
         }
       }
  2. doc_meta.json:
     Maps document IDs (0..N-1) to article titles for rapid search result display.

[SECONDARY STORAGE (DISK)]
  1. postings.dat:
     Stored on disk. Contains the serialized JSON postings for every term:
       {"doc_id_0": [pos0, pos1, ...], "doc_id_1": [pos0, ...]}
  2. On-demand seek:
     When querying a term, the engine looks up the term in vocabulary.json (RAM),
     reads its byte offset and length, seeks directly to that byte in postings.dat
     (disk), reads only the required slice, and decodes the postings.
     The full postings file is NEVER loaded into memory.


--------------------------------------------------------------------------------
4. MODULE DESCRIPTIONS
--------------------------------------------------------------------------------
- preprocess.py:
  Contains the Tamil text processing pipeline:
  - normalize_text(): Applies Unicode NFC normalization and strips non-breaking
    or zero-width spaces.
  - tokenize(): Uses Unicode-aware regular expression
    r'[\p{L}\p{N}][\p{L}\p{M}\p{N}]*' to preserve Tamil base consonants,
    combining vowel signs (matras), pulli/virama, and alphanumeric tokens.

- extract_wiki.py:
  Extracts articles from the Wikipedia XML dump using WikiExtractor, removes
  MediaWiki syntax, filters out redirects and short stubs, and outputs clean
  JSON lines ({"id": ..., "title": ..., "text": ...}) into data/extracted/.

- build_index.py:
  Constructs the disk-based positional inverted index:
  1. Scans articles.jsonl and extracts zero-based token positions for combined
     "{title} {text}".
  2. Computes Document Frequency (df) and Collection Frequency (cf) for each term.
  3. Writes positional postings to postings.dat on secondary storage while
     tracking byte offsets.
  4. Saves vocabulary.json (with offsets), doc_meta.json, and metadata.json.

- search.py:
  Core retrieval engine (TamilSearchEngine):
  - Loads vocabulary.json into RAM.
  - get_postings(term): Seeks into postings.dat and retrieves postings for a term.
  - single_term_search(): Direct lookup of a single term.
  - multi_term_search(): Union of documents containing any query term.
  - phrase_search(): Verifies exact consecutive token offsets (pos + 1) across
    all phrase words in positional postings.
  - Snippet generation centered on match positions.

- main.py:
  Central command-line interface providing subcommands:
  build, search, phrase, inspect, stats, generate, package, and test.

- generate_outputs.py:
  Generates the assignment deliverable output file containing:
  - Part 1: 5 Test Cases (Common term, Less frequent term, Phrase query,
            Multi-term query, Unknown edge case).
  - Part 2: 10 Rare Terms with their positional postings.
  - Part 3: Index Statistics report.

- tests/ (Test Suite):
  40 automated test cases across 4 test modules:
  - test_preprocessing.py: Unicode NFC, combining marks, Grantha consonants.
  - test_indexing.py: Index construction, byte offset accuracy, df/cf values.
  - test_search.py: Term search, phrase search, disk read verification.
  - test_integration.py: End-to-end index and search pipeline validation.


--------------------------------------------------------------------------------
5. STEP-BY-STEP EXECUTION GUIDE
--------------------------------------------------------------------------------
Prerequisites:
  - Python 3.10 or higher
  - pip

Step 1: Install dependencies
  pip install -r requirements.txt

Step 2: Build the Positional Index
  python main.py build
  (Reads articles from data/extracted/articles.jsonl and creates indexes/)

Step 3: Run Searches via CLI
  - Single Term Search:
      python main.py search "தமிழ்நாடு"

  - Exact Phrase Search:
      python main.py phrase "ஒரு பெரிய"

  - Multi-Term Search:
      python search.py --query "தமிழ்நாடு சென்னை" --mode multi

Step 4: Inspect a Term's Positional Postings on Disk
  python main.py inspect "தமிழ்நாடு"

Step 5: View Index Statistics
  python main.py stats

Step 6: Run Automated Tests
  python main.py test
  (Runs all 40 tests to verify preprocessing, disk index, and phrase search)

Step 7: Generate Assignment Outputs
  python main.py generate
  (Creates test_cases.txt, rare_terms.txt, index_statistics.txt, and assignment_output.txt)

Step 8: Package Submission Deliverables
  python main.py package --roll S20240010238
  (Creates S20240010238-src.zip and S20240010238-A1-PostIndex-IR-M2026.zip)


--------------------------------------------------------------------------------
6. SUBMISSION DELIVERABLES STRUCTURE
--------------------------------------------------------------------------------
The final compressed archive S20240010238-A1-PostIndex-IR-M2026.zip contains
strictly the 3 required deliverable files:

  S20240010238-A1-PostIndex-IR-M2026.zip
  |-- S20240010238-README.txt  (This documentation file)
  |-- S20240010238-src.zip     (Compressed source code and test suite)
  `-- S20240010238-output.txt  (5 test cases, 10 rare-term positional indexes, stats)

Contents of S20240010238-src.zip:
  - preprocess.py
  - extract_wiki.py
  - build_index.py
  - search.py
  - main.py
  - generate_outputs.py
  - requirements.txt
  - tests/
      - __init__.py
      - test_preprocessing.py
      - test_indexing.py
      - test_search.py
      - test_integration.py
================================================================================
