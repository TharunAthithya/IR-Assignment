"""Disk-Based Tamil Information Retrieval Search Module.

This module implements a search engine that keeps the vocabulary in RAM and
retrieves positional postings from secondary storage (disk) on demand.

Architecture:
    - vocabulary.json is loaded into RAM at initialization. It contains each
      term's df, cf, byte offset, and byte length in postings.dat.
    - postings.dat remains on disk. When a query term is looked up, only that
      term's postings are read by seeking to the recorded byte offset and
      reading exactly the recorded number of bytes.
    - doc_meta.json is loaded into RAM for displaying document titles.

Search Modes:
    1. Single-Term Search — direct vocabulary lookup + disk postings retrieval.
    2. Multi-Term Search — retrieves postings for each term, returns all docs.
    3. Phrase Search — retrieves postings for each phrase token, verifies
       consecutive positional alignment.
    4. Unknown-term handling — returns empty results gracefully.

All query text is normalized and tokenized using the same preprocess module
as the indexer, ensuring consistency.

Positions are zero-based throughout the implementation.
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

# Reconfigure stdout/stdin to UTF-8
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from preprocess import normalize_text, tokenize


@dataclass
class SearchResultItem:
    """Represents a single matching document in a search result."""
    doc_id: int
    title: str
    matched_positions: Optional[List[int]] = None


@dataclass
class SearchResult:
    """Encapsulates the complete search result set and execution metadata."""
    query: str
    search_type: str
    total_matches: int
    search_time_ms: float
    matching_doc_ids: List[int] = field(default_factory=list)
    matching_positions: Optional[Dict[int, List[int]]] = None
    items: List[SearchResultItem] = field(default_factory=list)
    query_terms: List[str] = field(default_factory=list)
    error_message: Optional[str] = None


class TamilSearchEngine:
    """Disk-based search engine: vocabulary in RAM, postings on secondary storage.

    The vocabulary (vocabulary.json) is loaded into memory at initialization.
    The postings file (postings.dat) is accessed via random file seeks — only
    the postings for queried terms are read from disk, never the entire file.

    Attributes:
        vocabulary: Dict mapping term -> {df, cf, offset, length}
        doc_meta: Dict mapping doc_id_str -> title
        metadata: Dict with index construction statistics
        postings_path: Path to the postings.dat file on disk
    """

    def __init__(self, index_dir: str = "indexes"):
        """Initializes the search engine by loading vocabulary into RAM.

        Args:
            index_dir: Directory containing index files.

        Raises:
            FileNotFoundError: If required index files are missing.
        """
        self.index_dir = index_dir
        self.vocabulary: Dict[str, Dict[str, int]] = {}
        self.doc_meta: Dict[str, str] = {}
        self.metadata: Dict[str, Any] = {}
        self.postings_path: str = ""
        self.is_loaded: bool = False

        self._load_index(index_dir)

    def _load_index(self, index_dir: str) -> None:
        """Loads vocabulary and document metadata into RAM.

        Note: postings.dat is NOT loaded. It is accessed on-demand via seek.

        Args:
            index_dir: Directory containing index files.
        """
        required_files = {
            "vocabulary.json": "Vocabulary index",
            "postings.dat": "Positional postings (disk)",
            "doc_meta.json": "Document metadata",
            "metadata.json": "Index metadata",
        }

        for fname, desc in required_files.items():
            fpath = os.path.join(index_dir, fname)
            if not os.path.exists(fpath):
                raise FileNotFoundError(
                    f"Index file '{fname}' ({desc}) not found in '{index_dir}'. "
                    f"Please run 'python main.py build' first."
                )

        # Load vocabulary into RAM
        vocab_path = os.path.join(index_dir, "vocabulary.json")
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocabulary = json.load(f)

        # Load document metadata into RAM
        doc_meta_path = os.path.join(index_dir, "doc_meta.json")
        with open(doc_meta_path, "r", encoding="utf-8") as f:
            self.doc_meta = json.load(f)

        # Load index metadata
        metadata_path = os.path.join(index_dir, "metadata.json")
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        # Store postings file path for on-demand access
        self.postings_path = os.path.join(index_dir, "postings.dat")

        self.is_loaded = True

    def get_postings(self, term: str) -> Dict[str, List[int]]:
        """Retrieves a term's positional postings from secondary storage (disk).

        This is the core disk-access method. It:
        1. Looks up the term in the in-memory vocabulary.
        2. Reads the term's byte offset and length.
        3. Opens postings.dat and seeks to the byte offset.
        4. Reads exactly 'length' bytes.
        5. Decodes and returns the JSON-parsed postings dict.

        CRITICAL: Only the requested term's postings are read from disk.
        The complete postings file is NEVER loaded into RAM.

        Args:
            term: Normalized term to look up.

        Returns:
            Dict mapping doc_id_str to list of zero-based positions.
            Returns empty dict if term is not in vocabulary.
        """
        vocab_entry = self.vocabulary.get(term)
        if vocab_entry is None:
            return {}

        offset = vocab_entry["offset"]
        length = vocab_entry["length"]

        with open(self.postings_path, "rb") as f:
            f.seek(offset)
            raw_bytes = f.read(length)

        postings = json.loads(raw_bytes.decode("utf-8"))
        return postings

    def get_doc_ids_for_term(self, term: str) -> List[int]:
        """Returns sorted list of document IDs containing the term.

        Uses vocabulary df to check existence first, then retrieves postings
        from disk only if the term exists.

        Args:
            term: Normalized term.

        Returns:
            Sorted list of integer document IDs.
        """
        postings = self.get_postings(term)
        if not postings:
            return []
        return sorted(int(did) for did in postings.keys())

    # -------------------------------------------------------------------------
    # Core Postings List Operations
    # -------------------------------------------------------------------------

    @staticmethod
    def intersect_postings(list_a: List[int], list_b: List[int]) -> List[int]:
        """Two-pointer intersection of two sorted postings lists. O(|A|+|B|)."""
        result: List[int] = []
        i, j = 0, 0
        len_a, len_b = len(list_a), len(list_b)
        while i < len_a and j < len_b:
            if list_a[i] == list_b[j]:
                result.append(list_a[i])
                i += 1
                j += 1
            elif list_a[i] < list_b[j]:
                i += 1
            else:
                j += 1
        return result

    @staticmethod
    def union_postings(list_a: List[int], list_b: List[int]) -> List[int]:
        """Two-pointer union of two sorted postings lists. O(|A|+|B|)."""
        result: List[int] = []
        i, j = 0, 0
        len_a, len_b = len(list_a), len(list_b)
        while i < len_a and j < len_b:
            if list_a[i] == list_b[j]:
                result.append(list_a[i])
                i += 1
                j += 1
            elif list_a[i] < list_b[j]:
                result.append(list_a[i])
                i += 1
            else:
                result.append(list_b[j])
                j += 1
        while i < len_a:
            result.append(list_a[i])
            i += 1
        while j < len_b:
            result.append(list_b[j])
            j += 1
        return result

    # -------------------------------------------------------------------------
    # Search Modes
    # -------------------------------------------------------------------------

    def single_term_search(self, query: str, max_results: int = 50) -> SearchResult:
        """Single-term search: tokenize query → retrieve postings from disk.

        Args:
            query: Tamil or alphanumeric query term.
            max_results: Maximum number of result items to return.

        Returns:
            SearchResult with matching documents and positions.
        """
        t0 = time.perf_counter()
        tokens = tokenize(query)

        if not tokens:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return SearchResult(
                query=query, search_type="Single-Term Search",
                total_matches=0, search_time_ms=round(elapsed_ms, 3),
                error_message="Query contains no valid searchable terms.",
            )

        term = tokens[0]

        # Retrieve postings from disk
        postings = self.get_postings(term)
        sorted_doc_ids = sorted(int(did) for did in postings.keys())

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Build result items with positions
        items = []
        positions_map: Dict[int, List[int]] = {}
        for did in sorted_doc_ids[:max_results]:
            did_str = str(did)
            title = self.doc_meta.get(did_str, "")
            positions = postings.get(did_str, [])
            positions_map[did] = positions
            items.append(SearchResultItem(
                doc_id=did, title=title, matched_positions=positions,
            ))

        return SearchResult(
            query=query, search_type="Single-Term Search",
            total_matches=len(sorted_doc_ids),
            search_time_ms=round(elapsed_ms, 3),
            matching_doc_ids=sorted_doc_ids,
            matching_positions=positions_map,
            items=items, query_terms=[term],
        )

    def multi_term_search(self, query: str, max_results: int = 50) -> SearchResult:
        """Multi-term search: returns documents containing ANY of the query terms.

        Args:
            query: Space-separated Tamil terms.
            max_results: Maximum result items.

        Returns:
            SearchResult with union of matching documents.
        """
        t0 = time.perf_counter()
        tokens = tokenize(query)
        unique_terms = list(dict.fromkeys(tokens))

        if not unique_terms:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return SearchResult(
                query=query, search_type="Multi-Term Search",
                total_matches=0, search_time_ms=round(elapsed_ms, 3),
                error_message="Query contains no valid searchable terms.",
            )

        if len(unique_terms) == 1:
            return self.single_term_search(unique_terms[0], max_results=max_results)

        # Retrieve and merge postings from disk for each term
        result_doc_ids: List[int] = []
        all_postings: Dict[str, Dict[str, List[int]]] = {}

        for term in unique_terms:
            postings = self.get_postings(term)
            all_postings[term] = postings
            term_doc_ids = sorted(int(did) for did in postings.keys())
            if not result_doc_ids:
                result_doc_ids = term_doc_ids
            else:
                result_doc_ids = self.union_postings(result_doc_ids, term_doc_ids)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Build result items
        items = []
        for did in result_doc_ids[:max_results]:
            did_str = str(did)
            title = self.doc_meta.get(did_str, "")
            items.append(SearchResultItem(doc_id=did, title=title))

        return SearchResult(
            query=query, search_type="Multi-Term Search",
            total_matches=len(result_doc_ids),
            search_time_ms=round(elapsed_ms, 3),
            matching_doc_ids=result_doc_ids,
            items=items, query_terms=unique_terms,
        )

    def phrase_search(self, phrase: str, max_results: int = 50) -> SearchResult:
        """Exact phrase search using positional information from disk.

        Verifies that all tokens appear consecutively in the correct order:
        pos(w_0) = p, pos(w_1) = p+1, ..., pos(w_{k-1}) = p+k-1

        Args:
            phrase: Exact phrase string (quotes stripped if present).
            max_results: Maximum result items.

        Returns:
            SearchResult with matching documents and their match start positions.
        """
        t0 = time.perf_counter()

        # Strip enclosing quotes
        cleaned = phrase.strip().strip('"').strip("'").strip("\u201c").strip("\u201d")
        tokens = tokenize(cleaned)

        if not tokens:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return SearchResult(
                query=phrase, search_type="Phrase Search",
                total_matches=0, search_time_ms=round(elapsed_ms, 3),
                error_message="Phrase query contains no valid tokens.",
            )

        # Single-token phrase → degenerate to single-term search
        if len(tokens) == 1:
            result = self.single_term_search(tokens[0], max_results=max_results)
            result.query = phrase
            result.search_type = "Phrase Search"
            return result

        # Retrieve postings from disk for each term
        term_postings: Dict[str, Dict[str, List[int]]] = {}
        for t in tokens:
            if t not in self.vocabulary:
                # If any term is missing, phrase cannot match
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                return SearchResult(
                    query=phrase, search_type="Phrase Search",
                    total_matches=0, search_time_ms=round(elapsed_ms, 3),
                    matching_doc_ids=[], items=[], query_terms=tokens,
                )
            term_postings[t] = self.get_postings(t)

        # Candidate pruning: documents must contain ALL distinct terms
        distinct_terms = list(dict.fromkeys(tokens))
        # Start with docs containing the rarest term
        distinct_terms.sort(key=lambda t: self.vocabulary[t]["df"])
        candidate_docs: Set[str] = set(term_postings[distinct_terms[0]].keys())
        for t in distinct_terms[1:]:
            candidate_docs &= set(term_postings[t].keys())
            if not candidate_docs:
                break

        if not candidate_docs:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return SearchResult(
                query=phrase, search_type="Phrase Search",
                total_matches=0, search_time_ms=round(elapsed_ms, 3),
                matching_doc_ids=[], items=[], query_terms=tokens,
            )

        # Positional verification
        first_term = tokens[0]
        phrase_len = len(tokens)
        sorted_candidates = sorted(int(did) for did in candidate_docs)

        matching_doc_ids: List[int] = []
        matching_pos_map: Dict[int, List[int]] = {}

        for doc_id in sorted_candidates:
            doc_id_str = str(doc_id)
            first_positions = term_postings[first_term].get(doc_id_str, [])
            if not first_positions:
                continue

            # Pre-fetch position sets for subsequent tokens
            subsequent_sets: List[Set[int]] = []
            for k in range(1, phrase_len):
                pos_list = term_postings[tokens[k]].get(doc_id_str, [])
                subsequent_sets.append(set(pos_list))

            # Check for consecutive positions
            matched_starts: List[int] = []
            for start_pos in first_positions:
                match = True
                for offset, pos_set in enumerate(subsequent_sets, start=1):
                    if (start_pos + offset) not in pos_set:
                        match = False
                        break
                if match:
                    matched_starts.append(start_pos)

            if matched_starts:
                matching_doc_ids.append(doc_id)
                matching_pos_map[doc_id] = matched_starts

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Build result items
        items = []
        for did in matching_doc_ids[:max_results]:
            did_str = str(did)
            title = self.doc_meta.get(did_str, "")
            items.append(SearchResultItem(
                doc_id=did, title=title,
                matched_positions=matching_pos_map.get(did),
            ))

        return SearchResult(
            query=phrase, search_type="Phrase Search",
            total_matches=len(matching_doc_ids),
            search_time_ms=round(elapsed_ms, 3),
            matching_doc_ids=matching_doc_ids,
            matching_positions=matching_pos_map,
            items=items, query_terms=tokens,
        )

    # -------------------------------------------------------------------------
    # Term Inspection
    # -------------------------------------------------------------------------

    def get_term_info(self, term_input: str) -> Optional[Dict[str, Any]]:
        """Returns detailed information about a term including postings from disk.

        Args:
            term_input: Raw term input (will be tokenized).

        Returns:
            Dict with term info, or None if not found.
        """
        tokens = tokenize(term_input)
        if not tokens:
            return None

        term = tokens[0]
        vocab_entry = self.vocabulary.get(term)
        if vocab_entry is None:
            return None

        # Retrieve full postings from disk
        postings = self.get_postings(term)

        # Build detailed info
        sample_docs = {}
        for i, (did_str, positions) in enumerate(sorted(postings.items(), key=lambda x: int(x[0]))):
            if i >= 5:
                break
            title = self.doc_meta.get(did_str, "")
            sample_docs[did_str] = {
                "title": title,
                "positions": positions,
                "occurrences": len(positions),
            }

        return {
            "term": term,
            "df": vocab_entry["df"],
            "cf": vocab_entry["cf"],
            "offset": vocab_entry["offset"],
            "length": vocab_entry["length"],
            "total_docs": len(postings),
            "sample_docs": sample_docs,
        }

    # -------------------------------------------------------------------------
    # Output Formatting
    # -------------------------------------------------------------------------

    @staticmethod
    def format_output(result: SearchResult, max_display: int = 10) -> str:
        """Formats the search result into readable text output."""
        lines = []
        lines.append("=" * 60)
        lines.append("TAMIL WIKIPEDIA INFORMATION RETRIEVAL SYSTEM")
        lines.append("=" * 60)
        lines.append(f"Query: {result.query}")
        lines.append(f"Search Type: {result.search_type}")
        lines.append(f"Query Terms: {result.query_terms}")
        lines.append(f"Matching Documents: {result.total_matches}")

        if result.error_message:
            lines.append(f"Message: {result.error_message}")

        if result.total_matches > 0:
            display_count = min(len(result.items), max_display)
            for i in range(display_count):
                item = result.items[i]
                lines.append("-" * 60)
                lines.append(f"Result #{i + 1}")
                lines.append(f"  Document ID: {item.doc_id}")
                lines.append(f"  Title: {item.title}")
                if item.matched_positions is not None:
                    lines.append(f"  Positions: {item.matched_positions}")

            if result.total_matches > display_count:
                remaining = result.total_matches - display_count
                lines.append("-" * 60)
                lines.append(
                    f"(Showing {display_count} of {result.total_matches} total. "
                    f"{remaining} more omitted.)"
                )

        lines.append(f"Search Time: {result.search_time_ms:.2f} ms")
        lines.append("=" * 60)
        return "\n".join(lines)


def main():
    """CLI driver for standalone search queries."""
    parser = argparse.ArgumentParser(
        description="Search Tamil Wikipedia Positional Index"
    )
    parser.add_argument(
        "--query", "-q", type=str, required=True,
        help="Tamil or alphanumeric search query",
    )
    parser.add_argument(
        "--mode", "-m", type=str,
        choices=["term", "multi", "phrase"],
        default="term",
        help="Search mode: term (single), multi (multiple terms), phrase (exact phrase)",
    )
    parser.add_argument(
        "--index-dir", type=str, default="indexes",
        help="Path to indexes directory",
    )
    parser.add_argument(
        "--limit", type=int, default=10,
        help="Max results to display",
    )

    args = parser.parse_args()

    engine = TamilSearchEngine(index_dir=args.index_dir)

    if args.mode == "term":
        res = engine.single_term_search(args.query)
    elif args.mode == "multi":
        res = engine.multi_term_search(args.query)
    elif args.mode == "phrase":
        res = engine.phrase_search(args.query)
    else:
        print(f"Unknown mode: {args.mode}")
        return

    print(engine.format_output(res, max_display=args.limit))


if __name__ == "__main__":
    main()
