"""Tamil Information Retrieval Search Module.

This module implements the retrieval engine supporting four core search modes:
1. Single-Term Search:
   - Fast lookup using the regular inverted index.
   - Retrieves all documents containing the normalized query term.

2. Boolean AND Search:
   - Postings-list intersection algorithm.
   - Evaluates queries containing multiple terms (e.g., 'சென்னை AND மழை').
   - Sorts postings by increasing document frequency (size) to optimize intersection speed.
   - Returns only documents containing all query terms.

3. Boolean OR Search:
   - Postings-list union algorithm.
   - Evaluates queries containing multiple terms (e.g., 'தமிழ்நாடு OR சென்னை').
   - Returns documents containing at least one of the query terms.

4. Exact Phrase Search:
   - Positional inverted index verification.
   - Evaluates multiword phrases (e.g., '"தமிழக அரசு"').
   - Filters candidate documents containing all distinct phrase terms.
   - Enforces strict consecutive token ordering: word_k at position p_0 + k.
   - Handles multi-word phrases, repeated terms, and edge cases.
   - Reports exact matching start positions for every matching document.

Also handles:
- Relevant text snippet extraction centered around query terms.
- Academic performance measurement (pure search execution time in milliseconds).
- Deterministic document ordering.
- Comprehensive edge-case handling (missing terms, empty queries, punctuation).
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
    snippet: str
    author: str = ""
    city: str = ""
    published: str = ""
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
    """Search Engine implementation for Tamil text using Inverted and Positional indexes."""

    def __init__(self, index_dir: str = "indexes"):
        """Initializes the search engine by loading indexes from disk.

        Args:
            index_dir: Directory containing pre-built JSON index files.
        """
        self.index_dir = index_dir
        self.inverted_index: Dict[str, List[int]] = {}
        self.positional_index: Dict[str, Dict[str, List[int]]] = {}
        self.dictionary: Dict[str, Dict[str, int]] = {}
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.metadata: Dict[str, Any] = {}
        self.is_loaded: bool = False

        self.load_indexes()

    def load_indexes(self) -> None:
        """Loads inverted, positional, dictionary, and document stores into memory.

        Raises:
            FileNotFoundError: If index directory or required JSON files are missing.
        """
        required_files = [
            "inverted_index.json",
            "positional_index.json",
            "dictionary.json",
            "documents.json",
            "metadata.json",
        ]

        for fname in required_files:
            fpath = os.path.join(self.index_dir, fname)
            if not os.path.exists(fpath):
                raise FileNotFoundError(
                    f"Index file '{fname}' not found in '{self.index_dir}'. "
                    f"Please run build_index.py first."
                )

        # Load regular inverted index
        with open(os.path.join(self.index_dir, "inverted_index.json"), "r", encoding="utf-8") as f:
            self.inverted_index = json.load(f)

        # Load positional index
        with open(os.path.join(self.index_dir, "positional_index.json"), "r", encoding="utf-8") as f:
            self.positional_index = json.load(f)

        # Load vocabulary dictionary
        with open(os.path.join(self.index_dir, "dictionary.json"), "r", encoding="utf-8") as f:
            self.dictionary = json.load(f)

        # Load documents store
        with open(os.path.join(self.index_dir, "documents.json"), "r", encoding="utf-8") as f:
            self.documents = json.load(f)

        # Load metadata
        with open(os.path.join(self.index_dir, "metadata.json"), "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.is_loaded = True

    # -------------------------------------------------------------------------
    # Core Postings List Operations (Academic Set Intersect & Union)
    # -------------------------------------------------------------------------

    @staticmethod
    def intersect_postings(list_a: List[int], list_b: List[int]) -> List[int]:
        """Performs two-pointer intersection of two sorted postings lists.

        Time Complexity: O(|list_a| + |list_b|)

        Args:
            list_a: First sorted list of integer document IDs.
            list_b: Second sorted list of integer document IDs.

        Returns:
            Sorted list of document IDs present in both lists.
        """
        result: List[int] = []
        i, j = 0, 0
        len_a, len_b = len(list_a), len(list_b)

        while i < len_a and j < len_b:
            doc_a = list_a[i]
            doc_b = list_b[j]
            if doc_a == doc_b:
                result.append(doc_a)
                i += 1
                j += 1
            elif doc_a < doc_b:
                i += 1
            else:
                j += 1
        return result

    @staticmethod
    def union_postings(list_a: List[int], list_b: List[int]) -> List[int]:
        """Performs two-pointer union of two sorted postings lists.

        Time Complexity: O(|list_a| + |list_b|)

        Args:
            list_a: First sorted list of integer document IDs.
            list_b: Second sorted list of integer document IDs.

        Returns:
            Sorted deduplicated list of document IDs present in either list.
        """
        result: List[int] = []
        i, j = 0, 0
        len_a, len_b = len(list_a), len(list_b)

        while i < len_a and j < len_b:
            doc_a = list_a[i]
            doc_b = list_b[j]
            if doc_a == doc_b:
                result.append(doc_a)
                i += 1
                j += 1
            elif doc_a < doc_b:
                result.append(doc_a)
                i += 1
            else:
                result.append(doc_b)
                j += 1

        # Append remaining elements
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

    def single_term_search(self, query: str, prefetch_items: int = 50) -> SearchResult:
        """Executes a single-term search using the regular inverted index.

        Args:
            query: Tamil or alphanumeric query term.
            prefetch_items: Number of top result items to pre-render snippets for.

        Returns:
            SearchResult object with matching documents and snippet context.
        """
        t0 = time.perf_counter()
        tokens = tokenize(query)

        if not tokens:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return SearchResult(
                query=query,
                search_type="Single-Term Search",
                total_matches=0,
                search_time_ms=round(elapsed_ms, 3),
                error_message="Query contains no valid searchable terms.",
            )

        term = tokens[0]
        # Direct O(1) lookup into inverted index dictionary
        postings = self.inverted_index.get(term, [])

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Build initial items with snippets
        items = self.get_result_items(postings[:prefetch_items], [term])

        return SearchResult(
            query=query,
            search_type="Single-Term Search",
            total_matches=len(postings),
            search_time_ms=round(elapsed_ms, 3),
            matching_doc_ids=postings,
            items=items,
            query_terms=[term],
        )

    def boolean_and_search(self, query: str, prefetch_items: int = 50) -> SearchResult:
        """Executes Boolean AND search across two or more terms.

        Query format can be:
        - Explicit: 'சென்னை AND மழை' or 'சென்னை and மழை'
        - Multi-term: 'சென்னை AND மழை AND வானிலை'

        Uses postings-list intersection, optimized by ordering terms by increasing
        document frequency (df) to minimize intermediate postings list sizes.

        Args:
            query: String containing query terms separated by AND.
            prefetch_items: Number of top result items to pre-render snippets for.

        Returns:
            SearchResult containing documents matching ALL terms.
        """
        t0 = time.perf_counter()

        # Split query terms removing the literal Boolean operator 'AND' / 'and'
        raw_parts = [p.strip() for p in query.split(" AND ") if p.strip()]
        if len(raw_parts) <= 1:
            raw_parts = [p.strip() for p in query.split(" and ") if p.strip()]

        terms: List[str] = []
        for part in raw_parts:
            terms.extend(tokenize(part))

        # Deduplicate terms while preserving order
        unique_terms = list(dict.fromkeys(terms))

        if not unique_terms:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return SearchResult(
                query=query,
                search_type="Boolean AND Search",
                total_matches=0,
                search_time_ms=round(elapsed_ms, 3),
                error_message="Query contains no valid terms for Boolean AND search.",
            )

        if len(unique_terms) == 1:
            return self.single_term_search(unique_terms[0], prefetch_items=prefetch_items)

        # Optimize intersection order by sorting terms by Document Frequency (df) ascending
        terms_with_df = []
        for t in unique_terms:
            postings = self.inverted_index.get(t, [])
            terms_with_df.append((t, postings))

        # If any term has 0 postings, intersection is empty immediately!
        if any(len(p) == 0 for _, p in terms_with_df):
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return SearchResult(
                query=query,
                search_type="Boolean AND Search",
                total_matches=0,
                search_time_ms=round(elapsed_ms, 3),
                matching_doc_ids=[],
                items=[],
                query_terms=unique_terms,
            )

        # Sort by postings list length ascending
        terms_with_df.sort(key=lambda x: len(x[1]))

        # Iterative intersection
        result_postings = terms_with_df[0][1]
        for _, next_postings in terms_with_df[1:]:
            result_postings = self.intersect_postings(result_postings, next_postings)
            if not result_postings:
                break

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        items = self.get_result_items(result_postings[:prefetch_items], unique_terms)

        return SearchResult(
            query=query,
            search_type="Boolean AND Search",
            total_matches=len(result_postings),
            search_time_ms=round(elapsed_ms, 3),
            matching_doc_ids=result_postings,
            items=items,
            query_terms=unique_terms,
        )

    def boolean_or_search(self, query: str, prefetch_items: int = 50) -> SearchResult:
        """Executes Boolean OR search across two or more terms.

        Query format:
        - Explicit: 'தமிழ்நாடு OR சென்னை' or 'தமிழ்நாடு or சென்னை'

        Uses postings-list union to merge matching document sets.

        Args:
            query: String containing query terms separated by OR.
            prefetch_items: Number of top result items to pre-render snippets for.

        Returns:
            SearchResult containing documents matching AT LEAST ONE term.
        """
        t0 = time.perf_counter()

        # Split query terms removing the literal Boolean operator 'OR' / 'or'
        raw_parts = [p.strip() for p in query.split(" OR ") if p.strip()]
        if len(raw_parts) <= 1:
            raw_parts = [p.strip() for p in query.split(" or ") if p.strip()]

        terms: List[str] = []
        for part in raw_parts:
            terms.extend(tokenize(part))

        unique_terms = list(dict.fromkeys(terms))

        if not unique_terms:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return SearchResult(
                query=query,
                search_type="Boolean OR Search",
                total_matches=0,
                search_time_ms=round(elapsed_ms, 3),
                error_message="Query contains no valid terms for Boolean OR search.",
            )

        if len(unique_terms) == 1:
            return self.single_term_search(unique_terms[0], prefetch_items=prefetch_items)

        # Iterative union of postings lists
        result_postings: List[int] = []
        for t in unique_terms:
            postings = self.inverted_index.get(t, [])
            if postings:
                if not result_postings:
                    result_postings = list(postings)
                else:
                    result_postings = self.union_postings(result_postings, postings)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        items = self.get_result_items(result_postings[:prefetch_items], unique_terms)

        return SearchResult(
            query=query,
            search_type="Boolean OR Search",
            total_matches=len(result_postings),
            search_time_ms=round(elapsed_ms, 3),
            matching_doc_ids=result_postings,
            items=items,
            query_terms=unique_terms,
        )

    def exact_phrase_search(self, phrase: str, prefetch_items: int = 50) -> SearchResult:
        """Executes an exact phrase search using the positional inverted index.

        Strictly enforces that all tokens of the phrase appear consecutively and in
        the specified order:
        pos(w_0) = p, pos(w_1) = p + 1, ..., pos(w_{k-1}) = p + (k - 1)

        Handles:
        - Multiword phrases (2, 3, or more words).
        - Repeated words within a phrase (e.g., 'நாடு நாடு').
        - Mixed Tamil and English terms.
        - Accurate start position reporting.

        Args:
            phrase: Exact phrase string (quotes are stripped if present).
            prefetch_items: Number of top result items to pre-render snippets for.

        Returns:
            SearchResult containing matching documents with their matching positions.
        """
        t0 = time.perf_counter()

        # Clean enclosing quotation marks
        cleaned_phrase = phrase.strip().strip('"').strip("'").strip("“").strip("”")
        tokens = tokenize(cleaned_phrase)

        if not tokens:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return SearchResult(
                query=phrase,
                search_type="Exact Phrase Search",
                total_matches=0,
                search_time_ms=round(elapsed_ms, 3),
                error_message="Phrase query contains no valid tokens.",
            )

        # Single-token phrase degrades to single term lookup with positions
        if len(tokens) == 1:
            term = tokens[0]
            doc_positions = self.positional_index.get(term, {})
            sorted_doc_ids = sorted([int(did) for did in doc_positions.keys()])
            matching_pos_map = {int(did): pos for did, pos in doc_positions.items()}

            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            items = self.get_result_items(
                sorted_doc_ids[:prefetch_items],
                [term],
                positions_map=matching_pos_map,
                phrase_match=True,
            )

            return SearchResult(
                query=phrase,
                search_type="Exact Phrase Search",
                total_matches=len(sorted_doc_ids),
                search_time_ms=round(elapsed_ms, 3),
                matching_doc_ids=sorted_doc_ids,
                matching_positions=matching_pos_map,
                items=items,
                query_terms=[term],
            )

        # Check if every term in the phrase exists in the positional index
        for t in tokens:
            if t not in self.positional_index:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                return SearchResult(
                    query=phrase,
                    search_type="Exact Phrase Search",
                    total_matches=0,
                    search_time_ms=round(elapsed_ms, 3),
                    matching_doc_ids=[],
                    items=[],
                    query_terms=tokens,
                )

        # Candidate Document Pruning:
        # A document must contain all distinct terms in the phrase
        distinct_terms = list(dict.fromkeys(tokens))
        # Start candidate set with documents containing the rarest term
        distinct_terms.sort(key=lambda t: len(self.positional_index[t]))
        candidate_docs: Set[str] = set(self.positional_index[distinct_terms[0]].keys())
        for t in distinct_terms[1:]:
            candidate_docs &= set(self.positional_index[t].keys())
            if not candidate_docs:
                break

        if not candidate_docs:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return SearchResult(
                query=phrase,
                search_type="Exact Phrase Search",
                total_matches=0,
                search_time_ms=round(elapsed_ms, 3),
                matching_doc_ids=[],
                items=[],
                query_terms=tokens,
            )

        # Positional Matching Verification
        # For each candidate document, find all start positions p in positions(token_0)
        # such that for all offset k in 1..len(tokens)-1: (p + k) in positions(token_k)
        first_term = tokens[0]
        phrase_len = len(tokens)

        # Convert candidate document IDs to sorted integers for deterministic output
        sorted_candidates = sorted([int(did) for did in candidate_docs])
        matching_doc_ids: List[int] = []
        matching_pos_map: Dict[int, List[int]] = {}

        for doc_id in sorted_candidates:
            doc_id_str = str(doc_id)
            first_positions = self.positional_index[first_term].get(doc_id_str, [])
            if not first_positions:
                continue

            # Pre-fetch position sets for subsequent tokens in this document
            token_position_sets: List[Set[int]] = []
            for k in range(1, phrase_len):
                token_positions = self.positional_index[tokens[k]].get(doc_id_str, [])
                token_position_sets.append(set(token_positions))

            # Check consecutive offsets
            matched_start_positions: List[int] = []
            for start_pos in first_positions:
                is_phrase_match = True
                for offset, pos_set in enumerate(token_position_sets, start=1):
                    if (start_pos + offset) not in pos_set:
                        is_phrase_match = False
                        break
                if is_phrase_match:
                    matched_start_positions.append(start_pos)

            if matched_start_positions:
                matching_doc_ids.append(doc_id)
                matching_pos_map[doc_id] = matched_start_positions

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        items = self.get_result_items(
            matching_doc_ids[:prefetch_items],
            tokens,
            positions_map=matching_pos_map,
            phrase_match=True,
        )

        return SearchResult(
            query=phrase,
            search_type="Exact Phrase Search",
            total_matches=len(matching_doc_ids),
            search_time_ms=round(elapsed_ms, 3),
            matching_doc_ids=matching_doc_ids,
            matching_positions=matching_pos_map,
            items=items,
            query_terms=tokens,
        )

    # -------------------------------------------------------------------------
    # Snippet Generation & Result Items Fetching
    # -------------------------------------------------------------------------

    def get_result_items(
        self,
        doc_ids: List[int],
        query_terms: List[str],
        positions_map: Optional[Dict[int, List[int]]] = None,
        phrase_match: bool = False,
    ) -> List[SearchResultItem]:
        """Generates detailed SearchResultItems with snippets for a list of doc IDs."""
        items: List[SearchResultItem] = []
        for did in doc_ids:
            doc_data = self.documents.get(str(did), {})
            snippet = self._generate_snippet(doc_data, query_terms, phrase_match=phrase_match)
            matched_pos = positions_map.get(did) if positions_map else None
            items.append(
                SearchResultItem(
                    doc_id=did,
                    title=doc_data.get("title", ""),
                    snippet=snippet,
                    author=doc_data.get("author", ""),
                    city=doc_data.get("city", ""),
                    published=doc_data.get("published", ""),
                    matched_positions=matched_pos,
                )
            )
        return items

    def _generate_snippet(
        self,
        doc_data: Dict[str, Any],
        query_terms: List[str],
        phrase_match: bool = False,
        snippet_length: int = 220,
    ) -> str:
        """Extracts a contextual snippet from document text centered around matching terms."""
        text = doc_data.get("text", "").strip()
        title = doc_data.get("title", "").strip()
        full_content = f"{title}. {text}" if title else text

        if not full_content:
            return ""

        norm_content = normalize_text(full_content)

        match_start = -1
        match_end = -1

        if phrase_match and len(query_terms) > 1:
            phrase_str = " ".join(query_terms)
            idx = norm_content.find(phrase_str)
            if idx != -1:
                match_start = idx
                match_end = idx + len(phrase_str)

        if match_start == -1:
            for term in query_terms:
                idx = norm_content.find(term)
                if idx != -1 and (match_start == -1 or idx < match_start):
                    match_start = idx
                    match_end = idx + len(term)

        if match_start == -1:
            snippet = full_content[:snippet_length].strip()
            if len(full_content) > snippet_length:
                snippet += "..."
            return snippet

        half_window = snippet_length // 2
        start_char = max(0, match_start - half_window)
        end_char = min(len(full_content), match_end + half_window)

        if start_char > 0:
            space_idx = full_content.find(" ", start_char)
            if space_idx != -1 and space_idx < match_start:
                start_char = space_idx + 1

        if end_char < len(full_content):
            space_idx = full_content.rfind(" ", match_end, end_char)
            if space_idx != -1:
                end_char = space_idx

        snippet = full_content[start_char:end_char].strip()
        prefix = "... " if start_char > 0 else ""
        suffix = " ..." if end_char < len(full_content) else ""
        return f"{prefix}{snippet}{suffix}"

    # -------------------------------------------------------------------------
    # Result Presentation and Formatting
    # -------------------------------------------------------------------------

    @staticmethod
    def format_output(
        result: SearchResult,
        max_display: int = 10,
    ) -> str:
        """Formats the search result into the standardized academic presentation format."""
        lines = []
        lines.append("=" * 60)
        lines.append("TAMIL INFORMATION RETRIEVAL SYSTEM")
        lines.append("=" * 60)
        lines.append(f"Query: {result.query}")
        lines.append(f"Search Type: {result.search_type}")
        lines.append(f"Matching Documents: {result.total_matches}")

        if result.error_message:
            lines.append(f"Message: {result.error_message}")

        if result.total_matches > 0:
            display_count = min(len(result.items), max_display)
            for i in range(display_count):
                item = result.items[i]
                lines.append("-" * 60)
                lines.append(f"Result #{i + 1}")
                lines.append(f"Document ID: {item.doc_id}")
                lines.append(f"Title: {item.title}")
                if item.matched_positions is not None:
                    lines.append(f"Matched Positions: {item.matched_positions}")
                lines.append(f"Snippet: {item.snippet}")
                if item.city or item.published:
                    lines.append(f"Metadata: City={item.city} | Published={item.published}")

            if result.total_matches > display_count:
                remaining = result.total_matches - display_count
                lines.append("-" * 60)
                lines.append(f"(Showing {display_count} of {result.total_matches} total matches. {remaining} more matches omitted.)")

        lines.append(f"Search Time: {result.search_time_ms:.2f} ms")
        lines.append("=" * 60)
        return "\n".join(lines)


def main():
    """CLI driver for standalone search queries."""
    parser = argparse.ArgumentParser(description="Search Tamil News Information Retrieval System")
    parser.add_argument("--query", "-q", type=str, required=True, help="Tamil or alphanumeric search query")
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        choices=["term", "and", "or", "phrase"],
        default="term",
        help="Search mode: term (single), and (Boolean AND), or (Boolean OR), phrase (exact phrase)",
    )
    parser.add_argument("--index-dir", type=str, default="indexes", help="Path to indexes directory")
    parser.add_argument("--limit", type=int, default=10, help="Max results to display")

    args = parser.parse_args()

    engine = TamilSearchEngine(index_dir=args.index_dir)

    if args.mode == "term":
        res = engine.single_term_search(args.query)
    elif args.mode == "and":
        res = engine.boolean_and_search(args.query)
    elif args.mode == "or":
        res = engine.boolean_or_search(args.query)
    elif args.mode == "phrase":
        res = engine.exact_phrase_search(args.query)
    else:
        print(f"Unknown mode: {args.mode}")
        return

    print(engine.format_output(res, max_display=args.limit))


if __name__ == "__main__":
    main()
