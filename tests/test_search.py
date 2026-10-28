"""Unit tests for Search Engine implementation."""

import json
import os
import shutil
import tempfile
import unittest

from build_index import build_indexes, save_indexes
from search import TamilSearchEngine


class TestSearchEngine(unittest.TestCase):
    """Test suite covering single-term, Boolean AND, Boolean OR, and exact phrase search."""

    @classmethod
    def setUpClass(cls):
        """Build a deterministic synthetic collection for testing."""
        cls.synthetic_records = [
            {
                "Title": "தமிழ்நாடு முதலமைச்சர் அறிவிப்பு",
                "Text": "சென்னை தலைமைச் செயலகத்தில் தமிழக அரசு புதிய திட்டங்களை அறிமுகம் செய்தது. தமிழ்நாடு வளர்ச்சி அடைகிறது.",
                "Author": "செய்திப்பிரிவு",
                "City": "சென்னை",
                "Published": "2024-01-01",
            },
            {
                "Title": "சென்னை வானிலை அறிக்கை",
                "Text": "சென்னை மற்றும் புறநகர் பகுதிகளில் கனமழை பெய்துள்ளது. மழை தொடர்ந்து பெய்யும் என அறிவிக்கப்பட்டுள்ளது.",
                "Author": "வானிலை மையம்",
                "City": "சென்னை",
                "Published": "2024-01-02",
            },
            {
                "Title": "மதுரை மீனாட்சி அம்மன் கோவில்",
                "Text": "மதுரை மாநகரில் சித்திரை திருவிழா கோலாகலமாக தொடங்கியது. தமிழ்நாடு மக்கள் திரண்டனர்.",
                "Author": "செய்தியாளர்",
                "City": "மதுரை",
                "Published": "2024-01-03",
            },
            {
                "Title": "தமிழ் வளர்ச்சி மாநாடு",
                "Text": "தமிழ்நாடு நாடு முழுதும் வாழும் தமிழ் மக்களை இணைக்கிறது. நாடு நாடு எங்கும் தமிழ் வாழ்க.",
                "Author": "ஆசிரியர்",
                "City": "கோவை",
                "Published": "2024-01-04",
            },
            {
                "Title": "Artificial Intelligence Research",
                "Text": "AI and Machine Learning technologies are advancing rapidly in Chennai and Tamil Nadu.",
                "Author": "Tech Desk",
                "City": "சென்னை",
                "Published": "2024-01-05",
            },
        ]
        cls.temp_dir = tempfile.mkdtemp()
        pos, inv, dic, docs, stats = build_indexes(cls.synthetic_records)
        save_indexes(cls.temp_dir, pos, inv, dic, docs, stats)
        cls.engine = TamilSearchEngine(index_dir=cls.temp_dir)

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary test index directory."""
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    # -------------------------------------------------------------------------
    # Single-Term Search Tests
    # -------------------------------------------------------------------------

    def test_single_term_search_found(self):
        """Verify single term search returns all documents containing the term."""
        res_tamil = self.engine.single_term_search("சென்னை")
        self.assertEqual(res_tamil.search_type, "Single-Term Search")
        # Tamil "சென்னை" is in Doc 0 and Doc 1
        self.assertEqual(res_tamil.matching_doc_ids, [0, 1])
        self.assertEqual(res_tamil.total_matches, 2)

        # English "chennai" is in Doc 4
        res_eng = self.engine.single_term_search("chennai")
        self.assertEqual(res_eng.matching_doc_ids, [4])
        self.assertEqual(res_eng.total_matches, 1)

    def test_single_term_search_not_found(self):
        """Verify non-existent term returns 0 results gracefully."""
        res = self.engine.single_term_search("இமயமலை")
        self.assertEqual(res.total_matches, 0)
        self.assertEqual(res.matching_doc_ids, [])

    # -------------------------------------------------------------------------
    # Boolean AND Search Tests
    # -------------------------------------------------------------------------

    def test_boolean_and_two_terms(self):
        """Verify Boolean AND returns intersection of postings lists."""
        # Doc 1 contains both "சென்னை" and "மழை"
        res = self.engine.boolean_and_search("சென்னை AND மழை")
        self.assertEqual(res.search_type, "Boolean AND Search")
        self.assertEqual(res.matching_doc_ids, [1])

    def test_boolean_and_three_terms(self):
        """Verify Boolean AND supports 3 or more terms."""
        # "தமிழ்நாடு AND சென்னை AND அரசு" -> Doc 0
        res = self.engine.boolean_and_search("தமிழ்நாடு AND சென்னை AND அரசு")
        self.assertEqual(res.matching_doc_ids, [0])

    def test_boolean_and_no_match(self):
        """Verify Boolean AND returns 0 if any term is missing from intersection."""
        res = self.engine.boolean_and_search("சென்னை AND இமயமலை")
        self.assertEqual(res.total_matches, 0)

    # -------------------------------------------------------------------------
    # Boolean OR Search Tests
    # -------------------------------------------------------------------------

    def test_boolean_or_two_terms(self):
        """Verify Boolean OR returns union of postings lists."""
        # "மதுரை" in [2], "வானிலை" in [1] -> Union: [1, 2]
        res = self.engine.boolean_or_search("மதுரை OR வானிலை")
        self.assertEqual(res.search_type, "Boolean OR Search")
        self.assertEqual(res.matching_doc_ids, [1, 2])

    def test_boolean_or_three_terms(self):
        """Verify Boolean OR supports 3 or more terms."""
        # "மதுரை" [2], "வானிலை" [1], "திட்டங்களை" [0] -> Union: [0, 1, 2]
        res = self.engine.boolean_or_search("மதுரை OR வானிலை OR திட்டங்களை")
        self.assertEqual(res.matching_doc_ids, [0, 1, 2])

    # -------------------------------------------------------------------------
    # Exact Phrase Search Tests
    # -------------------------------------------------------------------------

    def test_exact_phrase_two_words(self):
        """Verify exact phrase search checks consecutive word positions."""
        # "தமிழக அரசு" appears in Doc 0 text consecutively: "... தமிழக அரசு புதிய ..."
        res = self.engine.exact_phrase_search("தமிழக அரசு")
        self.assertEqual(res.search_type, "Exact Phrase Search")
        self.assertEqual(res.matching_doc_ids, [0])
        self.assertIn(0, res.matching_positions)
        self.assertTrue(len(res.matching_positions[0]) > 0)

    def test_exact_phrase_multiword(self):
        """Verify exact phrase search supports 3 or more consecutive terms."""
        # In Doc 0: "தமிழக அரசு புதிய"
        res = self.engine.exact_phrase_search("தமிழக அரசு புதிய")
        self.assertEqual(res.matching_doc_ids, [0])

    def test_exact_phrase_reversed_order_fails(self):
        """Verify exact phrase search returns 0 matches when term order is inverted."""
        # "அரசு தமிழக" does NOT occur consecutively in Doc 0
        res = self.engine.exact_phrase_search("அரசு தமிழக")
        self.assertEqual(res.total_matches, 0)
        self.assertEqual(res.matching_doc_ids, [])

    def test_exact_phrase_repeated_terms(self):
        """Verify exact phrase search with repeated words (e.g. 'நாடு நாடு')."""
        # In Doc 3: "நாடு நாடு எங்கும் தமிழ் வாழ்க"
        res = self.engine.exact_phrase_search("நாடு நாடு")
        self.assertEqual(res.matching_doc_ids, [3])

    def test_exact_phrase_missing_word(self):
        """Verify phrase containing non-existent word returns 0 matches."""
        res = self.engine.exact_phrase_search("தமிழக இமயமலை")
        self.assertEqual(res.total_matches, 0)

    # -------------------------------------------------------------------------
    # Edge Cases & Formatting Tests
    # -------------------------------------------------------------------------

    def test_empty_query(self):
        """Verify empty and whitespace queries are handled gracefully."""
        res = self.engine.single_term_search("")
        self.assertEqual(res.total_matches, 0)
        self.assertIsNotNone(res.error_message)

        res_and = self.engine.boolean_and_search("   ")
        self.assertEqual(res_and.total_matches, 0)

    def test_punctuation_in_query(self):
        """Verify queries with extra punctuation/quotes are handled cleanly."""
        res = self.engine.exact_phrase_search('"தமிழக அரசு"!')
        self.assertEqual(res.matching_doc_ids, [0])

    def test_english_case_insensitivity(self):
        """Verify uppercase English query matches lowercased index."""
        # In Doc 4: "AI and Machine Learning"
        res_upper = self.engine.single_term_search("MACHINE")
        self.assertEqual(res_upper.matching_doc_ids, [4])

        res_phrase = self.engine.exact_phrase_search("MACHINE LEARNING")
        self.assertEqual(res_phrase.matching_doc_ids, [4])

    def test_index_reload_consistency(self):
        """Verify search results are identical after reloading saved indexes from disk."""
        reloaded_engine = TamilSearchEngine(index_dir=self.temp_dir)
        res_original = self.engine.exact_phrase_search("தமிழக அரசு")
        res_reloaded = reloaded_engine.exact_phrase_search("தமிழக அரசு")
        self.assertEqual(res_original.matching_doc_ids, res_reloaded.matching_doc_ids)
        self.assertEqual(res_original.matching_positions, res_reloaded.matching_positions)


if __name__ == "__main__":
    unittest.main()
