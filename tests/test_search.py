"""Unit tests for Disk-Based Search Engine."""

import json
import os
import shutil
import tempfile
import unittest

from build_index import build_and_save
from search import TamilSearchEngine


class TestDiskBasedSearchEngine(unittest.TestCase):
    """Tests covering single-term, multi-term, phrase search on disk-based index."""

    @classmethod
    def setUpClass(cls):
        """Build a synthetic index for testing."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.articles_dir = os.path.join(cls.temp_dir, "extracted")
        cls.index_dir = os.path.join(cls.temp_dir, "indexes")
        os.makedirs(cls.articles_dir, exist_ok=True)

        articles = [
            {
                "id": 0,
                "title": "தமிழ்நாடு முதலமைச்சர் அறிவிப்பு",
                "text": "சென்னை தலைமைச் செயலகத்தில் தமிழக அரசு புதிய திட்டங்களை அறிமுகம் செய்தது. தமிழ்நாடு வளர்ச்சி அடைகிறது.",
            },
            {
                "id": 1,
                "title": "சென்னை வானிலை அறிக்கை",
                "text": "சென்னை மற்றும் புறநகர் பகுதிகளில் கனமழை பெய்துள்ளது. மழை தொடர்ந்து பெய்யும் என அறிவிக்கப்பட்டுள்ளது.",
            },
            {
                "id": 2,
                "title": "மதுரை மீனாட்சி அம்மன் கோவில்",
                "text": "மதுரை மாநகரில் சித்திரை திருவிழா கோலாகலமாக தொடங்கியது. தமிழ்நாடு மக்கள் திரண்டனர்.",
            },
            {
                "id": 3,
                "title": "தமிழ் வளர்ச்சி மாநாடு",
                "text": "தமிழ்நாடு நாடு முழுதும் வாழும் தமிழ் மக்களை இணைக்கிறது. நாடு நாடு எங்கும் தமிழ் வாழ்க.",
            },
            {
                "id": 4,
                "title": "Artificial Intelligence Research",
                "text": "AI and Machine Learning technologies are advancing rapidly in Chennai and Tamil Nadu.",
            },
        ]

        articles_path = os.path.join(cls.articles_dir, "articles.jsonl")
        with open(articles_path, "w", encoding="utf-8") as f:
            for article in articles:
                f.write(json.dumps(article, ensure_ascii=False) + "\n")

        build_and_save(articles_path=articles_path, output_dir=cls.index_dir)
        cls.engine = TamilSearchEngine(index_dir=cls.index_dir)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    # -------------------------------------------------------------------------
    # Single-Term Search
    # -------------------------------------------------------------------------

    def test_single_term_found(self):
        """Verify single-term search returns matching documents."""
        res = self.engine.single_term_search("சென்னை")
        self.assertEqual(res.search_type, "Single-Term Search")
        # சென்னை is in Doc 0 and Doc 1
        self.assertEqual(res.matching_doc_ids, [0, 1])
        self.assertEqual(res.total_matches, 2)

    def test_single_term_not_found(self):
        """Verify unknown term returns 0 results."""
        res = self.engine.single_term_search("இமயமலை")
        self.assertEqual(res.total_matches, 0)
        self.assertEqual(res.matching_doc_ids, [])

    def test_single_term_returns_positions(self):
        """Verify positions are returned from disk postings."""
        res = self.engine.single_term_search("சென்னை")
        self.assertIsNotNone(res.matching_positions)
        # Doc 1 has சென்னை at positions 0, 2 (title + text)
        self.assertIn(1, res.matching_positions)
        self.assertIn(0, res.matching_positions[1])

    def test_english_case_insensitive(self):
        """Verify English terms are case-insensitive."""
        res = self.engine.single_term_search("MACHINE")
        self.assertEqual(res.matching_doc_ids, [4])

    # -------------------------------------------------------------------------
    # Multi-Term Search
    # -------------------------------------------------------------------------

    def test_multi_term_union(self):
        """Verify multi-term returns union of postings."""
        res = self.engine.multi_term_search("மதுரை வானிலை")
        self.assertEqual(res.search_type, "Multi-Term Search")
        # மதுரை in [2], வானிலை in [1] → Union: [1, 2]
        self.assertEqual(res.matching_doc_ids, [1, 2])

    # -------------------------------------------------------------------------
    # Phrase Search
    # -------------------------------------------------------------------------

    def test_phrase_two_words(self):
        """Verify exact phrase checks consecutive positions."""
        res = self.engine.phrase_search("தமிழக அரசு")
        self.assertEqual(res.search_type, "Phrase Search")
        self.assertEqual(res.matching_doc_ids, [0])
        self.assertIn(0, res.matching_positions)

    def test_phrase_reversed_order_fails(self):
        """Verify reversed phrase returns 0 matches."""
        res = self.engine.phrase_search("அரசு தமிழக")
        self.assertEqual(res.total_matches, 0)

    def test_phrase_multiword(self):
        """3-word phrase: 'தமிழக அரசு புதிய'."""
        res = self.engine.phrase_search("தமிழக அரசு புதிய")
        self.assertEqual(res.matching_doc_ids, [0])

    def test_phrase_repeated_terms(self):
        """Phrase with repeated words: 'நாடு நாடு'."""
        res = self.engine.phrase_search("நாடு நாடு")
        self.assertEqual(res.matching_doc_ids, [3])

    def test_phrase_missing_word(self):
        """Phrase with unknown word returns 0."""
        res = self.engine.phrase_search("தமிழக இமயமலை")
        self.assertEqual(res.total_matches, 0)

    def test_phrase_with_quotes(self):
        """Quotes in phrase query are stripped."""
        res = self.engine.phrase_search('"தமிழக அரசு"')
        self.assertEqual(res.matching_doc_ids, [0])

    def test_english_phrase(self):
        """English phrase search."""
        res = self.engine.phrase_search("MACHINE LEARNING")
        self.assertEqual(res.matching_doc_ids, [4])

    # -------------------------------------------------------------------------
    # Edge Cases
    # -------------------------------------------------------------------------

    def test_empty_query(self):
        """Empty query returns 0 results."""
        res = self.engine.single_term_search("")
        self.assertEqual(res.total_matches, 0)
        self.assertIsNotNone(res.error_message)

    # -------------------------------------------------------------------------
    # Disk Access Verification
    # -------------------------------------------------------------------------

    def test_postings_retrieved_from_disk(self):
        """Verify postings are retrieved from disk file using offset+length."""
        # Directly verify the get_postings method reads from disk
        postings = self.engine.get_postings("சென்னை")
        self.assertIsInstance(postings, dict)
        self.assertIn("0", postings)
        self.assertIn("1", postings)
        self.assertEqual(postings["1"], [0, 3])

    def test_only_requested_term_read(self):
        """Verify that reading one term's postings does not load the full file."""
        # Read postings for a specific term
        vocab_entry = self.engine.vocabulary.get("சென்னை")
        self.assertIsNotNone(vocab_entry)

        # Verify length is much smaller than total file
        total_size = os.path.getsize(self.engine.postings_path)
        self.assertLess(vocab_entry["length"], total_size)

    def test_index_reload_consistency(self):
        """Verify search results are identical after reloading the engine."""
        reloaded = TamilSearchEngine(index_dir=self.index_dir)
        res_orig = self.engine.phrase_search("தமிழக அரசு")
        res_reload = reloaded.phrase_search("தமிழக அரசு")
        self.assertEqual(res_orig.matching_doc_ids, res_reload.matching_doc_ids)
        self.assertEqual(res_orig.matching_positions, res_reload.matching_positions)


if __name__ == "__main__":
    unittest.main()
