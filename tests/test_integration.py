"""Integration tests against the actual Tamil News Dataset indexes."""

import os
import unittest
from search import TamilSearchEngine


class TestTamilIRIntegration(unittest.TestCase):
    """Integration test suite executing real queries on the indexed Hindu Tamil corpus."""

    @classmethod
    def setUpClass(cls):
        """Ensure indexes exist in 'indexes/' and load the search engine."""
        cls.index_dir = "indexes"
        if not os.path.exists(os.path.join(cls.index_dir, "inverted_index.json")):
            raise unittest.SkipTest("Index files not yet built in 'indexes/'. Skipping integration tests.")
        cls.engine = TamilSearchEngine(index_dir=cls.index_dir)

    def test_corpus_scale(self):
        """Verify the 10,334 documents and vocabulary size from metadata."""
        meta = self.engine.metadata
        self.assertEqual(meta["total_indexed_documents"], 10334)
        self.assertEqual(meta["total_unique_terms"], 168277)
        self.assertEqual(meta["total_indexed_tokens"], 2659417)

    def test_real_term_search(self):
        """Verify term search for prominent Tamil city 'சென்னை'."""
        res = self.engine.single_term_search("சென்னை")
        self.assertGreater(res.total_matches, 6000)
        self.assertIn(0, res.matching_doc_ids)

    def test_real_boolean_and_search(self):
        """Verify Boolean AND search for 'சென்னை AND மழை'."""
        res = self.engine.boolean_and_search("சென்னை AND மழை")
        self.assertGreater(res.total_matches, 500)
        # Ensure all matching documents contain both terms in regular inverted index
        inv = self.engine.inverted_index
        chennai_set = set(inv.get("சென்னை", []))
        mazhai_set = set(inv.get("மழை", []))
        for did in res.matching_doc_ids:
            self.assertIn(did, chennai_set)
            self.assertIn(did, mazhai_set)

    def test_real_boolean_or_search(self):
        """Verify Boolean OR search for 'தமிழ்நாடு OR சென்னை'."""
        res = self.engine.boolean_or_search("தமிழ்நாடு OR சென்னை")
        self.assertGreater(res.total_matches, 6500)

    def test_real_exact_phrase_search(self):
        """Verify Exact Phrase search for 'தமிழக அரசு'."""
        res = self.engine.exact_phrase_search("தமிழக அரசு")
        self.assertGreater(res.total_matches, 1500)
        self.assertIn(0, res.matching_doc_ids)
        # Position in doc 0 is known to be [53]
        self.assertIn(0, res.matching_positions)
        self.assertIn(53, res.matching_positions[0])

    def test_phrase_search_strictly_more_specific_than_and(self):
        """Demonstrate that exact phrase matching is strictly a subset of Boolean AND."""
        and_res = self.engine.boolean_and_search("தமிழக AND அரசு")
        phrase_res = self.engine.exact_phrase_search("தமிழக அரசு")
        self.assertLess(phrase_res.total_matches, and_res.total_matches)
        # Every document matching the exact phrase must also match the Boolean AND
        and_docs_set = set(and_res.matching_doc_ids)
        for did in phrase_res.matching_doc_ids:
            self.assertIn(did, and_docs_set)


if __name__ == "__main__":
    unittest.main()
