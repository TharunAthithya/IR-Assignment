"""Integration tests against the actual Tamil Wikipedia index.

These tests run against the full built index in 'indexes/' and verify
real-world corpus statistics and search correctness. They are skipped
if the index has not been built yet.
"""

import os
import unittest

from search import TamilSearchEngine


class TestWikipediaIntegration(unittest.TestCase):
    """Integration tests on the actual Tamil Wikipedia positional index."""

    @classmethod
    def setUpClass(cls):
        cls.index_dir = "indexes"
        vocab_path = os.path.join(cls.index_dir, "vocabulary.json")
        if not os.path.exists(vocab_path):
            raise unittest.SkipTest(
                "Index files not built in 'indexes/'. "
                "Run 'python main.py build' first. Skipping integration tests."
            )
        cls.engine = TamilSearchEngine(index_dir=cls.index_dir)

    def test_corpus_scale(self):
        """Verify the corpus contains a substantial number of documents."""
        meta = self.engine.metadata
        total_docs = meta.get("total_documents", 0)
        vocab_size = meta.get("vocabulary_size", 0)
        total_tokens = meta.get("total_tokens", 0)

        self.assertGreaterEqual(total_docs, 30, "Should have at least 30 documents")
        self.assertGreater(vocab_size, 5000, "Should have substantial vocabulary")
        self.assertGreater(total_tokens, 10000, "Should have substantial tokens")

    def test_vocabulary_in_ram(self):
        """Verify vocabulary is loaded into RAM."""
        self.assertGreater(len(self.engine.vocabulary), 0)
        # Check a sample entry has the required fields
        sample_term = next(iter(self.engine.vocabulary))
        entry = self.engine.vocabulary[sample_term]
        self.assertIn("df", entry)
        self.assertIn("cf", entry)
        self.assertIn("offset", entry)
        self.assertIn("length", entry)

    def test_postings_file_exists(self):
        """Verify postings.dat exists on disk."""
        self.assertTrue(
            os.path.exists(self.engine.postings_path),
            "postings.dat must exist"
        )
        self.assertGreater(
            os.path.getsize(self.engine.postings_path), 0,
            "postings.dat must not be empty"
        )

    def test_postings_from_disk(self):
        """Verify postings are retrieved from disk (not RAM)."""
        # Ensure positional_index is NOT an attribute (old architecture)
        self.assertFalse(
            hasattr(self.engine, "positional_index"),
            "Engine should NOT have a positional_index in RAM"
        )

        # Retrieve a term's postings from disk
        sample_term = next(
            t for t, v in self.engine.vocabulary.items()
            if v["df"] >= 5
        )
        postings = self.engine.get_postings(sample_term)
        self.assertIsInstance(postings, dict)
        self.assertEqual(len(postings), self.engine.vocabulary[sample_term]["df"])

    def test_single_term_search_live(self):
        """Verify single-term search on a common Tamil term."""
        # Find a common term
        common = max(self.engine.vocabulary.items(), key=lambda x: x[1]["df"])
        term = common[0]
        expected_df = common[1]["df"]

        res = self.engine.single_term_search(term)
        self.assertEqual(res.total_matches, expected_df)
        self.assertGreater(res.total_matches, 0)

    def test_phrase_search_live(self):
        """Verify phrase search returns fewer results than individual term search."""
        # Pick two common terms and try as phrase
        vocab = self.engine.vocabulary
        top_terms = sorted(vocab.items(), key=lambda x: x[1]["df"], reverse=True)

        # Find a phrase that matches
        for i in range(min(10, len(top_terms))):
            for j in range(i+1, min(10, len(top_terms))):
                t1 = top_terms[i][0]
                t2 = top_terms[j][0]
                phrase_res = self.engine.phrase_search(f"{t1} {t2}")
                if phrase_res.total_matches > 0:
                    # Phrase results should be <= multi-term results
                    multi_res = self.engine.multi_term_search(f"{t1} {t2}")
                    self.assertLessEqual(
                        phrase_res.total_matches,
                        multi_res.total_matches,
                    )
                    return

        # If no phrase found, that's OK for this test
        self.skipTest("No matching phrase found in top terms")

    def test_unknown_term_handling(self):
        """Verify unknown term returns empty result."""
        res = self.engine.single_term_search("இமயமலைரோபோ2099செவ்வாய்")
        self.assertEqual(res.total_matches, 0)

    def test_df_cf_consistency(self):
        """Verify df and cf values are consistent with actual postings."""
        # Check a few terms
        count = 0
        for term, entry in self.engine.vocabulary.items():
            if count >= 5:
                break
            postings = self.engine.get_postings(term)
            actual_df = len(postings)
            actual_cf = sum(len(positions) for positions in postings.values())
            self.assertEqual(entry["df"], actual_df, f"df mismatch for term '{term}'")
            self.assertEqual(entry["cf"], actual_cf, f"cf mismatch for term '{term}'")
            count += 1


if __name__ == "__main__":
    unittest.main()
