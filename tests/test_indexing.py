"""Unit tests for Index Construction and Serialization."""

import json
import os
import shutil
import tempfile
import unittest

from build_index import build_indexes, save_indexes


class TestIndexConstruction(unittest.TestCase):
    """Test suite verifying regular and positional inverted index construction."""

    def setUp(self):
        """Create deterministic synthetic documents for unit testing."""
        self.synthetic_records = [
            {
                "Title": "தமிழ்நாடு செய்திகள்",
                "Text": "சென்னை மற்றும் மதுரை தமிழகத்தின் முக்கிய நகரங்கள் ஆகும்.",
                "Author": "செய்தியாளர்",
                "City": "சென்னை",
                "Published": "2024-01-01",
            },
            {
                "Title": "சென்னை வானிலை",
                "Text": "சென்னை மாநகரில் கனமழை பெய்துள்ளது. சென்னை மக்கள் மகிழ்ச்சி அடைந்தனர்.",
                "Author": "வானிலை பிரிவு",
                "City": "சென்னை",
                "Published": "2024-01-02",
            },
            {
                "Title": "மதுரை திருவிழா",
                "Text": "மதுரை மீனாட்சி அம்மன் கோவில் திருவிழா தொடங்கியது.",
                "Author": "செய்தியாளர்",
                "City": "மதுரை",
                "Published": "2024-01-03",
            },
        ]
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_positional_index_structure_and_repeated_terms(self):
        """Verify positional index maps terms to doc IDs and sorted zero-based positions."""
        pos_idx, inv_idx, dictionary, docs, stats = build_indexes(self.synthetic_records)

        # In Doc 1: "சென்னை வானிலை சென்னை மாநகரில் கனமழை பெய்துள்ளது. சென்னை மக்கள் மகிழ்ச்சி அடைந்தனர்."
        # Tokens:
        # 0: சென்னை
        # 1: வானிலை
        # 2: சென்னை
        # 3: மாநகரில்
        # 4: கனமழை
        # 5: பெய்துள்ளது
        # 6: சென்னை
        # 7: மக்கள்
        # 8: மகிழ்ச்சி
        # 9: அடைந்தனர்
        self.assertIn("சென்னை", pos_idx)
        self.assertIn("1", pos_idx["சென்னை"])
        expected_positions_doc1 = [0, 2, 6]
        self.assertEqual(pos_idx["சென்னை"]["1"], expected_positions_doc1)

        # In Doc 0: "தமிழ்நாடு செய்திகள் சென்னை மற்றும் மதுரை தமிழகத்தின் முக்கிய நகரங்கள் ஆகும்"
        # 0: தமிழ்நாடு, 1: செய்திகள், 2: சென்னை
        self.assertIn("0", pos_idx["சென்னை"])
        self.assertEqual(pos_idx["சென்னை"]["0"], [2])

    def test_regular_inverted_index_derivation(self):
        """Verify regular inverted index contains sorted, unique document IDs."""
        pos_idx, inv_idx, dictionary, docs, stats = build_indexes(self.synthetic_records)

        # "சென்னை" appears in Doc 0 and Doc 1
        self.assertIn("சென்னை", inv_idx)
        self.assertEqual(inv_idx["சென்னை"], [0, 1])

        # "மதுரை" appears in Doc 0 and Doc 2
        self.assertIn("மதுரை", inv_idx)
        self.assertEqual(inv_idx["மதுரை"], [0, 2])

        # "வானிலை" appears only in Doc 1
        self.assertIn("வானிலை", inv_idx)
        self.assertEqual(inv_idx["வானிலை"], [1])

    def test_dictionary_df_and_cf(self):
        """Verify document frequency (df) and collection frequency (cf)."""
        pos_idx, inv_idx, dictionary, docs, stats = build_indexes(self.synthetic_records)

        # "சென்னை": appears in 2 docs (df=2), total occurrences = 1 (in doc 0) + 3 (in doc 1) = 4 (cf=4)
        self.assertIn("சென்னை", dictionary)
        self.assertEqual(dictionary["சென்னை"]["df"], 2)
        self.assertEqual(dictionary["சென்னை"]["cf"], 4)

        # "மதுரை": appears in 2 docs (df=2), 1 in doc 0 + 2 in doc 2 (Title and Text) = 3 (cf=3)
        self.assertEqual(dictionary["மதுரை"]["df"], 2)
        self.assertEqual(dictionary["மதுரை"]["cf"], 3)

    def test_index_serialization_and_deserialization(self):
        """Verify saving to JSON and reloading preserves identical data and UTF-8 encoding."""
        pos_idx, inv_idx, dictionary, docs, stats = build_indexes(self.synthetic_records)
        saved_paths = save_indexes(self.temp_dir, pos_idx, inv_idx, dictionary, docs, stats)

        # Verify files exist
        for fname in ["inverted_index.json", "positional_index.json", "dictionary.json", "documents.json", "metadata.json"]:
            self.assertTrue(os.path.exists(os.path.join(self.temp_dir, fname)))

        # Reload inverted index
        with open(saved_paths["inverted_index.json"], "r", encoding="utf-8") as f:
            reloaded_inv = json.load(f)
        self.assertEqual(reloaded_inv, inv_idx)

        # Reload positional index
        with open(saved_paths["positional_index.json"], "r", encoding="utf-8") as f:
            reloaded_pos = json.load(f)
        self.assertEqual(reloaded_pos, pos_idx)


if __name__ == "__main__":
    unittest.main()
