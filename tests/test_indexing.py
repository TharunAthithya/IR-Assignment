"""Unit tests for Disk-Based Positional Index Construction."""

import json
import os
import shutil
import tempfile
import unittest

from build_index import build_and_save
from preprocess import tokenize


class TestDiskBasedIndexConstruction(unittest.TestCase):
    """Tests verifying vocabulary.json + postings.dat architecture."""

    @classmethod
    def setUpClass(cls):
        """Create a small synthetic articles.jsonl for testing."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.articles_dir = os.path.join(cls.temp_dir, "extracted")
        cls.index_dir = os.path.join(cls.temp_dir, "indexes")
        os.makedirs(cls.articles_dir, exist_ok=True)

        # Write synthetic articles
        articles = [
            {
                "id": 0,
                "title": "தமிழ்நாடு செய்திகள்",
                "text": "சென்னை மற்றும் மதுரை தமிழகத்தின் முக்கிய நகரங்கள் ஆகும்.",
            },
            {
                "id": 1,
                "title": "சென்னை வானிலை",
                "text": "சென்னை மாநகரில் கனமழை பெய்துள்ளது. சென்னை மக்கள் மகிழ்ச்சி அடைந்தனர்.",
            },
            {
                "id": 2,
                "title": "மதுரை திருவிழா",
                "text": "மதுரை மீனாட்சி அம்மன் கோவில் திருவிழா தொடங்கியது.",
            },
        ]
        cls.articles_path = os.path.join(cls.articles_dir, "articles.jsonl")
        with open(cls.articles_path, "w", encoding="utf-8") as f:
            for article in articles:
                f.write(json.dumps(article, ensure_ascii=False) + "\n")

        # Build index
        cls.stats = build_and_save(
            articles_path=cls.articles_path,
            output_dir=cls.index_dir,
        )

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def test_output_files_exist(self):
        """Verify all required index files are created."""
        for fname in ["vocabulary.json", "postings.dat", "doc_meta.json", "metadata.json"]:
            fpath = os.path.join(self.index_dir, fname)
            self.assertTrue(os.path.exists(fpath), f"Missing: {fname}")

    def test_vocabulary_structure(self):
        """Verify vocabulary entries contain df, cf, offset, length."""
        with open(os.path.join(self.index_dir, "vocabulary.json"), "r", encoding="utf-8") as f:
            vocab = json.load(f)

        self.assertIn("சென்னை", vocab)
        entry = vocab["சென்னை"]
        self.assertIn("df", entry)
        self.assertIn("cf", entry)
        self.assertIn("offset", entry)
        self.assertIn("length", entry)
        self.assertIsInstance(entry["offset"], int)
        self.assertIsInstance(entry["length"], int)
        self.assertGreater(entry["length"], 0)

    def test_vocabulary_df_cf_accuracy(self):
        """Verify df and cf are computed correctly."""
        with open(os.path.join(self.index_dir, "vocabulary.json"), "r", encoding="utf-8") as f:
            vocab = json.load(f)

        # சென்னை: Doc 0 (pos 2), Doc 1 (pos 0, 2, 6) → df=2, cf=4
        self.assertEqual(vocab["சென்னை"]["df"], 2)
        self.assertEqual(vocab["சென்னை"]["cf"], 4)

        # மதுரை: Doc 0 (pos 4), Doc 2 (pos 0, 2) → df=2, cf=3
        self.assertEqual(vocab["மதுரை"]["df"], 2)
        self.assertEqual(vocab["மதுரை"]["cf"], 3)

    def test_postings_retrieval_by_offset(self):
        """Verify postings can be retrieved using byte offset and length."""
        with open(os.path.join(self.index_dir, "vocabulary.json"), "r", encoding="utf-8") as f:
            vocab = json.load(f)

        entry = vocab["சென்னை"]
        offset = entry["offset"]
        length = entry["length"]

        with open(os.path.join(self.index_dir, "postings.dat"), "rb") as f:
            f.seek(offset)
            raw = f.read(length)

        postings = json.loads(raw.decode("utf-8"))

        # Verify postings structure
        self.assertIn("0", postings)  # Doc 0
        self.assertIn("1", postings)  # Doc 1
        self.assertEqual(postings["0"], [2])  # Position 2 in Doc 0
        self.assertEqual(postings["1"], [0, 2, 6])  # Positions in Doc 1

    def test_postings_offsets_are_byte_accurate(self):
        """Verify multiple terms' offsets don't overlap and are byte-accurate."""
        with open(os.path.join(self.index_dir, "vocabulary.json"), "r", encoding="utf-8") as f:
            vocab = json.load(f)

        postings_path = os.path.join(self.index_dir, "postings.dat")
        postings_size = os.path.getsize(postings_path)

        for term, entry in vocab.items():
            offset = entry["offset"]
            length = entry["length"]
            # Offset + length must not exceed file size
            self.assertLessEqual(
                offset + length, postings_size,
                f"Term '{term}' offset+length ({offset}+{length}) exceeds file size ({postings_size})"
            )
            # Verify each term's postings can be decoded
            with open(postings_path, "rb") as f:
                f.seek(offset)
                raw = f.read(length)
            postings = json.loads(raw.decode("utf-8"))
            self.assertIsInstance(postings, dict)

    def test_doc_meta_structure(self):
        """Verify document metadata maps doc_id to title."""
        with open(os.path.join(self.index_dir, "doc_meta.json"), "r", encoding="utf-8") as f:
            doc_meta = json.load(f)

        self.assertEqual(len(doc_meta), 3)
        self.assertEqual(doc_meta["0"], "தமிழ்நாடு செய்திகள்")
        self.assertEqual(doc_meta["1"], "சென்னை வானிலை")
        self.assertEqual(doc_meta["2"], "மதுரை திருவிழா")

    def test_metadata_statistics(self):
        """Verify metadata contains accurate build statistics."""
        with open(os.path.join(self.index_dir, "metadata.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.assertEqual(meta["total_documents"], 3)
        self.assertGreater(meta["vocabulary_size"], 0)
        self.assertGreater(meta["total_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
