"""Unit tests for Unicode-aware Tamil text preprocessing and tokenization."""

import unittest
import unicodedata
from preprocess import normalize_text, tokenize, tokenize_with_spans


class TestTamilPreprocessing(unittest.TestCase):
    """Test suite verifying Unicode handling and Tamil tokenization correctness."""

    def test_unicode_nfc_normalization(self):
        """Verify NFC normalizes decomposed characters to canonical composed form."""
        # Decomposed form: base க + combining vowel sign ா
        decomposed = "\u0B95\u0BBE"
        composed = unicodedata.normalize("NFC", decomposed)
        normalized = normalize_text(decomposed)
        self.assertEqual(normalized, composed)

    def test_tamil_vowel_signs_not_split(self):
        """Verify combining vowel signs (matras) remain attached to consonants."""
        text = "தமிழ்நாடு"
        tokens = tokenize(text)
        self.assertEqual(tokens, ["தமிழ்நாடு"])

        # Words with complex combining marks
        test_words = [
            "கோவில்",      # Kombu + Ka + Kaal (ொ)
            "சென்னை",     # Double kombu + Na (ை)
            "கீழடி",       # Ka + ii matra (ீ), La, Da + i matra (ி)
            "புதுச்சேரி",   # Pa + u, Tha + u, Cha + pulli, Cha + e, Ra + ii
            "ஔவையார்",     # Independent Au (ஔ), Va + i, Ya + aa, Ra + pulli
        ]
        for word in test_words:
            tokens = tokenize(word)
            self.assertEqual(len(tokens), 1, f"Word '{word}' should not be split into multiple tokens.")
            self.assertEqual(tokens[0], word)

    def test_tamil_pulli_virama_preservation(self):
        """Verify pulli/virama marks (்) are preserved with consonants."""
        text = "அரசுப் பேருந்து ஓட்டுநர்"
        tokens = tokenize(text)
        self.assertEqual(tokens, ["அரசுப்", "பேருந்து", "ஓட்டுநர்"])

    def test_grantha_and_aytham_support(self):
        """Verify Grantha consonants (ஜ, ஷ, ஸ, ஹ, க்ஷ) and Aytham (ஃ) are tokenized."""
        text = "எஃகு ஜப்பான் ஷாப்பிங் ஹிந்தி லக்ஷ்மி"
        tokens = tokenize(text)
        expected = ["எஃகு", "ஜப்பான்", "ஷாப்பிங்", "ஹிந்தி", "லக்ஷ்மி"]
        self.assertEqual(tokens, expected)

    def test_english_and_case_folding(self):
        """Verify embedded English words are lowercased while Tamil remains intact."""
        text = "AI மற்றும் Machine Learning 2024"
        tokens = tokenize(text)
        expected = ["ai", "மற்றும்", "machine", "learning", "2024"]
        self.assertEqual(tokens, expected)

    def test_numbers_and_punctuation(self):
        """Verify punctuation acts as word boundary and numbers are preserved."""
        text = "ரூ.1.10 கோடி (2024-2025): 'அதிமுக' & 'திமுக'!"
        tokens = tokenize(text)
        expected = ["ரூ", "1", "10", "கோடி", "2024", "2025", "அதிமுக", "திமுக"]
        self.assertEqual(tokens, expected)

    def test_whitespace_and_zero_width_spaces(self):
        """Verify non-breaking space and zero-width spaces are cleaned."""
        text = "சென்னை\u00a0மாநகர்\u200bபோக்குவரத்து"
        tokens = tokenize(text)
        self.assertEqual(tokens, ["சென்னை", "மாநகர்", "போக்குவரத்து"])

    def test_empty_and_whitespace_only(self):
        """Verify empty and whitespace strings return empty token list."""
        self.assertEqual(tokenize(""), [])
        self.assertEqual(tokenize("   \t  \n  "), [])
        self.assertEqual(tokenize(None), [])

    def test_token_spans(self):
        """Verify token spans return accurate character boundary positions."""
        text = "தமிழ்நாடு அரசு"
        spans = tokenize_with_spans(text)
        self.assertEqual(len(spans), 2)
        self.assertEqual(spans[0], ("தமிழ்நாடு", 0, 9))
        self.assertEqual(spans[1], ("அரசு", 10, 14))


if __name__ == "__main__":
    unittest.main()
