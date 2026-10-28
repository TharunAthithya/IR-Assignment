"""Tamil and Multilingual Text Preprocessing and Tokenization Module.

This module provides Unicode-aware text normalization and tokenization specifically
designed for the Tamil language, while seamlessly supporting embedded English terms
and numbers commonly present in news articles.

Key linguistic and technical features:
1. Unicode NFC Normalization:
   Composes decomposed Unicode characters into Canonical Composition form (NFC).
   In Tamil, this ensures that combining characters and vowel signs are uniformly
   represented across all documents and search queries.

2. Tamil Orthography and Combining Character Preservation:
   Tamil script uses base consonants (e.g., க, ச, த), independent vowels (e.g., அ, ஆ, இ),
   aytham (ஃ), and combining marks consisting of vowel signs (matras, e.g., ா, ி, ீ, ு, ூ, ெ, ே, ை, ொ, ோ, ௌ)
   and the pulli/virama (்) which mutes the inherent vowel.
   This module uses the Unicode-aware `regex` engine with category properties
   `[\\p{L}\\p{N}][\\p{L}\\p{M}\\p{N}]*`, ensuring that combining marks (\\p{M})
   are never split or detached from their preceding base letters (\\p{L}).

3. Multilingual and Alphanumeric Support:
   Embedded English words are lowercased (case-folded) for case-insensitive matching.
   Tamil words (which do not have letter case) remain unaffected.
   Digits (both ASCII and Tamil) are preserved.

4. Punctuation and Whitespace Handling:
   Unusual whitespaces (non-breaking spaces, zero-width spaces, tabs, newlines) are
   normalized. Punctuation symbols (Tamil and Western) act as word boundaries and are stripped.

5. Academic IR Constraints:
   - Preserves token sequence order and zero-based positional offsets.
   - Preserves Tamil stop words by default (essential for phrase searching).
   - Avoids English-specific stemming on Tamil vocabulary.
"""

import sys
import unicodedata
from typing import List, Tuple
import regex as re

# Reconfigure stdout/stdin to UTF-8 if running directly or in interactive environments
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Regex pattern matching any valid word in Tamil, English, or alphanumeric sequences.
# \p{L}: Any Unicode letter (Tamil consonants, vowels, aytham, English, Grantha letters)
# \p{N}: Any Unicode number (0-9, Tamil numerals)
# \p{M}: Any Unicode combining mark (Tamil vowel signs/matras, pulli/virama)
# Token must begin with a letter or digit, followed by letters, marks, or digits.
TOKEN_PATTERN = re.compile(r"[\p{L}\p{N}][\p{L}\p{M}\p{N}]*", re.UNICODE)

# Pattern for normalizing whitespace (including non-breaking space \u00a0, zero-width space \u200b)
WHITESPACE_PATTERN = re.compile(r"[\s\u00a0\u200b\u200c\u200d]+", re.UNICODE)


def normalize_text(text: str) -> str:
    """Applies Unicode NFC normalization, whitespace normalization, and lowercase folding.

    Args:
        text: Raw input string (Tamil, English, or mixed).

    Returns:
        Normalized string in Unicode NFC form with single whitespace separation.
    """
    if not text:
        return ""

    # 1. Unicode NFC Normalization
    text = unicodedata.normalize("NFC", text)

    # 2. Case-folding (affects English letters, leaves Tamil characters intact)
    text = text.lower()

    # 3. Normalize all whitespace sequences to a single standard space
    text = WHITESPACE_PATTERN.sub(" ", text).strip()

    return text


def tokenize(text: str) -> List[str]:
    """Tokenizes normalized Tamil, English, and alphanumeric text into an ordered list of terms.

    Preserves:
    - Tamil combining vowel signs (matras) and pulli (virama).
    - Grantha consonants (ஜ, ஷ, ஸ, ஹ, க்ஷ).
    - Tamil Aytham (ஃ).
    - English words (lowercased).
    - Numbers and digits.
    - Zero-based token sequence and relative order.

    Args:
        text: Input string to tokenize.

    Returns:
        List of normalized term tokens in exact document order.
    """
    if not text:
        return []

    # Pre-normalize text (NFC + lowercasing)
    norm_text = normalize_text(text)

    # Extract all valid word tokens using Unicode-aware regular expression
    tokens = TOKEN_PATTERN.findall(norm_text)
    return tokens


def tokenize_with_spans(text: str) -> List[Tuple[str, int, int]]:
    """Tokenizes text and returns each token alongside its start and end character offsets.

    Useful for snippet generation and precise contextual alignment.

    Args:
        text: Input string.

    Returns:
        List of tuples: (token_str, start_char_index, end_char_index)
    """
    if not text:
        return []

    norm_text = normalize_text(text)
    spans = []
    for match in TOKEN_PATTERN.finditer(norm_text):
        spans.append((match.group(), match.start(), match.end()))
    return spans


if __name__ == "__main__":
    # Self-test demonstration when run as a standalone script
    demo_sentence = (
        "தமிழ்நாடு காங்கிரஸ் கமிட்டியில் புதிய நிர்வாகிகள் நியமனம்: செல்வப்பெருந்தகை அறிவிப்பு! "
        "ரூ.1.10 கோடி மான நஷ்டஈடு @ உயர் நீதிமன்றம். AI & Machine Learning 2024-ல் வளர்ச்சி."
    )
    print("=== Tamil Text Preprocessing Demonstration ===")
    print(f"Original Input:\n  {demo_sentence}\n")
    tokens = tokenize(demo_sentence)
    print(f"Tokens Count: {len(tokens)}")
    print(f"Tokens List:\n  {tokens}\n")
    print("Token Spans (first 5):")
    for tok, start, end in tokenize_with_spans(demo_sentence)[:5]:
        print(f"  [{start}:{end}] -> '{tok}'")
