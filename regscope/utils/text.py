"""Text cleaning and normalization utilities."""

import hashlib
import re
import unicodedata


def normalize_text(text: str) -> str:
    """Normalize text for consistent processing.

    Applies: unicode normalization, whitespace collapsing, lowercasing for
    comparison purposes. Returns the cleaned text (not lowercased — that's
    only for hashing/comparison).

    Args:
        text: Raw input text.

    Returns:
        Cleaned and normalized text.
    """
    if not text:
        return ""

    # Unicode normalization (NFC form)
    text = unicodedata.normalize("NFC", text)

    # Replace various dash/hyphen characters with standard hyphen
    text = re.sub(r"[\u2010-\u2015\u2212\uFE58\uFE63\uFF0D]", "-", text)

    # Replace various quote characters with standard quotes
    text = re.sub(r"[\u2018\u2019\u201A\uFF07]", "'", text)
    text = re.sub(r"[\u201C\u201D\u201E\uFF02]", '"', text)

    # Collapse multiple whitespace (spaces, tabs, etc.) into single space
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize line endings
    text = re.sub(r"\r\n?", "\n", text)

    # Collapse 3+ newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def normalize_for_hash(text: str) -> str:
    """Aggressively normalize text for exact-duplicate hashing.

    Lowercases, strips all non-alphanumeric characters except spaces,
    collapses whitespace.

    Args:
        text: Input text.

    Returns:
        Normalized text suitable for hashing.
    """
    text = normalize_text(text).lower()
    # Remove non-alphanumeric except spaces
    text = re.sub(r"[^a-z0-9\s]", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def compute_text_hash(text: str) -> str:
    """Compute SHA-256 hash of normalized text for exact deduplication.

    Args:
        text: Raw comment text.

    Returns:
        Hex-encoded SHA-256 hash of normalized text.
    """
    normalized = normalize_for_hash(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def strip_boilerplate(text: str) -> str:
    """Remove common boilerplate headers/footers added by Regulations.gov.

    Args:
        text: Comment text that may contain boilerplate.

    Returns:
        Text with boilerplate removed.
    """
    # Common Regulations.gov boilerplate patterns
    patterns = [
        r"^Comment on FR Doc #.*?\n",
        r"^Document ID:.*?\n",
        r"^Submitter Information\n.*?(?=\n\n)",
        r"See attached file\(s\)\s*$",
        r"^\s*Page \d+ of \d+\s*$",
    ]
    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.MULTILINE | re.IGNORECASE)

    return text.strip()


def word_ngrams(text: str, n: int = 3) -> list[str]:
    """Generate word n-grams from text for MinHash.

    Args:
        text: Input text.
        n: N-gram size (default 3).

    Returns:
        List of space-joined n-gram strings.
    """
    words = normalize_for_hash(text).split()
    if len(words) < n:
        return [" ".join(words)] if words else []
    return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]


def truncate_text(text: str, max_length: int = 500) -> str:
    """Truncate text to max_length, ending at a word boundary.

    Args:
        text: Input text.
        max_length: Maximum character count.

    Returns:
        Truncated text with ellipsis if truncated.
    """
    if len(text) <= max_length:
        return text

    truncated = text[:max_length]
    # Find last space to avoid cutting mid-word
    last_space = truncated.rfind(" ")
    if last_space > max_length * 0.8:
        truncated = truncated[:last_space]

    return truncated.rstrip() + "..."
