"""Tests for deduplication pipeline accuracy."""

import sqlite3

import pytest

from regscope.db import get_db
from regscope.utils.text import compute_text_hash, normalize_text, normalize_for_hash, word_ngrams


@pytest.fixture
def test_db(tmp_path):
    """Create a test database with sample data."""
    config = {"data": {"data_dir": str(tmp_path)}}
    db = get_db("TEST-DEDUP", config)
    db.execute("INSERT INTO dockets (docket_id, title) VALUES ('TEST-DEDUP', 'Dedup Test')")
    db.commit()
    return db


def _insert_comment(db, comment_id: str, text: str):
    """Helper to insert a comment with full_text."""
    db.execute(
        """INSERT INTO comments (comment_id, docket_id, full_text, comment_text, detail_fetched)
           VALUES (?, 'TEST-DEDUP', ?, ?, 1)""",
        (comment_id, text, text),
    )
    db.commit()


class TestExactDedup:
    """Test Tier 1: exact duplicate detection via SHA-256."""

    def test_identical_texts_same_hash(self) -> None:
        """Identical texts should produce the same hash."""
        text = "I support this rule."
        h1 = compute_text_hash(text)
        h2 = compute_text_hash(text)
        assert h1 == h2

    def test_case_insensitive(self) -> None:
        """Hashing should be case-insensitive."""
        h1 = compute_text_hash("I Support This Rule.")
        h2 = compute_text_hash("i support this rule.")
        assert h1 == h2

    def test_whitespace_normalization(self) -> None:
        """Extra whitespace should not change the hash."""
        h1 = compute_text_hash("I support this rule.")
        h2 = compute_text_hash("I   support   this   rule.")
        assert h1 == h2

    def test_different_texts_different_hash(self) -> None:
        """Different texts should produce different hashes."""
        h1 = compute_text_hash("I support this rule.")
        h2 = compute_text_hash("I oppose this rule.")
        assert h1 != h2

    def test_punctuation_ignored(self) -> None:
        """Punctuation should be stripped for hashing."""
        h1 = compute_text_hash("I support this rule!")
        h2 = compute_text_hash("I support this rule.")
        assert h1 == h2

    def test_exact_dedup_groups(self, test_db) -> None:
        """Exact duplicates should be grouped together."""
        from regscope.pipeline.dedup import _exact_dedup

        text = "I strongly support this proposed rule."
        _insert_comment(test_db, "C1", text)
        _insert_comment(test_db, "C2", text)
        _insert_comment(test_db, "C3", text)
        _insert_comment(test_db, "C4", "This is a different comment entirely.")

        groups = _exact_dedup(test_db, "TEST-DEDUP")
        assert groups == 1  # Only one group (the 3 duplicates)

        # C1-C3 should share a dedup_group_id, C4 should not
        rows = test_db.execute(
            "SELECT comment_id, dedup_group_id FROM comments ORDER BY comment_id"
        ).fetchall()
        group_ids = {r[0]: r[1] for r in rows}
        assert group_ids["C1"] == group_ids["C2"] == group_ids["C3"]
        assert group_ids["C4"] is None


class TestNearDedup:
    """Test Tier 2: near-duplicate detection via MinHash LSH."""

    def test_word_ngrams(self) -> None:
        """Word n-grams should be generated correctly."""
        ngrams = word_ngrams("the quick brown fox jumps", n=3)
        assert "the quick brown" in ngrams
        assert "quick brown fox" in ngrams
        assert len(ngrams) == 3

    def test_word_ngrams_short_text(self) -> None:
        """Short texts should still produce n-grams."""
        ngrams = word_ngrams("hello world", n=3)
        assert len(ngrams) == 1
        assert ngrams[0] == "hello world"

    def test_near_duplicates_detected(self, test_db) -> None:
        """Form letters with minor edits should be grouped as near-duplicates."""
        from regscope.pipeline.dedup import _near_dedup

        # Use a long base text so the signature changes have minimal impact on Jaccard similarity
        base = (
            "I strongly support this proposed rule because it will protect public health "
            "and the environment for future generations of Americans. The proposed emission "
            "standards are necessary to reduce harmful pollutants that cause respiratory "
            "disease and contribute to climate change. I urge the agency to finalize this "
            "rule as quickly as possible to protect our communities and our children"
        )
        _insert_comment(test_db, "N1", base + ". Sincerely, John Doe")
        _insert_comment(test_db, "N2", base + ". Sincerely, Jane Smith")
        _insert_comment(test_db, "N3", base + ". Best regards, Bob Wilson")
        _insert_comment(test_db, "N4", "This comment is completely different and unrelated to any other comments about environmental policy.")

        groups = _near_dedup(test_db, "TEST-DEDUP", threshold=0.75, num_perm=128)
        assert groups >= 1  # Should find at least one near-dedup group

        # N4 should not be in a group
        rows = test_db.execute(
            "SELECT comment_id, dedup_group_id FROM comments WHERE comment_id = 'N4'"
        ).fetchall()
        assert rows[0][1] is None


class TestTextNormalization:
    """Test text normalization functions."""

    def test_unicode_normalization(self) -> None:
        """Unicode characters should be normalized."""
        text = normalize_text("caf\u00e9")
        assert text == "caf\u00e9"

    def test_whitespace_collapse(self) -> None:
        """Multiple spaces/tabs should collapse to single space."""
        text = normalize_text("hello    world\t\there")
        assert text == "hello world here"

    def test_newline_normalization(self) -> None:
        """Three+ newlines should collapse to two."""
        text = normalize_text("hello\n\n\n\n\nworld")
        assert text == "hello\n\nworld"

    def test_smart_quotes(self) -> None:
        """Smart quotes should be normalized to standard quotes."""
        text = normalize_text("\u201cHello\u201d")
        assert text == '"Hello"'

    def test_empty_string(self) -> None:
        """Empty string should return empty string."""
        assert normalize_text("") == ""
        assert normalize_for_hash("") == ""
