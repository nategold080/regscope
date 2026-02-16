"""Tests for text extraction pipeline — stub detection and full_text building."""

import sqlite3

import pytest

from regscope.db import get_db
from regscope.pipeline.extract import _is_stub_text, _build_full_text


@pytest.fixture
def test_db(tmp_path):
    """Create a test database with a docket."""
    config = {"data": {"data_dir": str(tmp_path)}}
    db = get_db("TEST-EXTRACT", config)
    db.execute("INSERT INTO dockets (docket_id, title) VALUES ('TEST-EXTRACT', 'Extract Test')")
    db.commit()
    return db


class TestStubTextDetection:
    """Test that 'see attached' stub text is detected and replaced."""

    def test_see_attached_is_stub(self) -> None:
        """'See attached' variants should be recognized as stubs."""
        assert _is_stub_text("See attached.")
        assert _is_stub_text("See attached file")
        assert _is_stub_text("Please see the attached document")
        assert _is_stub_text("Attached is my comment")
        assert _is_stub_text("Comment attached")

    def test_real_comment_not_stub(self) -> None:
        """Real substantive comments should not be flagged as stubs."""
        assert not _is_stub_text("I support this rule because it protects public health.")
        assert not _is_stub_text("")  # empty is not a stub
        assert not _is_stub_text("A" * 300)  # long text is not a stub

    def test_stub_replaced_by_attachment_text(self, test_db) -> None:
        """When comment body is a stub and attachment text exists, full_text
        should contain only the attachment text, not the stub."""
        # Insert a comment with a stub body
        test_db.execute(
            """INSERT INTO comments
               (comment_id, docket_id, comment_text, detail_fetched)
               VALUES ('STUB-1', 'TEST-EXTRACT', 'See attached.', 1)"""
        )
        # Insert an attachment with extracted text
        test_db.execute(
            """INSERT INTO attachments
               (comment_id, file_url, file_format, extracted_text)
               VALUES ('STUB-1', 'http://example.com/doc.pdf', 'pdf',
                       'This is the real substantive comment from the PDF attachment.')"""
        )
        test_db.commit()

        _build_full_text(test_db, "TEST-EXTRACT")

        row = test_db.execute(
            "SELECT full_text FROM comments WHERE comment_id = 'STUB-1'"
        ).fetchone()
        full_text = row[0]

        # The stub text should NOT appear in full_text
        assert "see attached" not in full_text.lower()
        # The attachment text SHOULD appear
        assert "real substantive comment from the PDF" in full_text
