"""Tests for the comment ingestion pipeline."""

import json
import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from regscope.db import get_db
from regscope.pipeline.ingest import (
    _download_comment_headers,
    _fetch_comment_details,
    _store_attachment,
    _store_comment_header,
)


DOCKET_ID = "TEST-INGEST"


@pytest.fixture
def test_db(tmp_path):
    """Create a test database with a pre-existing docket row."""
    config = {"data": {"data_dir": str(tmp_path)}}
    db = get_db(DOCKET_ID, config)
    db.execute(
        "INSERT INTO dockets (docket_id, title) VALUES (?, 'Ingest Test')",
        (DOCKET_ID,),
    )
    db.commit()
    return db


def _make_comment_data(
    comment_id: str,
    text: str = "Sample comment text",
    first_name: str = "Jane",
    last_name: str = "Doe",
    organization: str = "",
    posted_date: str = "2025-06-01T00:00:00Z",
    last_modified: str = "2025-06-01T00:00:00Z",
) -> dict:
    """Build a fake API comment data dict matching the Regulations.gov list endpoint format."""
    return {
        "id": comment_id,
        "type": "comments",
        "attributes": {
            "objectId": f"DOC-{comment_id}",
            "title": f"Comment {comment_id}",
            "firstName": first_name,
            "lastName": last_name,
            "organization": organization,
            "comment": text,
            "postedDate": posted_date,
            "lastModifiedDate": last_modified,
        },
    }


def _make_detail_response(
    comment_id: str,
    text: str = "Full detail text",
    first_name: str = "Jane",
    last_name: str = "Doe",
    organization: str = "Acme Corp",
    attachments: list | None = None,
) -> dict:
    """Build a fake API detail response for a single comment."""
    detail = {
        "id": comment_id,
        "type": "comments",
        "attributes": {
            "firstName": first_name,
            "lastName": last_name,
            "organization": organization,
            "comment": text,
        },
        "included": attachments or [],
    }
    return detail


def _make_attachment(
    title: str = "Attachment 1",
    file_formats: list[dict] | None = None,
) -> dict:
    """Build a fake attachment record as returned in the comment detail 'included' array."""
    if file_formats is None:
        file_formats = [
            {"fileUrl": "https://example.com/file.pdf", "format": "pdf", "size": 12345},
        ]
    return {
        "id": "ATT-001",
        "type": "attachments",
        "attributes": {
            "title": title,
            "fileFormats": file_formats,
        },
    }


# ---------------------------------------------------------------------------
# _store_comment_header
# ---------------------------------------------------------------------------

class TestStoreCommentHeader:
    """Tests for _store_comment_header — storing individual comments from the list endpoint."""

    def test_stores_comment_fields(self, test_db: sqlite3.Connection) -> None:
        """A comment header should be inserted with all expected fields."""
        data = _make_comment_data(
            "CMT-001",
            text="I support this proposed rule.",
            first_name="John",
            last_name="Smith",
            organization="Sierra Club",
            posted_date="2025-07-15T12:00:00Z",
            last_modified="2025-07-16T08:30:00Z",
        )

        _store_comment_header(test_db, DOCKET_ID, data)
        test_db.commit()

        row = test_db.execute(
            "SELECT comment_id, docket_id, document_id, title, comment_text, "
            "submitter_name, organization, posted_date, last_modified_date, "
            "detail_fetched FROM comments WHERE comment_id = 'CMT-001'"
        ).fetchone()

        assert row is not None
        assert row[0] == "CMT-001"
        assert row[1] == DOCKET_ID
        assert row[2] == "DOC-CMT-001"
        assert row[3] == "Comment CMT-001"
        assert row[4] == "I support this proposed rule."
        assert row[5] == "John Smith"
        assert row[6] == "Sierra Club"
        assert row[7] == "2025-07-15T12:00:00Z"
        assert row[8] == "2025-07-16T08:30:00Z"
        assert row[9] == 0  # detail_fetched

    def test_insert_or_ignore_on_duplicate(self, test_db: sqlite3.Connection) -> None:
        """Inserting the same comment_id twice should not raise; original is kept."""
        data = _make_comment_data("CMT-DUP", text="Original text")
        _store_comment_header(test_db, DOCKET_ID, data)
        test_db.commit()

        data2 = _make_comment_data("CMT-DUP", text="Updated text")
        _store_comment_header(test_db, DOCKET_ID, data2)
        test_db.commit()

        row = test_db.execute(
            "SELECT comment_text FROM comments WHERE comment_id = 'CMT-DUP'"
        ).fetchone()
        assert row[0] == "Original text"

    def test_missing_optional_fields(self, test_db: sqlite3.Connection) -> None:
        """Missing optional fields should default to empty strings, not None errors."""
        data = {
            "id": "CMT-SPARSE",
            "attributes": {},
        }
        _store_comment_header(test_db, DOCKET_ID, data)
        test_db.commit()

        row = test_db.execute(
            "SELECT comment_id, submitter_name, organization, comment_text "
            "FROM comments WHERE comment_id = 'CMT-SPARSE'"
        ).fetchone()
        assert row is not None
        assert row[0] == "CMT-SPARSE"
        assert row[1] == ""  # submitter_name
        assert row[2] == ""  # organization
        assert row[3] == ""  # comment_text

    def test_stores_raw_json(self, test_db: sqlite3.Connection) -> None:
        """The raw API response should be stored as JSON in raw_json."""
        data = _make_comment_data("CMT-RAW")
        _store_comment_header(test_db, DOCKET_ID, data)
        test_db.commit()

        row = test_db.execute(
            "SELECT raw_json FROM comments WHERE comment_id = 'CMT-RAW'"
        ).fetchone()
        parsed = json.loads(row[0])
        assert parsed["id"] == "CMT-RAW"
        assert "attributes" in parsed


# ---------------------------------------------------------------------------
# _store_attachment
# ---------------------------------------------------------------------------

class TestStoreAttachment:
    """Tests for _store_attachment — storing attachment records from comment detail."""

    def _insert_parent_comment(self, db: sqlite3.Connection, comment_id: str = "CMT-ATT") -> None:
        """Insert a parent comment so the foreign key is satisfied."""
        db.execute(
            "INSERT INTO comments (comment_id, docket_id, detail_fetched) VALUES (?, ?, 1)",
            (comment_id, DOCKET_ID),
        )
        db.commit()

    def test_stores_single_format(self, test_db: sqlite3.Connection) -> None:
        """An attachment with a single format should be stored correctly."""
        self._insert_parent_comment(test_db)
        attachment = _make_attachment(
            title="My Document",
            file_formats=[
                {"fileUrl": "https://example.com/doc.pdf", "format": "pdf", "size": 5000},
            ],
        )

        _store_attachment(test_db, "CMT-ATT", attachment)
        test_db.commit()

        row = test_db.execute(
            "SELECT comment_id, file_url, file_format, title, file_size "
            "FROM attachments WHERE comment_id = 'CMT-ATT'"
        ).fetchone()
        assert row is not None
        assert row[0] == "CMT-ATT"
        assert row[1] == "https://example.com/doc.pdf"
        assert row[2] == "pdf"
        assert row[3] == "My Document"
        assert row[4] == 5000

    def test_prefers_pdf_format(self, test_db: sqlite3.Connection) -> None:
        """When multiple formats are available, PDF should be preferred."""
        self._insert_parent_comment(test_db)
        attachment = _make_attachment(
            title="Multi-Format Doc",
            file_formats=[
                {"fileUrl": "https://example.com/doc.docx", "format": "docx", "size": 3000},
                {"fileUrl": "https://example.com/doc.pdf", "format": "pdf", "size": 4000},
                {"fileUrl": "https://example.com/doc.html", "format": "html", "size": 2000},
            ],
        )

        _store_attachment(test_db, "CMT-ATT", attachment)
        test_db.commit()

        row = test_db.execute(
            "SELECT file_url, file_format FROM attachments WHERE comment_id = 'CMT-ATT'"
        ).fetchone()
        assert row[0] == "https://example.com/doc.pdf"
        assert row[1] == "pdf"

    def test_prefers_docx_over_other_non_pdf(self, test_db: sqlite3.Connection) -> None:
        """When PDF is not available, DOCX should be preferred over other formats."""
        self._insert_parent_comment(test_db)
        attachment = _make_attachment(
            title="No PDF Doc",
            file_formats=[
                {"fileUrl": "https://example.com/doc.html", "format": "html", "size": 1000},
                {"fileUrl": "https://example.com/doc.docx", "format": "docx", "size": 6000},
            ],
        )

        _store_attachment(test_db, "CMT-ATT", attachment)
        test_db.commit()

        row = test_db.execute(
            "SELECT file_url, file_format FROM attachments WHERE comment_id = 'CMT-ATT'"
        ).fetchone()
        assert row[0] == "https://example.com/doc.docx"
        assert row[1] == "docx"

    def test_fallback_to_first_format(self, test_db: sqlite3.Connection) -> None:
        """When neither PDF nor DOCX is available, the first format should be used."""
        self._insert_parent_comment(test_db)
        attachment = _make_attachment(
            title="Exotic Format",
            file_formats=[
                {"fileUrl": "https://example.com/doc.rtf", "format": "rtf", "size": 800},
                {"fileUrl": "https://example.com/doc.txt", "format": "txt", "size": 500},
            ],
        )

        _store_attachment(test_db, "CMT-ATT", attachment)
        test_db.commit()

        row = test_db.execute(
            "SELECT file_url, file_format FROM attachments WHERE comment_id = 'CMT-ATT'"
        ).fetchone()
        assert row[0] == "https://example.com/doc.rtf"
        assert row[1] == "rtf"

    def test_empty_file_formats_uses_flat_fields(self, test_db: sqlite3.Connection) -> None:
        """When fileFormats is empty, fall back to flat fileUrl/format attributes."""
        self._insert_parent_comment(test_db)
        attachment = {
            "id": "ATT-FLAT",
            "type": "attachments",
            "attributes": {
                "title": "Flat Attachment",
                "fileFormats": [],
                "fileUrl": "https://example.com/flat.pdf",
                "format": "pdf",
                "size": 9999,
            },
        }

        _store_attachment(test_db, "CMT-ATT", attachment)
        test_db.commit()

        row = test_db.execute(
            "SELECT file_url, file_format, file_size FROM attachments WHERE comment_id = 'CMT-ATT'"
        ).fetchone()
        assert row[0] == "https://example.com/flat.pdf"
        assert row[1] == "pdf"
        assert row[2] == 9999

    def test_stores_raw_json(self, test_db: sqlite3.Connection) -> None:
        """The full attachment data should be persisted in raw_json."""
        self._insert_parent_comment(test_db)
        attachment = _make_attachment(title="JSON Test")
        _store_attachment(test_db, "CMT-ATT", attachment)
        test_db.commit()

        row = test_db.execute(
            "SELECT raw_json FROM attachments WHERE comment_id = 'CMT-ATT'"
        ).fetchone()
        parsed = json.loads(row[0])
        assert parsed["attributes"]["title"] == "JSON Test"


# ---------------------------------------------------------------------------
# _download_comment_headers (mocked client)
# ---------------------------------------------------------------------------

class TestDownloadCommentHeaders:
    """Tests for _download_comment_headers with a mocked RegulationsClient."""

    def test_single_page_docket(self, test_db: sqlite3.Connection) -> None:
        """A docket with fewer than 250 comments should download in one page."""
        client = MagicMock()
        comments = [_make_comment_data(f"C-{i}") for i in range(10)]
        # First call returns 10 comments (< 250, so pagination stops)
        client.list_comments.return_value = comments

        with patch("rich.progress.Progress"):
            total = _download_comment_headers(test_db, DOCKET_ID, client)

        assert total == 10
        count = test_db.execute(
            "SELECT COUNT(*) FROM comments WHERE docket_id = ?", (DOCKET_ID,)
        ).fetchone()[0]
        assert count == 10

        # Only one call needed (single page with < 250 results)
        assert client.list_comments.call_count == 1

    def test_multi_page_docket(self, test_db: sqlite3.Connection) -> None:
        """A docket with > 250 comments but <= 5000 should paginate across multiple pages."""
        client = MagicMock()

        page1 = [_make_comment_data(f"P1-{i}") for i in range(250)]
        page2 = [_make_comment_data(f"P2-{i}") for i in range(100)]

        # Page 1 returns 250 (full page), page 2 returns 100 (partial = last page)
        client.list_comments.side_effect = [page1, page2]

        with patch("rich.progress.Progress"):
            total = _download_comment_headers(test_db, DOCKET_ID, client)

        assert total == 350
        count = test_db.execute(
            "SELECT COUNT(*) FROM comments WHERE docket_id = ?", (DOCKET_ID,)
        ).fetchone()[0]
        assert count == 350
        assert client.list_comments.call_count == 2

    def test_empty_docket(self, test_db: sqlite3.Connection) -> None:
        """A docket with no comments should return 0 and make one API call."""
        client = MagicMock()
        client.list_comments.return_value = []

        with patch("rich.progress.Progress"):
            total = _download_comment_headers(test_db, DOCKET_ID, client)

        assert total == 0
        client.list_comments.assert_called_once()

    def test_cursor_advance_on_full_window(self, test_db: sqlite3.Connection) -> None:
        """When 20 pages of 250 are exhausted (5000), cursor should advance via lastModifiedDate."""
        client = MagicMock()

        # Window 1: 20 full pages = 5000 comments, all with date "2025-06-01"
        window1_pages = []
        for page_num in range(1, 21):
            page = [
                _make_comment_data(
                    f"W1P{page_num}-{i}",
                    last_modified="2025-06-01T00:00:00Z",
                )
                for i in range(250)
            ]
            window1_pages.append(page)

        # Window 2: After cursor advance to "2025-06-02", one partial page
        window2_page = [
            _make_comment_data(
                f"W2-{i}",
                last_modified="2025-06-02T00:00:00Z",
            )
            for i in range(50)
        ]

        call_count = 0

        def mock_list_comments(docket_id, page=1, last_modified_date=None):
            nonlocal call_count
            call_count += 1
            if last_modified_date is None or last_modified_date == "2025-06-01T00:00:00Z":
                if page <= 20 and (page - 1) < len(window1_pages):
                    return window1_pages[page - 1]
                return []
            elif last_modified_date == "2025-06-02T00:00:00Z":
                # After cursor advance
                if page == 1:
                    return window2_page
                return []
            return []

        client.list_comments.side_effect = mock_list_comments

        # We need the cursor to see different dates. Insert a comment with a later date
        # to simulate the cursor advancing. The trick: window1 all have "2025-06-01",
        # so MAX(last_modified_date) stays "2025-06-01" → cursor can't advance → loop breaks.
        # To test actual cursor advance, window 1's LAST page must have a later date.
        # Let me fix: give the last page of window 1 a later date.
        window1_pages[19] = [
            _make_comment_data(
                f"W1P20-{i}",
                last_modified="2025-06-02T00:00:00Z",
            )
            for i in range(250)
        ]

        with patch("rich.progress.Progress"):
            total = _download_comment_headers(test_db, DOCKET_ID, client)

        # Should have 5000 from window 1 + 50 from window 2 = 5050
        assert total == 5050
        count = test_db.execute(
            "SELECT COUNT(*) FROM comments WHERE docket_id = ?", (DOCKET_ID,)
        ).fetchone()[0]
        assert count == 5050

        # Verify cursor advance: list_comments should have been called with
        # last_modified_date="2025-06-02..." for the second window
        calls = client.list_comments.call_args_list
        cursor_calls = [c for c in calls if c[1].get("last_modified_date") == "2025-06-02T00:00:00Z"]
        assert len(cursor_calls) > 0, "Expected cursor advance calls with updated lastModifiedDate"


# ---------------------------------------------------------------------------
# _fetch_comment_details (mocked client)
# ---------------------------------------------------------------------------

class TestFetchCommentDetails:
    """Tests for _fetch_comment_details with a mocked RegulationsClient."""

    def _insert_unfetched_comment(
        self,
        db: sqlite3.Connection,
        comment_id: str,
        detail_fetched: int = 0,
    ) -> None:
        """Insert a comment row that has not yet had its details fetched."""
        db.execute(
            "INSERT INTO comments (comment_id, docket_id, comment_text, detail_fetched) "
            "VALUES (?, ?, 'placeholder', ?)",
            (comment_id, DOCKET_ID, detail_fetched),
        )
        db.commit()

    def test_fetches_details_and_updates(self, test_db: sqlite3.Connection) -> None:
        """detail_fetched=0 comments should be updated with full detail from the API."""
        self._insert_unfetched_comment(test_db, "D-001")

        client = MagicMock()
        detail = _make_detail_response(
            "D-001",
            text="Full comment text from the detail endpoint.",
            organization="EPA Watch",
        )
        client.get_comment.return_value = detail

        with patch("rich.progress.Progress"):
            total = _fetch_comment_details(test_db, DOCKET_ID, client)

        assert total == 1

        row = test_db.execute(
            "SELECT comment_text, organization, detail_fetched "
            "FROM comments WHERE comment_id = 'D-001'"
        ).fetchone()
        assert row[0] == "Full comment text from the detail endpoint."
        assert row[1] == "EPA Watch"
        assert row[2] == 1  # detail_fetched flipped to 1

    def test_skips_already_fetched(self, test_db: sqlite3.Connection) -> None:
        """Comments with detail_fetched=1 should be skipped entirely."""
        self._insert_unfetched_comment(test_db, "SKIP-001", detail_fetched=1)

        client = MagicMock()

        with patch("rich.progress.Progress"):
            total = _fetch_comment_details(test_db, DOCKET_ID, client)

        assert total == 0
        client.get_comment.assert_not_called()

    def test_resume_after_interruption(self, test_db: sqlite3.Connection) -> None:
        """Only unfetched comments should be processed when resuming."""
        # Simulate a prior partial run: 2 already fetched, 1 still pending
        self._insert_unfetched_comment(test_db, "DONE-1", detail_fetched=1)
        self._insert_unfetched_comment(test_db, "DONE-2", detail_fetched=1)
        self._insert_unfetched_comment(test_db, "PENDING-1", detail_fetched=0)

        client = MagicMock()
        client.get_comment.return_value = _make_detail_response("PENDING-1", text="Got it")

        with patch("rich.progress.Progress"):
            total = _fetch_comment_details(test_db, DOCKET_ID, client)

        assert total == 1
        # Only the pending comment should have been requested
        client.get_comment.assert_called_once_with("PENDING-1")

        # Verify the pending one is now marked as fetched
        row = test_db.execute(
            "SELECT detail_fetched FROM comments WHERE comment_id = 'PENDING-1'"
        ).fetchone()
        assert row[0] == 1

    def test_stores_attachments_from_detail(self, test_db: sqlite3.Connection) -> None:
        """Attachments included in the detail response should be stored."""
        self._insert_unfetched_comment(test_db, "ATT-CMT")

        attachment = _make_attachment(
            title="Important Letter",
            file_formats=[
                {"fileUrl": "https://example.com/letter.pdf", "format": "pdf", "size": 20000},
            ],
        )
        detail = _make_detail_response("ATT-CMT", attachments=[attachment])

        client = MagicMock()
        client.get_comment.return_value = detail

        with patch("rich.progress.Progress"):
            _fetch_comment_details(test_db, DOCKET_ID, client)

        row = test_db.execute(
            "SELECT file_url, file_format, title, file_size "
            "FROM attachments WHERE comment_id = 'ATT-CMT'"
        ).fetchone()
        assert row is not None
        assert row[0] == "https://example.com/letter.pdf"
        assert row[1] == "pdf"
        assert row[2] == "Important Letter"
        assert row[3] == 20000

    def test_handles_api_failure_gracefully(self, test_db: sqlite3.Connection) -> None:
        """If get_comment raises an exception, the pipeline should continue."""
        self._insert_unfetched_comment(test_db, "FAIL-1")
        self._insert_unfetched_comment(test_db, "OK-1")

        client = MagicMock()

        def side_effect(comment_id: str):
            if comment_id == "FAIL-1":
                raise ConnectionError("Simulated network error")
            return _make_detail_response(comment_id, text="Success")

        client.get_comment.side_effect = side_effect

        with patch("rich.progress.Progress"):
            total = _fetch_comment_details(test_db, DOCKET_ID, client)

        # Only the successful one should be counted
        assert total == 1

        # FAIL-1 should still be unfetched
        row = test_db.execute(
            "SELECT detail_fetched FROM comments WHERE comment_id = 'FAIL-1'"
        ).fetchone()
        assert row[0] == 0

        # OK-1 should be fetched
        row = test_db.execute(
            "SELECT detail_fetched FROM comments WHERE comment_id = 'OK-1'"
        ).fetchone()
        assert row[0] == 1

    def test_returns_zero_when_all_fetched(self, test_db: sqlite3.Connection) -> None:
        """If every comment already has details, return 0 immediately."""
        self._insert_unfetched_comment(test_db, "ALL-DONE-1", detail_fetched=1)
        self._insert_unfetched_comment(test_db, "ALL-DONE-2", detail_fetched=1)

        client = MagicMock()

        with patch("rich.progress.Progress"):
            total = _fetch_comment_details(test_db, DOCKET_ID, client)

        assert total == 0
        client.get_comment.assert_not_called()
