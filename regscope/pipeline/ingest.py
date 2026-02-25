"""Comment ingestion pipeline — downloads comments from Regulations.gov."""

import json
import logging
import sqlite3
from typing import Any

logger = logging.getLogger(__name__)


def run_ingest(db: sqlite3.Connection, docket_id: str, api_key: str, config: dict[str, Any]) -> None:
    """Download all comments for a docket from Regulations.gov.

    Two-phase download:
    1. Bulk download comment headers via paginated list endpoint.
    2. Fetch individual comment details for full text.

    Both phases are resumable — interrupted downloads pick up where they left off.

    Args:
        db: SQLite database connection.
        docket_id: The docket ID to download comments for.
        api_key: Regulations.gov API key.
        config: Application configuration dictionary.
    """
    from regscope.api.regulations import RegulationsClient

    with RegulationsClient(api_key, config) as client:
        # Phase 1: Download docket metadata
        logger.info("Fetching docket metadata for %s", docket_id)
        docket_data = client.get_docket(docket_id)
        if docket_data:
            _store_docket(db, docket_id, docket_data)

        # Phase 2: Download comment headers (list endpoint)
        logger.info("Downloading comment headers for %s", docket_id)
        total_downloaded = _download_comment_headers(db, docket_id, client)
        logger.info("Downloaded %d comment headers", total_downloaded)

        # Phase 3: Fetch individual comment details
        logger.info("Fetching comment details for %s", docket_id)
        total_detailed = _fetch_comment_details(db, docket_id, client)
        logger.info("Fetched details for %d comments", total_detailed)


def _store_docket(db: sqlite3.Connection, docket_id: str, data: dict) -> None:
    """Store docket metadata in the database."""
    attrs = data.get("attributes", {})
    db.execute(
        """INSERT OR REPLACE INTO dockets
           (docket_id, title, agency, docket_type, modified_date, highlighted, raw_json)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            docket_id,
            attrs.get("title", ""),
            attrs.get("agencyId", ""),
            attrs.get("docketType", ""),
            attrs.get("modifyDate", ""),
            attrs.get("highlightedContent", ""),
            json.dumps(data),
        ),
    )
    db.commit()


def _download_comment_headers(db: sqlite3.Connection, docket_id: str, client: "RegulationsClient") -> int:
    """Download comment headers via the list endpoint with cursor-based pagination."""
    from rich.progress import Progress

    last_modified_date = None
    api_fetched = 0

    # Check for existing comments to support resuming
    row = db.execute(
        "SELECT MAX(last_modified_date) FROM comments WHERE docket_id = ?",
        (docket_id,),
    ).fetchone()
    if row and row[0]:
        existing_count = db.execute(
            "SELECT COUNT(*) FROM comments WHERE docket_id = ?", (docket_id,)
        ).fetchone()[0]
        if existing_count > 0:
            logger.info("Resuming download from %s (%d existing comments)", row[0], existing_count)
            last_modified_date = row[0]

    with Progress() as progress:
        task = progress.add_task("Downloading comments...", total=None)

        while True:
            api_response_count = 0
            for page_num in range(1, 21):  # Max 20 pages per cursor window
                comments = client.list_comments(
                    docket_id,
                    page=page_num,
                    last_modified_date=last_modified_date,
                )

                if not comments:
                    break

                for comment_data in comments:
                    _store_comment_header(db, docket_id, comment_data)

                api_response_count += len(comments)
                api_fetched += len(comments)
                db_count = db.execute(
                    "SELECT COUNT(*) FROM comments WHERE docket_id = ?", (docket_id,)
                ).fetchone()[0]
                progress.update(task, completed=db_count, description=f"Downloaded {db_count} comments...")
                db.commit()

                if len(comments) < 250:
                    break

            if api_response_count == 0:
                break

            # If we got a full 20 pages (5000 comments), need to cursor forward
            if api_response_count >= 250 * 20:
                row = db.execute(
                    "SELECT MAX(last_modified_date) FROM comments WHERE docket_id = ?",
                    (docket_id,),
                ).fetchone()
                if row and row[0] and row[0] != last_modified_date:
                    last_modified_date = row[0]
                    continue

            break

    # Return actual count from DB, not the possibly-inflated API response count
    total = db.execute(
        "SELECT COUNT(*) FROM comments WHERE docket_id = ?", (docket_id,)
    ).fetchone()[0]
    return total


def _store_comment_header(db: sqlite3.Connection, docket_id: str, data: dict) -> None:
    """Store a comment header from the list endpoint."""
    comment_id = data.get("id", "")
    attrs = data.get("attributes", {})

    first = attrs.get("firstName") or ""
    last = attrs.get("lastName") or ""
    submitter = f"{first} {last}".strip()
    db.execute(
        """INSERT OR IGNORE INTO comments
           (comment_id, docket_id, document_id, title, posted_date,
            last_modified_date, submitter_name, organization, comment_text,
            detail_fetched, raw_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)""",
        (
            comment_id,
            docket_id,
            attrs.get("objectId") or "",
            attrs.get("title") or "",
            attrs.get("postedDate") or "",
            attrs.get("lastModifiedDate") or "",
            submitter,
            attrs.get("organization") or "",
            attrs.get("comment") or "",
            json.dumps(data),
        ),
    )


def _fetch_comment_details(db: sqlite3.Connection, docket_id: str, client: "RegulationsClient") -> int:
    """Fetch full details for each comment that hasn't been detail-fetched yet."""
    from rich.progress import Progress

    rows = db.execute(
        "SELECT comment_id FROM comments WHERE docket_id = ? AND detail_fetched = 0",
        (docket_id,),
    ).fetchall()

    if not rows:
        logger.info("All comments already have details fetched")
        return 0

    total = 0
    with Progress() as progress:
        task = progress.add_task("Fetching details...", total=len(rows))

        for (comment_id,) in rows:
            try:
                detail = client.get_comment(comment_id)
                if detail:
                    attrs = detail.get("attributes", {})
                    first = attrs.get("firstName") or ""
                    last = attrs.get("lastName") or ""
                    submitter = f"{first} {last}".strip()
                    db.execute(
                        """UPDATE comments SET
                           comment_text = COALESCE(?, comment_text),
                           submitter_name = COALESCE(?, submitter_name),
                           organization = COALESCE(?, organization),
                           detail_fetched = 1,
                           raw_json = ?
                           WHERE comment_id = ?""",
                        (
                            attrs.get("comment") or None,
                            submitter or None,
                            attrs.get("organization") or None,
                            json.dumps(detail),
                            comment_id,
                        ),
                    )

                    # Handle attachments
                    included = detail.get("included", [])
                    if not included and "relationships" in detail.get("attributes", {}):
                        pass  # No attachments in relationships

                    for attachment in (detail.get("included") or []):
                        _store_attachment(db, comment_id, attachment)

                    total += 1
                    if total % 10 == 0:
                        db.commit()
            except Exception:
                logger.exception("Failed to fetch detail for %s", comment_id)

            progress.update(task, advance=1)

    db.commit()
    return total


def _store_attachment(db: sqlite3.Connection, comment_id: str, attachment_data: dict) -> None:
    """Store attachment records from the comment detail response.

    The API returns attachments with a fileFormats array containing multiple
    format options (e.g., both PDF and DOCX). We prefer PDF for extraction,
    and store one row per attachment (not per format).
    """

    attrs = attachment_data.get("attributes", {})
    title = attrs.get("title", "")
    file_formats = attrs.get("fileFormats") or []

    if not file_formats:
        # Fallback: maybe flat structure
        file_url = attrs.get("fileUrl", "")
        file_format = attrs.get("format", "")
        size = attrs.get("size", 0)
    else:
        # Pick the best format: prefer PDF, then DOCX, then first available
        chosen = file_formats[0]
        for ff in file_formats:
            fmt = (ff.get("format") or "").lower()
            if fmt == "pdf":
                chosen = ff
                break
            elif fmt == "docx" and (chosen.get("format") or "").lower() != "pdf":
                chosen = ff

        file_url = chosen.get("fileUrl", "")
        file_format = chosen.get("format", "")
        size = chosen.get("size", 0)

    db.execute(
        """INSERT OR IGNORE INTO attachments
           (comment_id, file_url, file_format, title, file_size, raw_json)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            comment_id,
            file_url,
            file_format,
            title,
            size,
            json.dumps(attachment_data),
        ),
    )
