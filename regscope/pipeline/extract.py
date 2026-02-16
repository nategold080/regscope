"""Text extraction pipeline — extracts text from attachments and builds full_text."""

import logging
import sqlite3
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def run_extract(db: sqlite3.Connection, docket_id: str, config: dict[str, Any]) -> None:
    """Extract text from all attachments and build unified full_text for each comment.

    For PDF attachments: uses PyMuPDF (fitz) as primary extractor, falls back to
    Tesseract OCR for scanned documents.

    Args:
        db: SQLite database connection.
        docket_id: The docket ID to process.
        config: Application configuration dictionary.
    """
    from rich.progress import Progress

    # Process attachments that haven't been extracted yet
    attachments = db.execute(
        """SELECT a.attachment_id, a.comment_id, a.file_url, a.file_format
           FROM attachments a
           JOIN comments c ON a.comment_id = c.comment_id
           WHERE c.docket_id = ? AND a.extracted_text IS NULL AND a.file_url != ''""",
        (docket_id,),
    ).fetchall()

    if attachments:
        logger.info("Extracting text from %d attachments", len(attachments))
        with Progress() as progress:
            task = progress.add_task("Extracting text...", total=len(attachments))
            for att_id, comment_id, file_url, file_format in attachments:
                try:
                    text = _extract_attachment(file_url, file_format, config)
                    db.execute(
                        "UPDATE attachments SET extracted_text = ? WHERE attachment_id = ?",
                        (text, att_id),
                    )
                    if att_id % 10 == 0:
                        db.commit()
                except Exception:
                    logger.exception("Failed to extract text from attachment %d", att_id)

                progress.update(task, advance=1)

        db.commit()

    # Build full_text for all comments
    _build_full_text(db, docket_id)


def _extract_attachment(file_url: str, file_format: str, config: dict[str, Any]) -> str:
    """Extract text from an attachment file.

    Args:
        file_url: URL to download the attachment from.
        file_format: File format (pdf, docx, html, txt, etc.).
        config: Application configuration.

    Returns:
        Extracted text content.
    """
    import httpx
    import time

    # Download the file — downloads.regulations.gov WAF blocks bot User-Agents
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/pdf,application/octet-stream,*/*",
    }
    try:
        response = httpx.get(
            file_url, headers=headers, follow_redirects=True, timeout=60.0,
        )
        response.raise_for_status()
        content = response.content
    except httpx.HTTPStatusError as e:
        logger.error(
            "HTTP %d downloading attachment: %s", e.response.status_code, file_url,
        )
        return ""
    except Exception:
        logger.exception("Failed to download attachment: %s", file_url)
        return ""

    fmt = file_format.lower().strip(".")

    if fmt == "pdf":
        return _extract_pdf(content)
    elif fmt in ("txt", "text"):
        return content.decode("utf-8", errors="replace")
    elif fmt in ("htm", "html"):
        return _extract_html(content)
    elif fmt in ("docx", "doc"):
        return _extract_docx(content)
    else:
        logger.warning("Unsupported attachment format: %s", file_format)
        return ""


def _extract_pdf(content: bytes) -> str:
    """Extract text from PDF content using PyMuPDF, with OCR fallback.

    Args:
        content: Raw PDF file bytes.

    Returns:
        Extracted text.
    """
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(stream=content, filetype="pdf")
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        doc.close()

        text = "\n".join(text_parts).strip()

        # If we got very little text, it might be scanned — try OCR
        if len(text) < 50:
            logger.info("PDF appears scanned (only %d chars), attempting OCR", len(text))
            text = _ocr_pdf(content)

        return text

    except ImportError:
        logger.error("PyMuPDF (fitz) not installed")
        return ""
    except Exception:
        logger.exception("Failed to extract text from PDF")
        return ""


def _ocr_pdf(content: bytes) -> str:
    """OCR a scanned PDF using Tesseract.

    Args:
        content: Raw PDF file bytes.

    Returns:
        OCR'd text.
    """
    try:
        import tempfile
        from pdf2image import convert_from_bytes
        import pytesseract

        images = convert_from_bytes(content)
        text_parts = []
        for img in images:
            text_parts.append(pytesseract.image_to_string(img))

        return "\n".join(text_parts).strip()

    except ImportError:
        logger.warning("pytesseract or pdf2image not installed — OCR unavailable")
        return ""
    except Exception:
        logger.exception("OCR failed")
        return ""


def _extract_html(content: bytes) -> str:
    """Extract text from HTML content.

    Args:
        content: Raw HTML bytes.

    Returns:
        Plain text extracted from HTML.
    """
    import re

    text = content.decode("utf-8", errors="replace")
    # Remove script and style tags
    text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Remove all HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Clean up whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_docx(content: bytes) -> str:
    """Extract text from DOCX content.

    Args:
        content: Raw DOCX file bytes.

    Returns:
        Extracted text.
    """
    try:
        import zipfile
        import io
        import re

        zf = zipfile.ZipFile(io.BytesIO(content))
        xml_content = zf.read("word/document.xml").decode("utf-8")
        # Extract text between XML tags
        text = re.sub(r"<[^>]+>", "", xml_content)
        return text.strip()
    except Exception:
        logger.exception("Failed to extract text from DOCX")
        return ""


_STUB_PATTERNS = [
    r"^see attached\s*(?:file|document|comment)?\s*[.(s)]*\s*$",
    r"^please\s+(?:see|find)\s+(?:the\s+)?attached\b.*$",
    r"^attached\s+(?:is|are|please)\b.*$",
    r"^comment\s+(?:is\s+)?attached\b.*$",
]


def _is_stub_text(text: str) -> bool:
    """Check if comment text is just a 'see attached' stub.

    Args:
        text: Cleaned comment text.

    Returns:
        True if the text is a stub referencing an attachment.
    """
    import re

    if not text:
        return False
    stripped = re.sub(r"<[^>]+>", " ", text)  # Remove HTML tags
    stripped = re.sub(r"\s+", " ", stripped).strip()
    if len(stripped) > 200:
        return False
    for pattern in _STUB_PATTERNS:
        if re.match(pattern, stripped, re.IGNORECASE):
            return True
    return False


def _build_full_text(db: sqlite3.Connection, docket_id: str) -> None:
    """Build the full_text field for each comment by concatenating comment body and attachments.

    For comments where the body is just a "see attached" stub, the full_text
    is replaced entirely by the extracted attachment text. If attachment
    extraction failed for these stubs, the comment is flagged with empty text.

    Args:
        db: SQLite database connection.
        docket_id: The docket ID to process.
    """
    from regscope.utils.text import normalize_text, strip_boilerplate

    comments = db.execute(
        "SELECT comment_id, comment_text FROM comments WHERE docket_id = ?",
        (docket_id,),
    ).fetchall()

    logger.info("Building full_text for %d comments", len(comments))

    for comment_id, comment_text in comments:
        # Collect attachment texts
        att_rows = db.execute(
            "SELECT extracted_text FROM attachments WHERE comment_id = ? AND extracted_text IS NOT NULL",
            (comment_id,),
        ).fetchall()
        attachment_texts = []
        for (att_text,) in att_rows:
            if att_text:
                cleaned_att = normalize_text(att_text)
                if cleaned_att:
                    attachment_texts.append(cleaned_att)

        # Check if comment body is a stub
        cleaned_body = ""
        if comment_text:
            cleaned_body = strip_boilerplate(normalize_text(comment_text))

        is_stub = _is_stub_text(cleaned_body)

        if is_stub and attachment_texts:
            # Replace stub with attachment text only
            full_text = "\n\n".join(attachment_texts)
        elif is_stub and not attachment_texts:
            # Stub with no attachment extraction — leave empty so it's not
            # treated as a real comment
            full_text = ""
            logger.debug("Stub comment %s has no extracted attachment text", comment_id)
        else:
            # Normal: concatenate body + attachments
            parts = []
            if cleaned_body:
                parts.append(cleaned_body)
            parts.extend(attachment_texts)
            full_text = "\n\n".join(parts) if parts else ""

        db.execute(
            "UPDATE comments SET full_text = ? WHERE comment_id = ?",
            (full_text, comment_id),
        )

    db.commit()
