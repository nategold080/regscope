"""Database schema, connection management, and migrations for RegScope."""

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 3

SCHEMA_SQL = """
-- Migration tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Docket metadata
CREATE TABLE IF NOT EXISTS dockets (
    docket_id TEXT PRIMARY KEY,
    title TEXT,
    agency TEXT,
    docket_type TEXT,
    modified_date TEXT,
    highlighted TEXT,
    raw_json TEXT
);

-- Documents within a docket
CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    docket_id TEXT NOT NULL,
    title TEXT,
    document_type TEXT,
    posted_date TEXT,
    comment_start_date TEXT,
    comment_end_date TEXT,
    raw_json TEXT,
    FOREIGN KEY (docket_id) REFERENCES dockets(docket_id)
);

-- Comments
CREATE TABLE IF NOT EXISTS comments (
    comment_id TEXT PRIMARY KEY,
    docket_id TEXT NOT NULL,
    document_id TEXT,
    title TEXT,
    comment_text TEXT,
    full_text TEXT,
    submitter_name TEXT,
    organization TEXT,
    posted_date TEXT,
    last_modified_date TEXT,
    text_hash TEXT,
    dedup_group_id INTEGER,
    semantic_group_id INTEGER,
    detail_fetched INTEGER DEFAULT 0,
    raw_json TEXT,
    FOREIGN KEY (docket_id) REFERENCES dockets(docket_id),
    FOREIGN KEY (dedup_group_id) REFERENCES dedup_groups(dedup_group_id),
    FOREIGN KEY (semantic_group_id) REFERENCES dedup_groups(dedup_group_id)
);

-- Attachments
CREATE TABLE IF NOT EXISTS attachments (
    attachment_id INTEGER PRIMARY KEY,
    comment_id TEXT NOT NULL,
    file_url TEXT,
    file_format TEXT,
    title TEXT,
    file_size INTEGER,
    extracted_text TEXT,
    raw_json TEXT,
    FOREIGN KEY (comment_id) REFERENCES comments(comment_id),
    UNIQUE(comment_id, file_url)
);

-- Embeddings (stored as binary blobs)
CREATE TABLE IF NOT EXISTS embeddings (
    comment_id TEXT PRIMARY KEY,
    embedding BLOB NOT NULL,
    FOREIGN KEY (comment_id) REFERENCES comments(comment_id)
);

-- Dedup groups
CREATE TABLE IF NOT EXISTS dedup_groups (
    dedup_group_id INTEGER PRIMARY KEY,
    docket_id TEXT NOT NULL,
    group_type TEXT NOT NULL,  -- 'exact', 'near', 'semantic'
    group_size INTEGER DEFAULT 0,
    representative_comment_id TEXT,
    template_text TEXT,
    text_hash TEXT,
    FOREIGN KEY (docket_id) REFERENCES dockets(docket_id),
    FOREIGN KEY (representative_comment_id) REFERENCES comments(comment_id)
);

-- Topics
CREATE TABLE IF NOT EXISTS topics (
    topic_id INTEGER PRIMARY KEY,
    docket_id TEXT NOT NULL,
    bertopic_id INTEGER,
    label TEXT,
    keywords TEXT,  -- JSON array
    topic_size INTEGER DEFAULT 0,
    representative_texts TEXT,  -- JSON array
    llm_label TEXT,
    FOREIGN KEY (docket_id) REFERENCES dockets(docket_id)
);

-- Comment-Topic assignments (many-to-many)
CREATE TABLE IF NOT EXISTS comment_topics (
    comment_id TEXT NOT NULL,
    topic_id INTEGER NOT NULL,
    relevance_score REAL DEFAULT 0.0,
    PRIMARY KEY (comment_id, topic_id),
    FOREIGN KEY (comment_id) REFERENCES comments(comment_id),
    FOREIGN KEY (topic_id) REFERENCES topics(topic_id)
);

-- Comment classifications
CREATE TABLE IF NOT EXISTS comment_classifications (
    comment_id TEXT PRIMARY KEY,
    stakeholder_type TEXT,
    stance TEXT,
    stance_confidence REAL,
    substantiveness_score INTEGER,
    FOREIGN KEY (comment_id) REFERENCES comments(comment_id)
);

-- Pipeline run log
CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id INTEGER PRIMARY KEY,
    docket_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    status TEXT NOT NULL,  -- 'started', 'completed', 'failed'
    started_at TEXT NOT NULL DEFAULT (datetime('now')),
    completed_at TEXT,
    parameters TEXT,  -- JSON
    error_message TEXT
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_comments_docket ON comments(docket_id);
CREATE INDEX IF NOT EXISTS idx_comments_posted ON comments(posted_date);
CREATE INDEX IF NOT EXISTS idx_comments_dedup ON comments(dedup_group_id);
CREATE INDEX IF NOT EXISTS idx_comments_semantic ON comments(semantic_group_id);
CREATE INDEX IF NOT EXISTS idx_comments_hash ON comments(text_hash);
CREATE INDEX IF NOT EXISTS idx_comments_detail ON comments(docket_id, detail_fetched);
CREATE INDEX IF NOT EXISTS idx_comments_modified ON comments(last_modified_date);
CREATE INDEX IF NOT EXISTS idx_attachments_comment ON attachments(comment_id);
CREATE INDEX IF NOT EXISTS idx_dedup_groups_docket ON dedup_groups(docket_id);
CREATE INDEX IF NOT EXISTS idx_topics_docket ON topics(docket_id);
CREATE INDEX IF NOT EXISTS idx_comment_topics_topic ON comment_topics(topic_id);
CREATE INDEX IF NOT EXISTS idx_classifications_type ON comment_classifications(stakeholder_type);
CREATE INDEX IF NOT EXISTS idx_classifications_stance ON comment_classifications(stance);
CREATE INDEX IF NOT EXISTS idx_pipeline_runs_docket ON pipeline_runs(docket_id, stage);
CREATE INDEX IF NOT EXISTS idx_documents_docket ON documents(docket_id);
"""


def get_db(docket_id: str, config: dict[str, Any]) -> sqlite3.Connection:
    """Get a database connection for a specific docket.

    Creates the database and schema if they don't exist. Each docket gets
    its own SQLite file.

    Args:
        docket_id: The docket ID (used as filename).
        config: Application configuration dictionary.

    Returns:
        SQLite database connection.
    """
    data_dir = Path(config.get("data", {}).get("data_dir", "~/.regscope/data")).expanduser()
    data_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize docket_id for filename
    safe_name = docket_id.replace("/", "_").replace("\\", "_")
    db_path = data_dir / f"{safe_name}.db"

    logger.debug("Opening database: %s", db_path)

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")

    _apply_migrations(conn)

    return conn


def _apply_migrations(conn: sqlite3.Connection) -> None:
    """Apply database schema migrations.

    Uses a simple version-based migration system. Checks current version
    and applies any pending migrations sequentially.

    Args:
        conn: SQLite database connection.
    """
    # Check if schema_version table exists
    has_version_table = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='schema_version'"
    ).fetchone()[0]

    if has_version_table:
        current_version = conn.execute(
            "SELECT MAX(version) FROM schema_version"
        ).fetchone()[0] or 0
    else:
        current_version = 0

    if current_version < SCHEMA_VERSION:
        logger.info("Applying schema migration v%d → v%d", current_version, SCHEMA_VERSION)
        conn.executescript(SCHEMA_SQL)

        # v1 → v2: add llm_label column to topics
        if current_version < 2:
            try:
                conn.execute("ALTER TABLE topics ADD COLUMN llm_label TEXT")
            except sqlite3.OperationalError:
                pass  # Column already exists (fresh DB from SCHEMA_SQL)

        # v2 → v3: add unique constraint on attachments(comment_id, file_url)
        if current_version < 3:
            try:
                conn.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS idx_attachments_unique "
                    "ON attachments(comment_id, file_url)"
                )
            except sqlite3.OperationalError:
                pass  # Index already exists or duplicates present

        conn.execute(
            "INSERT OR REPLACE INTO schema_version (version) VALUES (?)",
            (SCHEMA_VERSION,),
        )
        conn.commit()
        logger.info("Schema migration complete")


def log_pipeline_run(
    db: sqlite3.Connection,
    docket_id: str,
    stage: str,
    status: str,
    parameters: dict | None = None,
    error_message: str | None = None,
) -> None:
    """Log a pipeline stage execution.

    Args:
        db: SQLite database connection.
        docket_id: The docket ID.
        stage: Pipeline stage name.
        status: Status ('started', 'completed', 'failed').
        parameters: Optional parameters dict (stored as JSON).
        error_message: Optional error message for failed runs.
    """
    db.execute(
        """INSERT INTO pipeline_runs
           (docket_id, stage, status, parameters, error_message, completed_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            docket_id,
            stage,
            status,
            json.dumps(parameters) if parameters else None,
            error_message,
            datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S") if status in ("completed", "failed") else None,
        ),
    )
    db.commit()


def get_pipeline_status(db: sqlite3.Connection, docket_id: str) -> list[dict[str, str]]:
    """Get the pipeline status for a docket.

    Args:
        db: SQLite database connection.
        docket_id: The docket ID.

    Returns:
        List of stage status dicts.
    """
    stages = ["ingest", "extract", "dedup", "embed", "topics", "classify", "report"]
    result = []

    comment_count = db.execute(
        "SELECT COUNT(*) FROM comments WHERE docket_id = ?", (docket_id,)
    ).fetchone()[0]

    for stage in stages:
        row = db.execute(
            """SELECT status, completed_at FROM pipeline_runs
               WHERE docket_id = ? AND stage = ?
               ORDER BY run_id DESC LIMIT 1""",
            (docket_id, stage),
        ).fetchone()

        if row:
            status, completed_at = row
            result.append({
                "stage": stage,
                "status": status,
                "last_run": completed_at or "",
                "details": _stage_details(db, docket_id, stage, comment_count),
            })
        else:
            result.append({
                "stage": stage,
                "status": "not run",
                "last_run": "",
                "details": "",
            })

    return result


def _stage_details(db: sqlite3.Connection, docket_id: str, stage: str, total: int) -> str:
    """Get summary details for a pipeline stage."""
    if stage == "ingest":
        fetched = db.execute(
            "SELECT COUNT(*) FROM comments WHERE docket_id = ? AND detail_fetched = 1",
            (docket_id,),
        ).fetchone()[0]
        return f"{total} comments ({fetched} with details)"

    elif stage == "extract":
        with_text = db.execute(
            "SELECT COUNT(*) FROM comments WHERE docket_id = ? AND full_text IS NOT NULL AND full_text != ''",
            (docket_id,),
        ).fetchone()[0]
        return f"{with_text}/{total} with text"

    elif stage == "dedup":
        groups = db.execute(
            "SELECT COUNT(*) FROM dedup_groups WHERE docket_id = ?", (docket_id,)
        ).fetchone()[0]
        return f"{groups} dedup groups"

    elif stage == "embed":
        embedded = db.execute(
            """SELECT COUNT(*) FROM embeddings e
               JOIN comments c ON e.comment_id = c.comment_id
               WHERE c.docket_id = ?""",
            (docket_id,),
        ).fetchone()[0]
        return f"{embedded} embeddings"

    elif stage == "topics":
        topic_count = db.execute(
            "SELECT COUNT(*) FROM topics WHERE docket_id = ?", (docket_id,)
        ).fetchone()[0]
        return f"{topic_count} topics"

    elif stage == "classify":
        classified = db.execute(
            """SELECT COUNT(*) FROM comment_classifications cc
               JOIN comments c ON cc.comment_id = c.comment_id
               WHERE c.docket_id = ?""",
            (docket_id,),
        ).fetchone()[0]
        return f"{classified}/{total} classified"

    return ""


def list_all_dockets(config: dict[str, Any]) -> list[dict[str, Any]]:
    """List all downloaded dockets across all database files.

    Args:
        config: Application configuration dictionary.

    Returns:
        List of docket info dicts.
    """
    data_dir = Path(config.get("data", {}).get("data_dir", "~/.regscope/data")).expanduser()
    if not data_dir.exists():
        return []

    dockets = []
    for db_file in sorted(data_dir.glob("*.db")):
        conn = None
        try:
            conn = sqlite3.connect(str(db_file))
            row = conn.execute(
                "SELECT docket_id, title FROM dockets LIMIT 1"
            ).fetchone()
            if row:
                count = conn.execute(
                    "SELECT COUNT(*) FROM comments WHERE docket_id = ?", (row[0],)
                ).fetchone()[0]

                last_run = conn.execute(
                    "SELECT MAX(completed_at) FROM pipeline_runs WHERE docket_id = ?",
                    (row[0],),
                ).fetchone()[0]

                dockets.append({
                    "docket_id": row[0],
                    "title": row[1] or "",
                    "comment_count": count,
                    "last_updated": last_run or "",
                })
        except Exception:
            logger.debug("Could not read database: %s", db_file)
        finally:
            if conn:
                conn.close()

    return dockets


# Whitelist of valid column names for comments table to prevent SQL injection
VALID_COMMENT_COLUMNS = {
    "comment_id", "docket_id", "document_id", "title", "comment_text",
    "full_text", "submitter_name", "organization", "posted_date",
    "last_modified_date", "text_hash", "dedup_group_id", "semantic_group_id",
    "detail_fetched", "raw_json",
}


def _validate_columns(columns: set[str]) -> None:
    """Validate that column names are in the whitelist.

    Args:
        columns: Set of column name strings to validate.

    Raises:
        ValueError: If any column name is not in the whitelist.
    """
    invalid = columns - VALID_COMMENT_COLUMNS
    if invalid:
        raise ValueError(f"Invalid column name(s): {', '.join(sorted(invalid))}")


# --- Convenience query functions ---

def count_comments(db: sqlite3.Connection, docket_id: str) -> int:
    """Count total comments for a docket."""
    return db.execute(
        "SELECT COUNT(*) FROM comments WHERE docket_id = ?", (docket_id,)
    ).fetchone()[0]


def get_comments_batch(
    db: sqlite3.Connection,
    docket_id: str,
    offset: int = 0,
    limit: int = 1000,
) -> list[sqlite3.Row]:
    """Get a batch of comments for a docket.

    Args:
        db: SQLite database connection.
        docket_id: The docket ID.
        offset: Row offset for pagination.
        limit: Maximum rows to return.

    Returns:
        List of Row objects.
    """
    db.row_factory = sqlite3.Row
    try:
        rows = db.execute(
            """SELECT * FROM comments
               WHERE docket_id = ?
               ORDER BY posted_date
               LIMIT ? OFFSET ?""",
            (docket_id, limit, offset),
        ).fetchall()
    finally:
        db.row_factory = None
    return rows


def insert_comment(db: sqlite3.Connection, **kwargs: Any) -> None:
    """Insert a comment row.

    Args:
        db: SQLite database connection.
        **kwargs: Column name/value pairs.

    Raises:
        ValueError: If any column name is not in the whitelist.
    """
    _validate_columns(set(kwargs.keys()))
    columns = ", ".join(kwargs.keys())
    placeholders = ", ".join("?" * len(kwargs))
    db.execute(
        f"INSERT OR IGNORE INTO comments ({columns}) VALUES ({placeholders})",
        tuple(kwargs.values()),
    )


def update_comment_field(db: sqlite3.Connection, comment_id: str, **kwargs: Any) -> None:
    """Update specific fields on a comment.

    Args:
        db: SQLite database connection.
        comment_id: The comment ID to update.
        **kwargs: Column name/value pairs to update.

    Raises:
        ValueError: If any column name is not in the whitelist.
    """
    _validate_columns(set(kwargs.keys()))
    set_clause = ", ".join(f"{k} = ?" for k in kwargs.keys())
    db.execute(
        f"UPDATE comments SET {set_clause} WHERE comment_id = ?",
        (*kwargs.values(), comment_id),
    )
