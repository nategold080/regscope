"""Embedding generation pipeline — generates sentence embeddings for comments."""

import json
import logging
import sqlite3
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _get_expected_embedding_dim(db: sqlite3.Connection, docket_id: str) -> int | None:
    """Get the embedding dimension previously recorded for this docket.

    Checks both the pipeline_runs parameters and existing embeddings in the
    database to determine the expected dimension.

    Args:
        db: SQLite database connection.
        docket_id: The docket ID.

    Returns:
        The expected embedding dimension, or None if no embeddings exist yet.
    """
    # First check pipeline_runs for a recorded dimension
    row = db.execute(
        """SELECT parameters FROM pipeline_runs
           WHERE docket_id = ? AND stage = 'embed' AND status = 'completed'
           ORDER BY run_id DESC LIMIT 1""",
        (docket_id,),
    ).fetchone()
    if row and row[0]:
        try:
            params = json.loads(row[0])
            dim = params.get("embedding_dim")
            if dim is not None:
                return int(dim)
        except (json.JSONDecodeError, ValueError):
            pass

    # Fallback: check the dimension of an existing embedding in the database
    row = db.execute(
        """SELECT e.embedding FROM embeddings e
           JOIN comments c ON e.comment_id = c.comment_id
           WHERE c.docket_id = ?
           LIMIT 1""",
        (docket_id,),
    ).fetchone()
    if row and row[0]:
        return len(row[0]) // 4  # float32 = 4 bytes each

    return None


def _record_embedding_dim(db: sqlite3.Connection, docket_id: str, dim: int, model_name: str) -> None:
    """Record the embedding dimension and model on the most recent embed pipeline run.

    Updates the parameters of the latest embed run rather than inserting a new row,
    to avoid duplicate pipeline_runs entries.

    Args:
        db: SQLite database connection.
        docket_id: The docket ID.
        dim: The embedding dimension.
        model_name: The model name used.
    """
    params = json.dumps({"embedding_dim": dim, "model": model_name})
    # Update the most recent embed run for this docket
    row = db.execute(
        """SELECT run_id FROM pipeline_runs
           WHERE docket_id = ? AND stage = 'embed'
           ORDER BY run_id DESC LIMIT 1""",
        (docket_id,),
    ).fetchone()
    if row:
        db.execute(
            "UPDATE pipeline_runs SET parameters = ? WHERE run_id = ?",
            (params, row[0]),
        )
    else:
        # No existing run (e.g., called outside CLI) — insert one
        db.execute(
            """INSERT INTO pipeline_runs
               (docket_id, stage, status, parameters)
               VALUES (?, 'embed', 'completed', ?)""",
            (docket_id, params),
        )
    db.commit()


def run_embed(db: sqlite3.Connection, docket_id: str, config: dict[str, Any]) -> None:
    """Generate sentence embeddings for unique/representative comments.

    Skips exact duplicates — only embeds comments that are either not in a
    dedup group or are the representative of their group. Processes in batches
    to manage memory.

    Validates that the embedding dimension is consistent with any existing
    embeddings for this docket. If a dimension mismatch is detected (e.g.,
    from changing the model mid-docket), raises an error.

    After embedding, runs semantic dedup (Tier 3) if not already done.

    Args:
        db: SQLite database connection.
        docket_id: The docket ID to process.
        config: Application configuration dictionary.

    Raises:
        ValueError: If the embedding dimension from the current model does not
            match the dimension of existing embeddings for this docket.
    """
    from rich.progress import Progress

    embed_cfg = config.get("embedding", {})
    model_name = embed_cfg.get("model", "all-MiniLM-L6-v2")
    batch_size = embed_cfg.get("batch_size", 64)

    # Find comments that need embeddings: not already embedded, has text,
    # and is either ungrouped or is the representative of a dedup group
    comments = db.execute(
        """SELECT c.comment_id, c.full_text
           FROM comments c
           LEFT JOIN embeddings e ON c.comment_id = e.comment_id
           WHERE c.docket_id = ?
             AND e.comment_id IS NULL
             AND c.full_text IS NOT NULL AND c.full_text != ''
             AND (c.dedup_group_id IS NULL OR c.comment_id IN (
                 SELECT representative_comment_id FROM dedup_groups WHERE docket_id = ?
             ))""",
        (docket_id, docket_id),
    ).fetchall()

    if not comments:
        logger.info("No comments need embedding")
        return

    logger.info("Generating embeddings for %d comments using %s", len(comments), model_name)

    batch_size = max(1, batch_size)  # Guard against misconfigured batch_size <= 0

    # Check for existing embedding dimension before loading the model
    expected_dim = _get_expected_embedding_dim(db, docket_id)

    # Load model
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    # Validate dimension on first batch before committing anything
    dim_validated = False
    actual_dim: int | None = None

    with Progress() as progress:
        task = progress.add_task("Generating embeddings...", total=len(comments))

        for i in range(0, len(comments), batch_size):
            batch = comments[i : i + batch_size]
            comment_ids = [c[0] for c in batch]
            texts = [c[1][:8192] for c in batch]  # Truncate very long texts for embedding

            embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

            # Validate embedding dimension on first batch
            if not dim_validated:
                actual_dim = embeddings.shape[1]
                if expected_dim is not None and actual_dim != expected_dim:
                    raise ValueError(
                        f"Embedding dimension mismatch: expected {expected_dim} but got "
                        f"{actual_dim}. Re-run embed stage with --force to regenerate "
                        f"all embeddings."
                    )
                dim_validated = True

            for cid, emb in zip(comment_ids, embeddings):
                emb_bytes = emb.astype(np.float32).tobytes()
                db.execute(
                    "INSERT OR REPLACE INTO embeddings (comment_id, embedding) VALUES (?, ?)",
                    (cid, emb_bytes),
                )

            db.commit()
            progress.update(task, advance=len(batch))

    # Record the dimension for future validation
    if actual_dim is not None:
        _record_embedding_dim(db, docket_id, actual_dim, model_name)

    logger.info("Embedding generation complete")

    # Run semantic dedup (Tier 3) now that embeddings exist.
    # Clear any existing semantic groups first to avoid duplicates on re-runs.
    from regscope.pipeline.dedup import _semantic_dedup

    db.execute("UPDATE comments SET semantic_group_id = NULL WHERE docket_id = ?", (docket_id,))
    db.execute(
        "DELETE FROM dedup_groups WHERE docket_id = ? AND group_type = 'semantic'",
        (docket_id,),
    )
    db.commit()

    sem_groups = _semantic_dedup(
        db,
        docket_id,
        threshold=config.get("dedup", {}).get("semantic_threshold", 0.92),
    )
    logger.info("Semantic dedup found %d groups", sem_groups)
