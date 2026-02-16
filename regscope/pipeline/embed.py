"""Embedding generation pipeline — generates sentence embeddings for comments."""

import logging
import sqlite3
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def run_embed(db: sqlite3.Connection, docket_id: str, config: dict[str, Any]) -> None:
    """Generate sentence embeddings for unique/representative comments.

    Skips exact duplicates — only embeds comments that are either not in a
    dedup group or are the representative of their group. Processes in batches
    to manage memory.

    After embedding, runs semantic dedup (Tier 3) if not already done.

    Args:
        db: SQLite database connection.
        docket_id: The docket ID to process.
        config: Application configuration dictionary.
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

    # Load model
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    with Progress() as progress:
        task = progress.add_task("Generating embeddings...", total=len(comments))

        for i in range(0, len(comments), batch_size):
            batch = comments[i : i + batch_size]
            comment_ids = [c[0] for c in batch]
            texts = [c[1][:8192] for c in batch]  # Truncate very long texts for embedding

            embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

            for cid, emb in zip(comment_ids, embeddings):
                emb_bytes = emb.astype(np.float32).tobytes()
                db.execute(
                    "INSERT OR REPLACE INTO embeddings (comment_id, embedding) VALUES (?, ?)",
                    (cid, emb_bytes),
                )

            db.commit()
            progress.update(task, advance=len(batch))

    logger.info("Embedding generation complete")

    # Run semantic dedup (Tier 3) now that embeddings exist
    from regscope.pipeline.dedup import _semantic_dedup

    sem_groups = _semantic_dedup(
        db,
        docket_id,
        threshold=config.get("dedup", {}).get("semantic_threshold", 0.92),
    )
    logger.info("Semantic dedup found %d groups", sem_groups)
