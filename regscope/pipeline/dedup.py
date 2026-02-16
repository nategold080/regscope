"""Deduplication pipeline — exact, near-duplicate, and semantic deduplication."""

import logging
import sqlite3
from typing import Any

logger = logging.getLogger(__name__)


def run_dedup(db: sqlite3.Connection, docket_id: str, config: dict[str, Any]) -> None:
    """Run all deduplication tiers on comments for a docket.

    Tier 1: Exact duplicates via SHA-256 hash.
    Tier 2: Near-duplicates via MinHash LSH (Jaccard threshold).
    Tier 3: Semantic duplicates via cosine similarity (runs after embeddings exist).

    Args:
        db: SQLite database connection.
        docket_id: The docket ID to process.
        config: Application configuration dictionary.
    """
    dedup_cfg = config.get("dedup", {})

    logger.info("Running exact dedup (Tier 1) for %s", docket_id)
    exact_groups = _exact_dedup(db, docket_id)
    logger.info("Found %d exact duplicate groups", exact_groups)

    logger.info("Running near-duplicate detection (Tier 2) for %s", docket_id)
    near_groups = _near_dedup(
        db,
        docket_id,
        threshold=dedup_cfg.get("near_duplicate_threshold", 0.85),
        num_perm=dedup_cfg.get("num_perm", 128),
    )
    logger.info("Found %d near-duplicate groups", near_groups)

    # Check if embeddings exist for semantic dedup
    has_embeddings = db.execute(
        """SELECT COUNT(*) FROM embeddings e
           JOIN comments c ON e.comment_id = c.comment_id
           WHERE c.docket_id = ?""",
        (docket_id,),
    ).fetchone()[0]

    if has_embeddings > 0:
        logger.info("Running semantic dedup (Tier 3) for %s", docket_id)
        sem_groups = _semantic_dedup(
            db,
            docket_id,
            threshold=dedup_cfg.get("semantic_threshold", 0.92),
        )
        logger.info("Found %d semantic similarity groups", sem_groups)
    else:
        logger.info("Skipping semantic dedup — no embeddings found. Run embed stage first.")


def _exact_dedup(db: sqlite3.Connection, docket_id: str) -> int:
    """Tier 1: Group comments with identical normalized text hashes.

    Args:
        db: SQLite database connection.
        docket_id: The docket ID to process.

    Returns:
        Number of duplicate groups found.
    """
    from regscope.utils.text import compute_text_hash

    # Compute hashes for all comments
    comments = db.execute(
        "SELECT comment_id, full_text FROM comments WHERE docket_id = ? AND full_text IS NOT NULL AND full_text != ''",
        (docket_id,),
    ).fetchall()

    hash_groups: dict[str, list[str]] = {}
    for comment_id, full_text in comments:
        text_hash = compute_text_hash(full_text)
        db.execute(
            "UPDATE comments SET text_hash = ? WHERE comment_id = ?",
            (text_hash, comment_id),
        )
        hash_groups.setdefault(text_hash, []).append(comment_id)

    db.commit()

    # Create dedup groups for hashes with multiple comments
    group_count = 0
    for text_hash, comment_ids in hash_groups.items():
        if len(comment_ids) < 2:
            continue

        # Find the representative (longest text)
        lengths = db.execute(
            f"SELECT comment_id, LENGTH(full_text) FROM comments WHERE comment_id IN ({','.join('?' * len(comment_ids))})",
            comment_ids,
        ).fetchall()
        representative_id = max(lengths, key=lambda x: x[1] or 0)[0]

        # Get template text
        template = db.execute(
            "SELECT full_text FROM comments WHERE comment_id = ?",
            (representative_id,),
        ).fetchone()[0]

        # Insert dedup group
        cursor = db.execute(
            """INSERT INTO dedup_groups
               (docket_id, group_type, group_size, representative_comment_id, template_text, text_hash)
               VALUES (?, 'exact', ?, ?, ?, ?)""",
            (docket_id, len(comment_ids), representative_id, template[:2000] if template else "", text_hash),
        )
        group_id = cursor.lastrowid

        # Update all comments in this group
        for cid in comment_ids:
            db.execute(
                "UPDATE comments SET dedup_group_id = ? WHERE comment_id = ?",
                (group_id, cid),
            )

        group_count += 1

    db.commit()
    return group_count


def _near_dedup(db: sqlite3.Connection, docket_id: str, threshold: float = 0.85, num_perm: int = 128) -> int:
    """Tier 2: Near-duplicate detection using MinHash LSH.

    Args:
        db: SQLite database connection.
        docket_id: The docket ID to process.
        threshold: Jaccard similarity threshold for near-duplicates.
        num_perm: Number of permutations for MinHash.

    Returns:
        Number of new near-duplicate groups found.
    """
    from datasketch import MinHash, MinHashLSH
    from regscope.utils.text import word_ngrams

    # Only process comments not already in an exact dedup group
    comments = db.execute(
        """SELECT comment_id, full_text FROM comments
           WHERE docket_id = ? AND dedup_group_id IS NULL
           AND full_text IS NOT NULL AND full_text != ''""",
        (docket_id,),
    ).fetchall()

    if len(comments) < 2:
        return 0

    # Build MinHash for each comment
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes: dict[str, MinHash] = {}

    for comment_id, full_text in comments:
        m = MinHash(num_perm=num_perm)
        for ngram in word_ngrams(full_text, n=3):
            m.update(ngram.encode("utf-8"))
        minhashes[comment_id] = m
        try:
            lsh.insert(comment_id, m)
        except ValueError:
            pass  # Duplicate key — already inserted

    # Query LSH for near-duplicates
    visited: set[str] = set()
    group_count = 0

    for comment_id, m in minhashes.items():
        if comment_id in visited:
            continue

        candidates = lsh.query(m)
        if len(candidates) < 2:
            continue

        # Verify actual Jaccard similarity
        group_members = []
        for cand in candidates:
            if cand in visited:
                continue
            if comment_id == cand or minhashes[comment_id].jaccard(minhashes[cand]) >= threshold:
                group_members.append(cand)

        if len(group_members) < 2:
            continue

        # Create the near-dedup group
        lengths = db.execute(
            f"SELECT comment_id, LENGTH(full_text) FROM comments WHERE comment_id IN ({','.join('?' * len(group_members))})",
            group_members,
        ).fetchall()
        representative_id = max(lengths, key=lambda x: x[1] or 0)[0]

        template = db.execute(
            "SELECT full_text FROM comments WHERE comment_id = ?",
            (representative_id,),
        ).fetchone()[0]

        cursor = db.execute(
            """INSERT INTO dedup_groups
               (docket_id, group_type, group_size, representative_comment_id, template_text)
               VALUES (?, 'near', ?, ?, ?)""",
            (docket_id, len(group_members), representative_id, template[:2000] if template else ""),
        )
        group_id = cursor.lastrowid

        for cid in group_members:
            db.execute(
                "UPDATE comments SET dedup_group_id = ? WHERE comment_id = ?",
                (group_id, cid),
            )
            visited.add(cid)

        group_count += 1

    db.commit()
    return group_count


def _semantic_dedup(db: sqlite3.Connection, docket_id: str, threshold: float = 0.92) -> int:
    """Tier 3: Semantic deduplication via cosine similarity on embeddings.

    Creates separate semantic_group_id (does not merge with exact/near groups).

    Args:
        db: SQLite database connection.
        docket_id: The docket ID to process.
        threshold: Cosine similarity threshold.

    Returns:
        Number of semantic similarity groups found.
    """
    import numpy as np

    # Load embeddings for unique/representative comments
    rows = db.execute(
        """SELECT c.comment_id, e.embedding
           FROM comments c
           JOIN embeddings e ON c.comment_id = e.comment_id
           WHERE c.docket_id = ?
           AND (c.dedup_group_id IS NULL OR c.comment_id IN (
               SELECT representative_comment_id FROM dedup_groups WHERE docket_id = ?
           ))""",
        (docket_id, docket_id),
    ).fetchall()

    if len(rows) < 2:
        return 0

    comment_ids = [r[0] for r in rows]
    embeddings = np.array([np.frombuffer(r[1], dtype=np.float32) for r in rows])

    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings_normed = embeddings / norms

    # Use batch processing to avoid all-pairs memory explosion
    visited: set[int] = set()
    group_count = 0
    batch_size = 500

    for i in range(0, len(comment_ids), batch_size):
        batch_end = min(i + batch_size, len(comment_ids))
        batch_sims = embeddings_normed[i:batch_end] @ embeddings_normed.T

        for bi in range(batch_end - i):
            idx = i + bi
            if idx in visited:
                continue

            similar_indices = np.where(batch_sims[bi] >= threshold)[0]
            group_indices = [j for j in similar_indices if j != idx and j not in visited]

            if not group_indices:
                continue

            all_in_group = [idx] + group_indices
            group_cids = [comment_ids[j] for j in all_in_group]

            # Find representative
            lengths = db.execute(
                f"SELECT comment_id, LENGTH(full_text) FROM comments WHERE comment_id IN ({','.join('?' * len(group_cids))})",
                group_cids,
            ).fetchall()
            representative_id = max(lengths, key=lambda x: x[1] or 0)[0]

            cursor = db.execute(
                """INSERT INTO dedup_groups
                   (docket_id, group_type, group_size, representative_comment_id)
                   VALUES (?, 'semantic', ?, ?)""",
                (docket_id, len(group_cids), representative_id),
            )
            sem_group_id = cursor.lastrowid

            for cid in group_cids:
                db.execute(
                    "UPDATE comments SET semantic_group_id = ? WHERE comment_id = ?",
                    (sem_group_id, cid),
                )

            for j in all_in_group:
                visited.add(j)

            group_count += 1

    db.commit()
    return group_count
