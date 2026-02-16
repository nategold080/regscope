"""Topic modeling pipeline — BERTopic clustering of comment embeddings."""

import html
import json
import logging
import re
import sqlite3
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def run_topics(db: sqlite3.Connection, docket_id: str, config: dict[str, Any]) -> None:
    """Run BERTopic topic modeling on unique comment embeddings.

    Only clusters unique comments (dedup-group representatives + ungrouped).
    This prevents form letter campaigns from dominating the topic space.
    After clustering, propagates topic assignments to all group members.

    Automatically adjusts parameters based on the number of unique documents
    to ensure meaningful topic diversity. Uses pre-computed embeddings
    with UMAP dimensionality reduction and HDBSCAN clustering.

    Args:
        db: SQLite database connection.
        docket_id: The docket ID to process.
        config: Application configuration dictionary.
    """
    topic_cfg = config.get("topics", {})

    # Load embeddings for UNIQUE comments only:
    #   - Comments not in any dedup group (ungrouped)
    #   - Representative comments from each dedup group
    rows = db.execute(
        """SELECT c.comment_id, c.full_text, e.embedding
           FROM comments c
           JOIN embeddings e ON c.comment_id = e.comment_id
           WHERE c.docket_id = ?
             AND c.full_text IS NOT NULL AND c.full_text != ''
             AND (
                 c.dedup_group_id IS NULL
                 OR c.comment_id IN (
                     SELECT representative_comment_id
                     FROM dedup_groups WHERE docket_id = ?
                 )
             )""",
        (docket_id, docket_id),
    ).fetchall()

    n_docs = len(rows)

    if n_docs < 5:
        logger.warning("Only %d unique comments with embeddings — too few for topic modeling", n_docs)
        return

    comment_ids = [r[0] for r in rows]
    # Clean HTML from texts before vectorization
    texts = [_clean_for_topics(r[1][:5000]) for r in rows]
    embeddings = np.array([np.frombuffer(r[2], dtype=np.float32) for r in rows])

    logger.info("Running BERTopic on %d unique comments (excluding duplicates)", n_docs)

    # Configure BERTopic components — adapt for docket size
    from umap import UMAP
    from hdbscan import HDBSCAN
    from bertopic import BERTopic
    from sklearn.feature_extraction.text import CountVectorizer

    # Scale parameters based on unique document count
    if n_docs < 30:
        min_cluster = 2
        min_samples = 1
        n_neighbors = min(5, n_docs - 1)
        n_components = min(3, n_docs - 2) if n_docs > 3 else 2
        min_topic_size = 2
    elif n_docs < 100:
        min_cluster = 3
        min_samples = 1
        n_neighbors = min(8, n_docs - 1)
        n_components = min(5, n_docs - 2)
        min_topic_size = 3
    elif n_docs < 500:
        min_cluster = max(3, topic_cfg.get("hdbscan_min_cluster_size", 5))
        min_samples = 2
        n_neighbors = min(12, topic_cfg.get("umap_n_neighbors", 12))
        n_components = topic_cfg.get("umap_n_components", 5)
        min_topic_size = max(3, topic_cfg.get("min_topic_size", 5))
    else:
        min_cluster = topic_cfg.get("hdbscan_min_cluster_size", 10)
        min_samples = topic_cfg.get("hdbscan_min_samples", 5)
        n_neighbors = topic_cfg.get("umap_n_neighbors", 15)
        n_components = topic_cfg.get("umap_n_components", 5)
        min_topic_size = topic_cfg.get("min_topic_size", 10)

    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=topic_cfg.get("umap_min_dist", 0.0),
        metric="cosine",
        random_state=42,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster,
        min_samples=min_samples,
        metric="euclidean",
        prediction_data=True,
    )

    # Use ngram_range (1,2) and stop_words to get better topic representations
    vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2 if n_docs >= 20 else 1,
        max_df=0.90,
    )

    nr_topics = topic_cfg.get("nr_topics", "auto")
    if nr_topics != "auto":
        nr_topics = int(nr_topics)

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        top_n_words=topic_cfg.get("top_n_words", 10),
        nr_topics=nr_topics if nr_topics != "auto" else None,
        min_topic_size=min_topic_size,
        verbose=False,
    )

    # Fit the model
    topics, probs = topic_model.fit_transform(texts, embeddings)

    # If we got very few topics, retry with smaller clusters
    real_topics = [t for t in set(topics) if t != -1]
    if len(real_topics) < 3 and n_docs >= 20:
        logger.info(
            "Only %d topics found for %d unique docs — retrying with smaller clusters",
            len(real_topics), n_docs,
        )
        hdbscan_model = HDBSCAN(
            min_cluster_size=max(2, min_cluster // 2),
            min_samples=1,
            metric="euclidean",
            prediction_data=True,
        )
        topic_model = BERTopic(
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer,
            top_n_words=topic_cfg.get("top_n_words", 10),
            min_topic_size=max(2, min_topic_size // 2),
            verbose=False,
        )
        topics, probs = topic_model.fit_transform(texts, embeddings)

    # Clear previous topic data for this docket
    db.execute(
        """DELETE FROM comment_topics WHERE comment_id IN (
               SELECT comment_id FROM comments WHERE docket_id = ?
           )""",
        (docket_id,),
    )
    db.execute("DELETE FROM topics WHERE docket_id = ?", (docket_id,))

    # Store topics
    topic_info = topic_model.get_topic_info()
    topic_id_map: dict[int, int] = {}  # BERTopic topic_id -> DB topic_id

    for _, row in topic_info.iterrows():
        bt_topic_id = row["Topic"]
        if bt_topic_id == -1:
            label = "Miscellaneous / Outliers"
        else:
            label = row.get("Name", f"Topic {bt_topic_id}")

        # Get keywords for this topic
        topic_words = topic_model.get_topic(bt_topic_id)
        keywords = [w for w, _ in topic_words] if topic_words else []

        # Get representative docs
        try:
            rep_docs = topic_model.get_representative_docs(bt_topic_id)
        except Exception:
            rep_docs = []

        cursor = db.execute(
            """INSERT INTO topics
               (docket_id, bertopic_id, label, keywords, topic_size, representative_texts)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                docket_id,
                bt_topic_id,
                label,
                json.dumps(keywords),
                int(row.get("Count", 0)),
                json.dumps(rep_docs[:3] if rep_docs else []),
            ),
        )
        topic_id_map[bt_topic_id] = cursor.lastrowid

    # Store per-comment topic assignments for the unique comments
    for i, (comment_id, topic_id) in enumerate(zip(comment_ids, topics)):
        db_topic_id = topic_id_map.get(topic_id)
        if db_topic_id is None:
            continue

        prob = float(probs[i]) if probs is not None and i < len(probs) else 0.0

        db.execute(
            """INSERT OR REPLACE INTO comment_topics
               (comment_id, topic_id, relevance_score)
               VALUES (?, ?, ?)""",
            (comment_id, db_topic_id, prob),
        )

    # Propagate topic assignments to all members of each dedup group
    _propagate_topics_to_groups(db, docket_id)

    db.commit()

    real_topic_count = len([t for t in topic_info["Topic"] if t != -1])
    outlier_count = sum(1 for t in topics if t == -1)
    logger.info(
        "Topic modeling complete: %d topics from %d unique docs (%d outliers)",
        real_topic_count, n_docs, outlier_count,
    )


def _propagate_topics_to_groups(db: sqlite3.Connection, docket_id: str) -> None:
    """Copy topic assignments from group representatives to all group members.

    For each dedup group, the representative's topic is assigned to every
    member that doesn't already have one.
    """
    groups = db.execute(
        """SELECT dg.dedup_group_id, dg.representative_comment_id
           FROM dedup_groups dg
           WHERE dg.docket_id = ?""",
        (docket_id,),
    ).fetchall()

    propagated = 0
    for group_id, rep_id in groups:
        # Get the representative's topic assignment
        rep_topic = db.execute(
            "SELECT topic_id, relevance_score FROM comment_topics WHERE comment_id = ?",
            (rep_id,),
        ).fetchone()

        if rep_topic is None:
            continue

        topic_id, relevance = rep_topic

        # Get all other members in this group
        members = db.execute(
            "SELECT comment_id FROM comments WHERE dedup_group_id = ? AND comment_id != ?",
            (group_id, rep_id),
        ).fetchall()

        for (member_id,) in members:
            db.execute(
                """INSERT OR REPLACE INTO comment_topics
                   (comment_id, topic_id, relevance_score)
                   VALUES (?, ?, ?)""",
                (member_id, topic_id, relevance),
            )
            propagated += 1

    if propagated:
        logger.info("Propagated topic assignments to %d duplicate comments", propagated)


def _clean_for_topics(text: str) -> str:
    """Clean HTML tags/entities from text before topic modeling."""
    text = html.unescape(text)
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
