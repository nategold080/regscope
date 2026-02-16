"""Report generation pipeline — Markdown reports and data exports."""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _clean_excerpt(text: str, max_len: int = 500) -> str:
    """Clean text for Markdown excerpts: strip HTML tags/entities, normalize whitespace."""
    import html
    import re

    text = html.unescape(text)
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_len:
        text = text[:max_len] + "..."
    return text


def run_report(db: sqlite3.Connection, docket_id: str, config: dict[str, Any], output_path: str | None = None) -> None:
    """Generate a comprehensive Markdown analysis report.

    Args:
        db: SQLite database connection.
        docket_id: The docket ID to report on.
        config: Application configuration dictionary.
        output_path: Output file path. Defaults to ./output/{docket_id}_report.md.
    """
    report_cfg = config.get("report", {})

    if output_path is None:
        output_path = f"./output/{docket_id}_report.md"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    sections = [
        _docket_overview(db, docket_id),
        _comment_landscape(db, docket_id),
        _topic_analysis(db, docket_id, report_cfg, config),
        _stakeholder_breakdown(db, docket_id),
        _stance_analysis(db, docket_id),
        _substantive_highlights(db, docket_id, report_cfg),
        _data_quality_notes(db, docket_id),
    ]

    report = "\n\n---\n\n".join(sections)
    report = (
        f"# RegScope Analysis Report: {docket_id}\n\n"
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
        f"*This report was generated automatically by [RegScope](https://github.com/regscope), "
        f"a tool for structured analysis of federal rulemaking public comments.*\n\n"
        f"{report}\n"
    )

    with open(output_path, "w") as f:
        f.write(report)

    logger.info("Report saved to %s", output_path)


def run_export(db: sqlite3.Connection, docket_id: str, fmt: str, output_dir: str, config: dict[str, Any]) -> None:
    """Export analysis results to CSV, JSON, or Excel.

    Args:
        db: SQLite database connection.
        docket_id: The docket ID to export.
        fmt: Export format ('csv', 'json', 'excel').
        output_dir: Directory to write output files to.
        config: Application configuration dictionary.
    """
    import pandas as pd

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Build main export query
    df = pd.read_sql_query(
        """SELECT
               c.comment_id,
               c.docket_id,
               c.title,
               c.submitter_name,
               c.organization,
               c.posted_date,
               c.comment_text,
               c.full_text,
               c.text_hash,
               c.dedup_group_id,
               c.semantic_group_id,
               dg.group_type AS dedup_type,
               dg.group_size AS dedup_group_size,
               cc.stakeholder_type,
               cc.stance,
               cc.stance_confidence,
               cc.substantiveness_score,
               t.label AS topic_label,
               ct.relevance_score AS topic_relevance
           FROM comments c
           LEFT JOIN dedup_groups dg ON c.dedup_group_id = dg.dedup_group_id
           LEFT JOIN comment_classifications cc ON c.comment_id = cc.comment_id
           LEFT JOIN comment_topics ct ON c.comment_id = ct.comment_id
           LEFT JOIN topics t ON ct.topic_id = t.topic_id
           WHERE c.docket_id = ?
           ORDER BY c.posted_date""",
        db,
        params=(docket_id,),
    )

    base_path = Path(output_dir) / docket_id

    if fmt == "csv":
        path = f"{base_path}_comments.csv"
        df.to_csv(path, index=False)
        logger.info("Exported %d rows to %s", len(df), path)

    elif fmt == "json":
        path = f"{base_path}_comments.json"
        # Build structured JSON
        output = _build_json_export(db, docket_id, df)
        with open(path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        logger.info("Exported to %s", path)

    elif fmt == "excel":
        path = f"{base_path}_comments.xlsx"
        df.to_excel(path, index=False, engine="openpyxl")
        logger.info("Exported %d rows to %s", len(df), path)


def _build_json_export(db: sqlite3.Connection, docket_id: str, df: "pd.DataFrame") -> dict:
    """Build structured JSON export with nested topics and statistics."""
    topics = db.execute(
        "SELECT topic_id, label, keywords, topic_size FROM topics WHERE docket_id = ?",
        (docket_id,),
    ).fetchall()

    return {
        "docket_id": docket_id,
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "total_comments": len(df),
            "unique_comments": int(df["dedup_group_id"].isna().sum() + df.drop_duplicates("dedup_group_id")["dedup_group_id"].notna().sum()),
        },
        "topics": [
            {
                "topic_id": t[0],
                "label": t[1],
                "keywords": json.loads(t[2]) if t[2] else [],
                "size": t[3],
            }
            for t in topics
        ],
        "comments": json.loads(df.to_json(orient="records", default_handler=str)),
    }


def _docket_overview(db: sqlite3.Connection, docket_id: str) -> str:
    """Generate the Docket Overview section."""
    docket = db.execute(
        "SELECT title, agency, docket_type, modified_date FROM dockets WHERE docket_id = ?",
        (docket_id,),
    ).fetchone()

    total_comments = db.execute(
        "SELECT COUNT(*) FROM comments WHERE docket_id = ?", (docket_id,)
    ).fetchone()[0]

    date_range = db.execute(
        "SELECT MIN(posted_date), MAX(posted_date) FROM comments WHERE docket_id = ?",
        (docket_id,),
    ).fetchone()

    title = docket[0] if docket else "Unknown"
    agency = docket[1] if docket else "Unknown"

    # Format date range nicely
    def _fmt_date(d: str | None) -> str:
        if not d:
            return "N/A"
        return d[:10]  # Extract YYYY-MM-DD from ISO format

    min_date = _fmt_date(date_range[0])
    max_date = _fmt_date(date_range[1])
    date_str = min_date if min_date == max_date else f"{min_date} to {max_date}"

    lines = [
        "## 1. Docket Overview",
        "",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| **Docket ID** | {docket_id} |",
        f"| **Title** | {title} |",
        f"| **Agency** | {agency} |",
        f"| **Total Comments** | {total_comments:,} |",
        f"| **Comment Period** | {date_str} |",
    ]

    return "\n".join(lines)


def _comment_landscape(db: sqlite3.Connection, docket_id: str) -> str:
    """Generate the Comment Landscape section."""
    total = db.execute(
        "SELECT COUNT(*) FROM comments WHERE docket_id = ?", (docket_id,)
    ).fetchone()[0]

    unique_no_group = db.execute(
        "SELECT COUNT(*) FROM comments WHERE docket_id = ? AND dedup_group_id IS NULL",
        (docket_id,),
    ).fetchone()[0]

    dedup_groups = db.execute(
        "SELECT COUNT(*), SUM(group_size) FROM dedup_groups WHERE docket_id = ? AND group_type IN ('exact', 'near')",
        (docket_id,),
    ).fetchall()[0]

    form_letters = db.execute(
        """SELECT dg.dedup_group_id, dg.group_size, dg.template_text, dg.group_type
           FROM dedup_groups dg
           WHERE dg.docket_id = ? AND dg.group_type IN ('exact', 'near') AND dg.group_size >= 5
           ORDER BY dg.group_size DESC
           LIMIT 10""",
        (docket_id,),
    ).fetchall()

    num_groups = dedup_groups[0] or 0
    num_in_groups = dedup_groups[1] or 0

    lines = [
        "## 2. Comment Landscape",
        "",
        f"| Metric | Count |",
        f"|--------|-------|",
        f"| Total comments | {total:,} |",
        f"| Unique comments (no duplicates) | {unique_no_group:,} |",
        f"| Duplicate groups | {num_groups:,} |",
        f"| Comments in duplicate groups | {num_in_groups:,} |",
        "",
    ]

    if form_letters:
        lines.append("### Form Letter Campaigns")
        lines.append("")
        for group_id, size, template, gtype in form_letters:
            excerpt = _clean_excerpt(template or "", 300)
            lines.append(f"**Campaign ({gtype}, {size:,} copies):**")
            lines.append(f"> {excerpt}")
            lines.append("")

    return "\n".join(lines)


def _readable_topic_label(label: str, keywords_json: str | None) -> str:
    """Convert a raw BERTopic label into a readable display name.

    Takes top 3 unique keywords, title-cases them, joins with " / ".
    E.g., "0_project help_help_communities_pollution" → "Project Help / Communities / Pollution"
    """
    kw = json.loads(keywords_json) if keywords_json else []
    if not kw:
        return label

    # Deduplicate keywords (BERTopic often repeats with/without bigrams)
    seen: set[str] = set()
    unique_kw: list[str] = []
    for k in kw:
        # Skip if this keyword is a substring of one already chosen
        k_lower = k.lower().strip()
        if k_lower in seen or any(k_lower in s for s in seen):
            continue
        # Skip if an existing keyword is a substring of this one
        seen_copy = set(seen)
        for s in seen_copy:
            if s in k_lower:
                seen.discard(s)
                unique_kw = [u for u in unique_kw if u.lower().strip() != s]
        seen.add(k_lower)
        unique_kw.append(k.strip())
        if len(unique_kw) >= 3:
            break

    if not unique_kw:
        return label

    _COMMON_WORDS = {"new", "old", "the", "and", "for", "all", "but", "not", "our",
                      "get", "set", "use", "how", "who", "why", "can", "may", "too",
                      "big", "low", "bad", "add", "run", "cut", "put", "top", "end",
                      "try", "let", "say", "see", "now", "out", "off", "has", "had",
                      "was", "did", "got", "per", "due", "via", "yet", "any", "few",
                      "own", "ask", "red", "go", "no", "so", "do", "up", "if", "or",
                      "dead", "stop", "just", "look", "plan", "farm", "near", "going"}
    _KNOWN_ACRONYMS = {"boem", "epa", "noaa", "faa", "nps", "eis", "osw", "lbi",
                        "nhtsa", "fhwa", "dot", "doj", "dhs", "uscg", "cfr"}

    def _format_kw(w: str) -> str:
        """Title-case a keyword, but uppercase known acronyms."""
        w = w.strip()
        if w.lower() in _KNOWN_ACRONYMS:
            return w.upper()
        if len(w) <= 3 and w.isalpha() and w.lower() not in _COMMON_WORDS:
            return w.upper()
        return w.title()

    return " / ".join(_format_kw(w) for w in unique_kw)


def _generate_llm_topic_labels(
    db: sqlite3.Connection,
    docket_id: str,
    topics_data: list[tuple],
    config: dict[str, Any],
) -> dict[str, str]:
    """Generate human-readable topic labels using an LLM (single batch call).

    Checks for cached llm_label values in the DB first. Only calls the API
    for topics missing labels. Falls back gracefully if openai is not installed,
    OPENAI_API_KEY is not set, or the API call fails.

    Args:
        db: SQLite database connection.
        docket_id: The docket ID.
        topics_data: List of (topic_id, label, keywords_json, topic_size, rep_texts_json) tuples.
        config: Application configuration dictionary.

    Returns:
        Dict mapping raw label string to LLM-generated label (or empty dict on failure).
    """
    import os

    llm_cfg = config.get("llm", {})
    if not llm_cfg.get("enabled", True):
        logger.debug("LLM topic labels disabled in config")
        return {}

    if not os.environ.get("OPENAI_API_KEY"):
        logger.debug("OPENAI_API_KEY not set, skipping LLM topic labels")
        return {}

    try:
        from openai import OpenAI
    except ImportError:
        logger.debug("openai package not installed, skipping LLM topic labels")
        return {}

    # Check which topics already have cached LLM labels
    cached = db.execute(
        "SELECT label, llm_label FROM topics WHERE docket_id = ? AND llm_label IS NOT NULL",
        (docket_id,),
    ).fetchall()
    label_map = {row[0]: row[1] for row in cached}

    # Filter to topics that need labels
    needs_labels = [t for t in topics_data if t[1] not in label_map]
    if not needs_labels:
        logger.debug("All topic labels already cached")
        return label_map

    # Build prompt with all unlabeled topics
    topic_descriptions = []
    for topic_id, label, keywords_json, size, rep_texts_json in needs_labels:
        kw = json.loads(keywords_json) if keywords_json else []
        rep_texts = json.loads(rep_texts_json) if rep_texts_json else []
        excerpt = ""
        if rep_texts:
            excerpt = _clean_excerpt(rep_texts[0], 150)

        topic_descriptions.append(
            f"- Topic {topic_id} ({size} comments): keywords=[{', '.join(kw[:8])}]"
            + (f' example="{excerpt}"' if excerpt else "")
        )

    prompt = (
        "You are labeling topics from a public comment analysis on a federal rulemaking.\n"
        "For each topic below, produce a short (3-7 word) descriptive label that a policy analyst would understand.\n"
        "Make each label distinct from the others. Use plain language, not jargon.\n\n"
        + "\n".join(topic_descriptions)
        + "\n\nRespond with ONLY a JSON object mapping each topic ID (as a string) to its label. "
        "Example: {\"1\": \"Environmental Impact Concerns\", \"2\": \"Cost Burden on Small Business\"}"
    )

    model = llm_cfg.get("model", "gpt-4o-mini")

    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        raw_response = response.choices[0].message.content
        llm_labels = json.loads(raw_response)
    except Exception as e:
        logger.warning("LLM topic label generation failed: %s", e)
        return label_map

    # Store generated labels in DB and build return map
    for topic_id, label, keywords_json, size, rep_texts_json in needs_labels:
        llm_label = llm_labels.get(str(topic_id))
        if llm_label:
            label_map[label] = llm_label
            db.execute(
                "UPDATE topics SET llm_label = ? WHERE docket_id = ? AND topic_id = ?",
                (llm_label, docket_id, topic_id),
            )

    db.commit()
    logger.info("Generated LLM labels for %d topics", len(llm_labels))
    return label_map


def _topic_analysis(db: sqlite3.Connection, docket_id: str, report_cfg: dict, config: dict[str, Any] | None = None) -> str:
    """Generate the Topic Analysis section."""
    topics = db.execute(
        """SELECT t.topic_id, t.label, t.keywords, t.topic_size, t.representative_texts
           FROM topics t
           WHERE t.docket_id = ? AND t.bertopic_id != -1
           ORDER BY t.topic_size DESC""",
        (docket_id,),
    ).fetchall()

    lines = [
        "## 3. Topic Analysis",
        "",
    ]

    if not topics:
        lines.append("*No topics identified (insufficient data or topic modeling not run).*")
        return "\n".join(lines)

    # Try to get LLM-generated labels
    llm_labels: dict[str, str] = {}
    if config is not None:
        llm_labels = _generate_llm_topic_labels(db, docket_id, topics, config)

    def _get_label(label: str, keywords_json: str | None) -> str:
        """Get LLM label if available, otherwise fall back to keyword label."""
        if label in llm_labels:
            return llm_labels[label]
        return _readable_topic_label(label, keywords_json)

    lines.extend([
        "| # | Topic | Keywords | Comments |",
        "|---|-------|----------|----------|",
    ])

    for i, (topic_id, label, keywords_json, size, rep_texts_json) in enumerate(topics, 1):
        kw = json.loads(keywords_json) if keywords_json else []
        display_label = _get_label(label, keywords_json)
        kw_str = ", ".join(kw[:5])
        lines.append(f"| {i} | {display_label} | {kw_str} | {size:,} |")

    # Add representative quotes for top topics
    lines.append("")
    lines.append("### Representative Quotes")
    lines.append("")

    max_excerpt = report_cfg.get("max_excerpt_length", 500)
    for topic_id, label, keywords_json, size, rep_texts_json in topics[:5]:
        display_label = _get_label(label, keywords_json)
        rep_texts = json.loads(rep_texts_json) if rep_texts_json else []
        if rep_texts:
            lines.append(f"**{display_label}** ({size:,} comments):")
            for rt in rep_texts[:2]:
                excerpt = _clean_excerpt(rt, max_excerpt)
                lines.append(f"> {excerpt}")
                lines.append("")

    return "\n".join(lines)


def _stakeholder_breakdown(db: sqlite3.Connection, docket_id: str) -> str:
    """Generate the Stakeholder Breakdown section."""
    rows = db.execute(
        """SELECT cc.stakeholder_type, COUNT(*) as cnt
           FROM comment_classifications cc
           JOIN comments c ON cc.comment_id = c.comment_id
           WHERE c.docket_id = ?
           GROUP BY cc.stakeholder_type
           ORDER BY cnt DESC""",
        (docket_id,),
    ).fetchall()

    total = sum(r[1] for r in rows) if rows else 0

    lines = [
        "## 4. Stakeholder Breakdown",
        "",
    ]

    if not rows:
        lines.append("*No stakeholder classifications available.*")
        return "\n".join(lines)

    lines.extend([
        "| Stakeholder Type | Count | Percentage |",
        "|------------------|-------|------------|",
    ])

    for stype, count in rows:
        pct = (count / total * 100) if total > 0 else 0
        lines.append(f"| {stype or 'unknown'} | {count:,} | {pct:.1f}% |")

    # Top organizations
    top_orgs = db.execute(
        """SELECT organization, COUNT(*) as cnt
           FROM comments
           WHERE docket_id = ? AND organization IS NOT NULL AND organization != ''
           GROUP BY organization
           ORDER BY cnt DESC
           LIMIT 15""",
        (docket_id,),
    ).fetchall()

    if top_orgs:
        lines.extend([
            "",
            "### Top Commenting Organizations",
            "",
            "| Organization | Comments |",
            "|-------------|----------|",
        ])
        for org, count in top_orgs:
            lines.append(f"| {org} | {count:,} |")

    return "\n".join(lines)


def _stance_analysis(db: sqlite3.Connection, docket_id: str) -> str:
    """Generate the Stance Analysis section."""
    # Overall stance
    overall = db.execute(
        """SELECT cc.stance, COUNT(*) as cnt
           FROM comment_classifications cc
           JOIN comments c ON cc.comment_id = c.comment_id
           WHERE c.docket_id = ? AND cc.stance IS NOT NULL
           GROUP BY cc.stance
           ORDER BY cnt DESC""",
        (docket_id,),
    ).fetchall()

    total = sum(r[1] for r in overall) if overall else 0

    lines = [
        "## 5. Stance Analysis",
        "",
    ]

    if not overall:
        lines.append("*No stance analysis available.*")
        return "\n".join(lines)

    lines.extend([
        "### Overall Stance Distribution",
        "",
        "| Stance | Count | Percentage |",
        "|--------|-------|------------|",
    ])

    for stance, count in overall:
        pct = (count / total * 100) if total > 0 else 0
        lines.append(f"| {stance} | {count:,} | {pct:.1f}% |")

    # Stance by stakeholder type
    cross_tab = db.execute(
        """SELECT cc.stakeholder_type, cc.stance, COUNT(*) as cnt
           FROM comment_classifications cc
           JOIN comments c ON cc.comment_id = c.comment_id
           WHERE c.docket_id = ? AND cc.stance IS NOT NULL AND cc.stakeholder_type IS NOT NULL
           GROUP BY cc.stakeholder_type, cc.stance
           ORDER BY cc.stakeholder_type, cnt DESC""",
        (docket_id,),
    ).fetchall()

    if cross_tab:
        lines.extend([
            "",
            "### Stance by Stakeholder Type",
            "",
        ])

        # Pivot the data
        pivot: dict[str, dict[str, int]] = {}
        all_stances: set[str] = set()
        for stype, stance, count in cross_tab:
            pivot.setdefault(stype, {})[stance] = count
            all_stances.add(stance)

        stance_order = ["support", "oppose", "conditional_support", "conditional_oppose", "neutral_informational", "unclear"]
        ordered_stances = [s for s in stance_order if s in all_stances]

        header = "| Stakeholder | " + " | ".join(ordered_stances) + " |"
        sep = "|" + "|".join(["---"] * (len(ordered_stances) + 1)) + "|"
        lines.extend([header, sep])

        for stype in sorted(pivot.keys()):
            row = f"| {stype} | "
            row += " | ".join(str(pivot[stype].get(s, 0)) for s in ordered_stances)
            row += " |"
            lines.append(row)

    return "\n".join(lines)


def _substantive_highlights(db: sqlite3.Connection, docket_id: str, report_cfg: dict) -> str:
    """Generate the Substantive Comment Highlights section."""
    top_n = report_cfg.get("top_substantive_count", 20)
    max_excerpt = report_cfg.get("max_excerpt_length", 500)

    top_comments = db.execute(
        """SELECT c.comment_id, c.submitter_name, c.organization, c.full_text,
                  cc.substantiveness_score, cc.stakeholder_type, cc.stance
           FROM comments c
           JOIN comment_classifications cc ON c.comment_id = cc.comment_id
           WHERE c.docket_id = ? AND cc.substantiveness_score IS NOT NULL
           ORDER BY cc.substantiveness_score DESC
           LIMIT ?""",
        (docket_id, top_n),
    ).fetchall()

    # Score distribution
    score_dist = db.execute(
        """SELECT
           SUM(CASE WHEN cc.substantiveness_score >= 80 THEN 1 ELSE 0 END),
           SUM(CASE WHEN cc.substantiveness_score >= 60 AND cc.substantiveness_score < 80 THEN 1 ELSE 0 END),
           SUM(CASE WHEN cc.substantiveness_score >= 40 AND cc.substantiveness_score < 60 THEN 1 ELSE 0 END),
           SUM(CASE WHEN cc.substantiveness_score >= 20 AND cc.substantiveness_score < 40 THEN 1 ELSE 0 END),
           SUM(CASE WHEN cc.substantiveness_score < 20 THEN 1 ELSE 0 END),
           COUNT(*)
           FROM comment_classifications cc
           JOIN comments c ON cc.comment_id = c.comment_id
           WHERE c.docket_id = ? AND cc.substantiveness_score IS NOT NULL""",
        (docket_id,),
    ).fetchone()

    lines = [
        "## 6. Substantive Comment Highlights",
        "",
    ]

    if not top_comments:
        lines.append("*No substantiveness scores available.*")
        return "\n".join(lines)

    if score_dist and score_dist[5] > 0:
        lines.extend([
            "### Score Distribution",
            "",
            "| Range | Count | Description |",
            "|-------|-------|-------------|",
            f"| 80-100 | {score_dist[0] or 0} | Highly substantive — detailed analysis with citations |",
            f"| 60-79 | {score_dist[1] or 0} | Substantive — specific arguments and evidence |",
            f"| 40-59 | {score_dist[2] or 0} | Moderate — some substance, general arguments |",
            f"| 20-39 | {score_dist[3] or 0} | Low — brief or general comments |",
            f"| 0-19 | {score_dist[4] or 0} | Minimal — form letters or very short |",
            "",
        ])

    lines.extend([
        f"### Top {top_n} Comments",
        "",
    ])

    for i, (cid, name, org, text, score, stype, stance) in enumerate(top_comments, 1):
        submitter = name or "Anonymous"
        if org:
            submitter = f"{submitter} ({org})"

        excerpt = _clean_excerpt(text or "", max_excerpt)

        lines.extend([
            f"### {i}. Score: {score}/100 — {submitter}",
            f"*Stakeholder: {stype or 'unknown'} | Stance: {stance or 'unknown'}*",
            "",
            f"> {excerpt}",
            "",
        ])

    return "\n".join(lines)


def _data_quality_notes(db: sqlite3.Connection, docket_id: str) -> str:
    """Generate the Data Quality Notes section."""
    total = db.execute(
        "SELECT COUNT(*) FROM comments WHERE docket_id = ?", (docket_id,)
    ).fetchone()[0]

    no_text = db.execute(
        "SELECT COUNT(*) FROM comments WHERE docket_id = ? AND (full_text IS NULL OR full_text = '')",
        (docket_id,),
    ).fetchone()[0]

    no_org = db.execute(
        "SELECT COUNT(*) FROM comments WHERE docket_id = ? AND (organization IS NULL OR organization = '')",
        (docket_id,),
    ).fetchone()[0]

    total_attachments = db.execute(
        """SELECT COUNT(*) FROM attachments a
           JOIN comments c ON a.comment_id = c.comment_id
           WHERE c.docket_id = ?""",
        (docket_id,),
    ).fetchone()[0]

    failed_extractions = db.execute(
        """SELECT COUNT(*) FROM attachments a
           JOIN comments c ON a.comment_id = c.comment_id
           WHERE c.docket_id = ? AND a.extracted_text IS NULL AND a.file_url != ''""",
        (docket_id,),
    ).fetchone()[0]

    # Count comments where the body was a stub ("see attached") but
    # full_text was populated from attachment extraction
    pdf_only = db.execute(
        """SELECT COUNT(DISTINCT c.comment_id) FROM comments c
           JOIN attachments a ON c.comment_id = a.comment_id
           WHERE c.docket_id = ?
           AND a.extracted_text IS NOT NULL AND a.extracted_text != ''
           AND (c.comment_text IS NULL OR c.comment_text = ''
                OR LENGTH(c.comment_text) < 200)""",
        (docket_id,),
    ).fetchone()[0]

    successfully_extracted = db.execute(
        """SELECT COUNT(*) FROM attachments a
           JOIN comments c ON a.comment_id = c.comment_id
           WHERE c.docket_id = ? AND a.extracted_text IS NOT NULL AND a.extracted_text != ''""",
        (docket_id,),
    ).fetchone()[0]

    lines = [
        "## 7. Data Quality Notes",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total comments | {total:,} |",
    ]

    if total > 0:
        lines.append(f"| Comments with no text | {no_text:,} ({no_text / total * 100:.1f}%) |")
        lines.append(f"| Comments without organization | {no_org:,} ({no_org / total * 100:.1f}%) |")
    else:
        lines.append(f"| Comments with no text | {no_text:,} |")
        lines.append(f"| Comments without organization | {no_org:,} |")

    lines.extend([
        f"| Total attachments | {total_attachments:,} |",
        f"| Successfully extracted | {successfully_extracted:,} |",
        f"| Failed extractions | {failed_extractions:,} |",
        f"| Attachment-dependent comments | {pdf_only:,} |",
    ])

    return "\n".join(lines)
