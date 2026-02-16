"""Classification pipeline — stakeholder type, stance detection, substantiveness scoring."""

import json
import logging
import math
import re
import sqlite3
from typing import Any

logger = logging.getLogger(__name__)

# Stakeholder classification patterns (checked in order — first match wins)
STAKEHOLDER_PATTERNS: dict[str, list[str]] = {
    "academic": [
        "university", "college", "school of", "professor", "dr.",
        "ph.d", "phd", "laboratory", "lab ", "labs ",
        "academic", "faculty", "institute of technology",
        "research center", "research institute", "ucla", "mit ",
    ],
    "government": [
        "department of", "agency", "commission", "state of", "county of",
        "city of", "town of", "village of", "bureau", "office of",
        "tribal", "municipality", "public utility", "authority",
        "commonwealth", "district", "administration", "resources board",
        "board of supervisors", "public health", "air resources",
        "water board", "planning commission", "u.s. ", "federal",
        "national highway", "nhtsa", "dot ", "faa ", "fhwa",
        "township", "borough", "parish", "county",
        "national park service", "park service", "fish and wildlife",
        "forest service", "geological survey", "coast guard",
        "army corps", "corps of engineers", "reclamation",
        "environmental protection", "epa ",
    ],
    "law_firm": [
        "law firm", "attorneys", "lawyers", "legal", "llp", "l.l.p.",
        "pllc", "law office", "counsel", "esquire", "esq.",
        "barristers", "solicitors",
    ],
    "trade_association": [
        "association", "council", "federation", "chamber", "alliance",
        "institute", "society of", "board of trade", "coalition", "league",
        "conference of", "congress of", "national association",
    ],
    "nonprofit": [
        "foundation", "fund", "trust", "center for", "centre for",
        "project", "initiative", "network", "action", "defense fund",
        "conservation", "environmental", "sierra club", "nrdc",
        "earthjustice", "advocacy", "watch", "policy", "501(c",
        "501c", "non-profit", "nonprofit", "not-for-profit",
        "boathouse", "conservancy", "community",
        "defenders", "defend ", "save ", "faith",
        "citizens", "friends of", "protect", "humane",
        "greenpeace", "audubon", "surfrider", "clean ocean",
    ],
    "industry": [
        "inc.", "inc,", "incorporated", "corp.", "corp,", "corporation",
        "llc", "l.l.c.", "company", "co.", "ltd", "limited",
        "enterprises", "industries", "group", "holdings", "partners",
        "solutions", "services", "technologies", "energy", "power",
        "utility", "utilities", "robotics", "analytics", "systems",
        "consulting", " ai", "software", "charters",
    ],
}

# Stance detection candidate labels
STANCE_LABELS = [
    "supports the proposed rule",
    "opposes the proposed rule",
    "supports with modifications",
    "opposes but suggests changes",
    "provides information without taking a position",
]

STANCE_LABEL_MAP = {
    "supports the proposed rule": "support",
    "opposes the proposed rule": "oppose",
    "supports with modifications": "conditional_support",
    "opposes but suggests changes": "conditional_oppose",
    "provides information without taking a position": "neutral_informational",
}


def run_classify(db: sqlite3.Connection, docket_id: str, config: dict[str, Any]) -> None:
    """Run all classification tasks: stakeholder type, stance, substantiveness.

    Args:
        db: SQLite database connection.
        docket_id: The docket ID to process.
        config: Application configuration dictionary.
    """
    logger.info("Running stakeholder classification for %s", docket_id)
    _classify_stakeholders(db, docket_id, config)

    logger.info("Running stance detection for %s", docket_id)
    _detect_stance(db, docket_id, config)

    logger.info("Running substantiveness scoring for %s", docket_id)
    _score_substantiveness(db, docket_id, config)


def _classify_stakeholders(db: sqlite3.Connection, docket_id: str, config: dict[str, Any]) -> None:
    """Classify each comment's source into stakeholder categories.

    Primary method: rule-based matching on organization field.
    Fallback for no-org comments: check if submitter name looks like an
    individual (first-name-only or first+last with no org → individual).

    Args:
        db: SQLite database connection.
        docket_id: The docket ID to process.
        config: Application configuration dictionary.
    """
    comments = db.execute(
        """SELECT comment_id, organization, full_text, submitter_name FROM comments
           WHERE docket_id = ?
           AND comment_id NOT IN (
               SELECT comment_id FROM comment_classifications WHERE stakeholder_type IS NOT NULL
           )""",
        (docket_id,),
    ).fetchall()

    if not comments:
        logger.info("All comments already classified")
        return

    for comment_id, org, full_text, submitter_name in comments:
        stakeholder_type = _classify_org(org)

        if stakeholder_type == "unknown":
            if not org or not org.strip():
                # No organization → individual
                stakeholder_type = "individual"
            # If org was set but didn't match any pattern and the heuristic
            # returned "industry", keep that result

        db.execute(
            """INSERT OR REPLACE INTO comment_classifications
               (comment_id, stakeholder_type)
               VALUES (?, ?)
               ON CONFLICT(comment_id) DO UPDATE SET stakeholder_type = ?""",
            (comment_id, stakeholder_type, stakeholder_type),
        )

    db.commit()


def _classify_org(org: str | None) -> str:
    """Classify an organization name into a stakeholder category.

    Uses pattern matching on the organization name. Checks patterns in
    priority order so academic/government/law_firm match before broader
    industry patterns.

    Args:
        org: Organization name from the comment.

    Returns:
        Stakeholder type string.
    """
    if not org or not org.strip():
        return "unknown"

    org_lower = org.lower().strip()

    for category, patterns in STAKEHOLDER_PATTERNS.items():
        for pattern in patterns:
            if pattern in org_lower:
                return category

    # Heuristic: if org name is a capitalized multi-word name that doesn't
    # match any pattern, it's likely a company (e.g., "Arcadis", "Merlin",
    # "Hayden AI"). Short single-word orgs with no pattern are assumed industry.
    words = org.strip().split()
    if len(words) >= 1 and words[0][0].isupper():
        return "industry"

    return "unknown"


def _detect_stance(db: sqlite3.Connection, docket_id: str, config: dict[str, Any]) -> None:
    """Detect stance (support/oppose/etc.) for unique comments.

    Uses zero-shot classification with facebook/bart-large-mnli model.
    Only processes unique/representative comments (not duplicates).

    Args:
        db: SQLite database connection.
        docket_id: The docket ID to process.
        config: Application configuration dictionary.
    """
    class_cfg = config.get("classification", {})
    confidence_threshold = class_cfg.get("stance_confidence_threshold", 0.4)

    # Get unique comments that need stance detection
    comments = db.execute(
        """SELECT c.comment_id, c.full_text FROM comments c
           LEFT JOIN comment_classifications cc ON c.comment_id = cc.comment_id
           WHERE c.docket_id = ?
             AND c.full_text IS NOT NULL AND c.full_text != ''
             AND (cc.stance IS NULL)
             AND (c.dedup_group_id IS NULL OR c.comment_id IN (
                 SELECT representative_comment_id FROM dedup_groups WHERE docket_id = ?
             ))""",
        (docket_id, docket_id),
    ).fetchall()

    if not comments:
        logger.info("All unique comments already have stance detection")
        return

    logger.info("Running stance detection on %d unique comments", len(comments))

    # Load zero-shot classifier
    from transformers import pipeline as hf_pipeline
    from rich.progress import Progress

    model_name = class_cfg.get("stance_model", "facebook/bart-large-mnli")
    try:
        classifier = hf_pipeline("zero-shot-classification", model=model_name)
    except Exception:
        logger.exception("Failed to load stance detection model %s", model_name)
        return

    with Progress() as progress:
        task = progress.add_task("Detecting stance...", total=len(comments))

        for comment_id, full_text in comments:
            try:
                # Clean HTML entities/tags before classification
                import html as html_mod
                text = html_mod.unescape(full_text)
                text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
                text = re.sub(r"<[^>]+>", "", text)
                text = re.sub(r"\s+", " ", text).strip()
                text = text[:1024]

                result = classifier(text, STANCE_LABELS, multi_label=False)

                # Build a label→score map
                score_map = dict(zip(result["labels"], result["scores"]))

                # Aggregate directional scores: support vs oppose vs neutral
                support_score = (
                    score_map.get("supports the proposed rule", 0)
                    + score_map.get("supports with modifications", 0)
                )
                oppose_score = (
                    score_map.get("opposes the proposed rule", 0)
                    + score_map.get("opposes but suggests changes", 0)
                )
                neutral_score = score_map.get(
                    "provides information without taking a position", 0
                )

                top_label = result["labels"][0]
                top_score = result["scores"][0]

                # Use directional aggregation: if support or oppose direction
                # clearly dominates, pick the specific stance within that direction
                direction_threshold = 0.35
                if support_score >= direction_threshold and support_score > oppose_score * 1.5:
                    # Support direction — pick between full and conditional
                    if score_map.get("supports with modifications", 0) > score_map.get("supports the proposed rule", 0):
                        stance = "conditional_support"
                    else:
                        stance = "support"
                    top_score = support_score
                elif oppose_score >= direction_threshold and oppose_score > support_score * 1.5:
                    # Oppose direction
                    if score_map.get("opposes but suggests changes", 0) > score_map.get("opposes the proposed rule", 0):
                        stance = "conditional_oppose"
                    else:
                        stance = "oppose"
                    top_score = oppose_score
                elif top_score >= confidence_threshold:
                    stance = STANCE_LABEL_MAP.get(top_label, "unclear")
                else:
                    stance = "unclear"

                db.execute(
                    """INSERT OR REPLACE INTO comment_classifications
                       (comment_id, stance, stance_confidence)
                       VALUES (?, ?, ?)
                       ON CONFLICT(comment_id) DO UPDATE SET stance = ?, stance_confidence = ?""",
                    (comment_id, stance, top_score, stance, top_score),
                )

                # Propagate stance to all comments in the same dedup group
                group = db.execute(
                    "SELECT dedup_group_id FROM comments WHERE comment_id = ?",
                    (comment_id,),
                ).fetchone()
                if group and group[0]:
                    members = db.execute(
                        "SELECT comment_id FROM comments WHERE dedup_group_id = ? AND comment_id != ?",
                        (group[0], comment_id),
                    ).fetchall()
                    for (member_id,) in members:
                        db.execute(
                            """INSERT OR REPLACE INTO comment_classifications
                               (comment_id, stance, stance_confidence)
                               VALUES (?, ?, ?)
                               ON CONFLICT(comment_id) DO UPDATE SET stance = ?, stance_confidence = ?""",
                            (member_id, stance, top_score, stance, top_score),
                        )

            except Exception:
                logger.exception("Stance detection failed for %s", comment_id)

            progress.update(task, advance=1)

            if comment_id and int(progress.tasks[task].completed) % 10 == 0:
                db.commit()

    db.commit()


def _score_substantiveness(db: sqlite3.Connection, docket_id: str, config: dict[str, Any]) -> None:
    """Score each comment 0-100 on substantiveness using weighted heuristics.

    Factors: text length, citations, section references, technical vocabulary,
    data/statistics, legal arguments, form letter status, organizational
    affiliation, and unique comment status.

    Args:
        db: SQLite database connection.
        docket_id: The docket ID to process.
        config: Application configuration dictionary.
    """
    sub_cfg = config.get("substantiveness", {})

    comments = db.execute(
        """SELECT c.comment_id, c.full_text, c.dedup_group_id, dg.group_size,
                  c.organization, cc.stakeholder_type
           FROM comments c
           LEFT JOIN dedup_groups dg ON c.dedup_group_id = dg.dedup_group_id
           LEFT JOIN comment_classifications cc ON c.comment_id = cc.comment_id
           WHERE c.docket_id = ?
             AND (cc.substantiveness_score IS NULL)
             AND c.full_text IS NOT NULL AND c.full_text != ''""",
        (docket_id,),
    ).fetchall()

    if not comments:
        logger.info("All comments already scored")
        return

    for comment_id, full_text, dedup_group_id, group_size, org, stakeholder_type in comments:
        is_form_letter = (group_size or 0) > 1
        is_representative = dedup_group_id is None or db.execute(
            "SELECT representative_comment_id FROM dedup_groups WHERE dedup_group_id = ?",
            (dedup_group_id,),
        ).fetchone()[0] == comment_id if dedup_group_id else True

        score = _compute_substantiveness(
            full_text,
            is_form_letter=is_form_letter,
            has_org=bool(org and org.strip()),
            stakeholder_type=stakeholder_type or "",
            is_representative=is_representative,
            config=sub_cfg,
        )

        db.execute(
            """INSERT OR REPLACE INTO comment_classifications
               (comment_id, substantiveness_score)
               VALUES (?, ?)
               ON CONFLICT(comment_id) DO UPDATE SET substantiveness_score = ?""",
            (comment_id, score, score),
        )

    db.commit()


def _compute_substantiveness(
    text: str,
    is_form_letter: bool,
    has_org: bool = False,
    stakeholder_type: str = "",
    is_representative: bool = True,
    config: dict | None = None,
) -> int:
    """Compute a substantiveness score (0-100) for a comment.

    Designed so that a comment which is long, from an identified org, contains
    specific regulatory references, is unique, uses technical vocabulary, and
    includes data/legal arguments can realistically reach 75-85.

    Scoring components (additive, theoretical max ~100):
    - Length:                0-20 points
    - Organizational affil: 0-10 points
    - Unique/representative: 0-8 points
    - Section references:   0-12 points
    - Citations:            0-12 points
    - Technical vocabulary:  0-12 points
    - Data/statistics:      0-10 points
    - Legal arguments:      0-10 points
    - Structural quality:   0-6 points
    - Form letter penalty:  -20 points

    Args:
        text: Full comment text.
        is_form_letter: Whether this comment is part of a form letter group.
        has_org: Whether the commenter has an organizational affiliation.
        stakeholder_type: The classified stakeholder type.
        is_representative: Whether this is a unique comment or dedup group representative.
        config: Optional weight overrides.

    Returns:
        Integer score 0-100.
    """
    if config is None:
        config = {}

    score = 0.0

    # --- Length (0-20 points, logarithmic) ---
    # 200 chars → ~6, 500 → ~10, 1000 → ~13, 2000 → ~16, 5000 → ~19, 10000+ → 20
    text_len = len(text)
    if text_len > 0:
        length_pts = min(20.0, math.log(1 + text_len / 100.0) * 5.2)
        score += length_pts

    # --- Organizational affiliation (0-10 points) ---
    if has_org:
        score += 5.0
        if stakeholder_type in ("trade_association", "nonprofit", "government", "academic", "law_firm"):
            score += 5.0

    # --- Unique / representative comment (0-8 points) ---
    if is_representative and not is_form_letter:
        score += 8.0

    # --- Section references (0-12 points) ---
    section_patterns = [
        r"\b\d+\s*C\.?F\.?R\.?\s*(?:Part|§|Section)?\s*\d+",
        r"\b\d+\s*Fed\.?\s*Reg\.?\s*\d+",
        r"§\s*\d+",
        r"\bSection\s+\d+\.\d+",
        r"\bPart\s+\d+",
        r"\bE\.?O\.?\s*\d{4,5}",
        r"\bExecutive Order\s+\d+",
        r"\bDocket\s+(?:No\.?\s*)?[A-Z]",
    ]
    section_count = sum(len(re.findall(p, text, re.IGNORECASE)) for p in section_patterns)
    score += min(section_count * 4.0, 12.0)

    # --- Citations (0-12 points) ---
    citation_patterns = [
        r"https?://\S+",
        r"\bdoi:\s*\S+",
        r"\bcit(?:e[ds]?|ation)\b",
        r"\bsee\s+(?:also\s+)?\d+\s+(?:U\.?S\.?C|C\.?F\.?R|Fed\.?\s*Reg)",
        r"\bfootnote\s*\d+",
        r"\bid\.\s+at\b",
        r"\bsupra\b",
        r"\bibid\b",
        r"\baccording to\b",
    ]
    citation_count = sum(len(re.findall(p, text, re.IGNORECASE)) for p in citation_patterns)
    score += min(citation_count * 3.0, 12.0)

    # --- Technical vocabulary (0-12 points) ---
    # Count distinct term categories that appear (not raw count)
    technical_terms = [
        r"\bemission[s]?\b", r"\bpollutant[s]?\b", r"\bstandard[s]?\b",
        r"\bcompliance\b", r"\bregulat\w+\b", r"\bstatut\w+\b",
        r"\bprovision[s]?\b", r"\bamendment[s]?\b", r"\bprohibit\w+\b",
        r"\bmandat\w+\b", r"\benforceable\b", r"\bimplementat\w+\b",
        r"\bcost-benefit\b", r"\beconomic impact\b", r"\brisk assessment\b",
        r"\bbaseline\b", r"\bthreshold[s]?\b", r"\bmitigation\b",
        r"\binfrastructure\b", r"\bframework\b", r"\bmethodolog\w+\b",
        r"\balgorithm\w*\b", r"\bautonomous\b", r"\bmachine learning\b",
        r"\bartificial intelligence\b", r"\bsafety\b", r"\bsecurity\b",
        r"\bpropos\w+\b", r"\brequirement[s]?\b", r"\bimpact\b",
    ]
    # Count distinct terms present (not total occurrences)
    tech_present = sum(1 for t in technical_terms if re.search(t, text, re.IGNORECASE))
    score += min(tech_present * 1.2, 12.0)

    # --- Data and statistics (0-10 points) ---
    data_patterns = [
        r"\b\d+(?:\.\d+)?%",
        r"\$\s*[\d,.]+",
        r"\b\d+(?:,\d{3})+\b",
        r"\bdata\s+show[s]?\b",
        r"\bstud(?:y|ies)\s+(?:show|find|demonstrate|indicate)",
        r"\btable\s+\d+",
        r"\bfigure\s+\d+",
        r"\bappendix\b",
    ]
    data_count = sum(len(re.findall(p, text, re.IGNORECASE)) for p in data_patterns)
    score += min(data_count * 2.5, 10.0)

    # --- Legal arguments (0-10 points) ---
    legal_patterns = [
        r"\bArbitrary and Capricious\b",
        r"\bAdministrative Procedure Act\b",
        r"\bAPA\b",
        r"\bChevron\b",
        r"\bdue process\b",
        r"\bconstitutional\b",
        r"\blegal\s+authority\b",
        r"\bstatutory\b",
        r"\bjudicial review\b",
        r"\bclean air act\b",
        r"\bclean water act\b",
        r"\bendangered species\b",
    ]
    legal_count = sum(len(re.findall(p, text, re.IGNORECASE)) for p in legal_patterns)
    score += min(legal_count * 3.0, 10.0)

    # --- Structural quality (0-6 points) ---
    # Numbered lists / Q&A format — hallmark of substantive organizational comments
    numbered = len(re.findall(r"(?:^|\n)\s*(?:\d+[\.\):]|[a-z][\.\)]|[-•])\s+\w", text))
    score += min(numbered * 1.0, 3.0)
    # Paragraph structure (multiple substantial paragraphs)
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 80]
    score += min(len(paragraphs) * 0.5, 3.0)

    # --- Form letter penalty ---
    if is_form_letter:
        score -= 20.0

    return max(0, min(int(score), 100))
