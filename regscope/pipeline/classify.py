"""Classification pipeline — stakeholder type, stance detection, substantiveness scoring."""

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

# Business indicator words for heuristic org classification
_BUSINESS_INDICATORS = {
    "inc", "corp", "llc", "ltd", "group", "association", "company", "co",
    "foundation", "institute", "council", "board", "commission",
    "university", "college", "enterprises", "holdings", "partners",
    "solutions", "services", "technologies",
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
        """SELECT comment_id, organization, submitter_name FROM comments
           WHERE docket_id = ?
           AND comment_id NOT IN (
               SELECT comment_id FROM comment_classifications WHERE stakeholder_type IS NOT NULL
           )""",
        (docket_id,),
    ).fetchall()

    if not comments:
        logger.info("All comments already classified")
        return

    for comment_id, org, submitter_name in comments:
        stakeholder_type = _classify_org(org)

        if stakeholder_type == "unknown":
            if not org or not org.strip():
                # No organization → individual
                stakeholder_type = "individual"
            # If org was set but didn't match any pattern and the heuristic
            # returned "industry", keep that result

        db.execute(
            """INSERT INTO comment_classifications
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

    # Heuristic: classify as industry only if the name contains a business
    # indicator word. Acronyms alone (e.g., NAACP, ACLU, AFL-CIO) are not
    # reliable industry indicators — many nonprofits and unions use them.
    org_lower = org.lower()
    for indicator in _BUSINESS_INDICATORS:
        if indicator in org_lower.replace(".", "").replace(",", "").split():
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

        from regscope.utils.text import strip_html

        for comment_id, full_text in comments:
            try:
                text = strip_html(full_text)[:1024]

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
                    """INSERT INTO comment_classifications
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
                            """INSERT INTO comment_classifications
                               (comment_id, stance, stance_confidence)
                               VALUES (?, ?, ?)
                               ON CONFLICT(comment_id) DO UPDATE SET stance = ?, stance_confidence = ?""",
                            (member_id, stance, top_score, stance, top_score),
                        )

            except Exception:
                logger.exception("Stance detection failed for %s", comment_id)

            progress.update(task, advance=1)

            if comment_id and int(progress.tasks[task].completed or 0) % 10 == 0:
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
        if dedup_group_id is None:
            is_representative = True
        else:
            row = db.execute(
                "SELECT representative_comment_id FROM dedup_groups WHERE dedup_group_id = ?",
                (dedup_group_id,),
            ).fetchone()
            is_representative = row is None or row[0] == comment_id

        score = _compute_substantiveness(
            full_text,
            is_form_letter=is_form_letter,
            has_org=bool(org and org.strip()),
            stakeholder_type=stakeholder_type or "",
            is_representative=is_representative,
            config=sub_cfg,
        )

        db.execute(
            """INSERT INTO comment_classifications
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

    Each component produces a 0-100 subscore. The final score is the
    weighted sum of all subscores, using weights from config. If no config
    weights are provided, uses defaults matching config.toml.example.

    Components:
    - length: Text length (logarithmic scale)
    - citations: URLs, DOIs, legal citations
    - section_references: CFR/Fed Reg section references
    - technical_vocab: Distinct technical terms present
    - data_statistics: Percentages, dollar amounts, tables/figures
    - legal_arguments: Legal framework references
    - not_form_letter: Bonus for unique, non-form-letter comments

    Additional adjustments (not weighted):
    - Organizational affiliation bonus (0-10 points on final score)
    - Structural quality bonus (0-6 points on final score)
    - Form letter penalty (-20 points on final score)

    Args:
        text: Full comment text.
        is_form_letter: Whether this comment is part of a form letter group.
        has_org: Whether the commenter has an organizational affiliation.
        stakeholder_type: The classified stakeholder type.
        is_representative: Whether this is a unique comment or dedup group representative.
        config: Optional config dict with weight overrides.

    Returns:
        Integer score 0-100.
    """
    if config is None:
        config = {}

    # Load weights from config, with defaults
    w_length = config.get("weight_length", 0.15)
    w_citations = config.get("weight_citations", 0.20)
    w_sections = config.get("weight_section_references", 0.20)
    w_tech = config.get("weight_technical_vocab", 0.15)
    w_data = config.get("weight_data_statistics", 0.15)
    w_legal = config.get("weight_legal_arguments", 0.10)
    w_form = config.get("weight_not_form_letter", 0.05)

    # --- Length subscore (0-100) ---
    # 200 chars → ~30, 500 → ~50, 1000 → ~65, 2000 → ~80, 5000 → ~95, 10000+ → 100
    text_len = len(text)
    if text_len > 0:
        length_sub = min(100.0, math.log(1 + text_len / 100.0) * 26.0)
    else:
        length_sub = 0.0

    # --- Section references subscore (0-100) ---
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
    sections_sub = min(section_count * 33.0, 100.0)

    # --- Citations subscore (0-100) ---
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
    citations_sub = min(citation_count * 25.0, 100.0)

    # --- Technical vocabulary subscore (0-100) ---
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
    tech_present = sum(1 for t in technical_terms if re.search(t, text, re.IGNORECASE))
    tech_sub = min(tech_present * 10.0, 100.0)

    # --- Data and statistics subscore (0-100) ---
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
    data_sub = min(data_count * 25.0, 100.0)

    # --- Legal arguments subscore (0-100) ---
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
    legal_sub = min(legal_count * 30.0, 100.0)

    # --- Not-form-letter subscore (0 or 100) ---
    form_sub = 100.0 if (is_representative and not is_form_letter) else 0.0

    # Weighted sum of subscores
    score = (
        w_length * length_sub
        + w_citations * citations_sub
        + w_sections * sections_sub
        + w_tech * tech_sub
        + w_data * data_sub
        + w_legal * legal_sub
        + w_form * form_sub
    )

    # --- Organizational affiliation bonus (additive, not weighted) ---
    if has_org:
        score += 5.0
        if stakeholder_type in ("trade_association", "nonprofit", "government", "academic", "law_firm"):
            score += 5.0

    # --- Structural quality bonus (additive, not weighted) ---
    numbered = len(re.findall(r"(?:^|\n)\s*(?:\d+[\.\):]|[a-z][\.\)]|[-•])\s+\w", text))
    score += min(numbered * 1.0, 3.0)
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 80]
    score += min(len(paragraphs) * 0.5, 3.0)

    # --- Form letter penalty ---
    if is_form_letter:
        score -= 20.0

    return max(0, min(int(score), 100))
