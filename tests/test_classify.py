"""Tests for stakeholder classification patterns and pipeline integration."""

import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from regscope.pipeline.classify import (
    _classify_org,
    _classify_stakeholders,
    _compute_substantiveness,
    _score_substantiveness,
    run_classify,
)


class TestStakeholderClassification:
    """Test rule-based stakeholder classification."""

    def test_trade_association(self) -> None:
        """Trade associations should be correctly classified."""
        assert _classify_org("American Petroleum Institute") == "trade_association"
        assert _classify_org("National Mining Association") == "trade_association"
        assert _classify_org("Chamber of Commerce") == "trade_association"
        assert _classify_org("American Chemistry Council") == "trade_association"

    def test_government(self) -> None:
        """Government entities should be correctly classified."""
        assert _classify_org("State of California") == "government"
        assert _classify_org("California Air Resources Board") == "government"
        assert _classify_org("County of Los Angeles") == "government"
        assert _classify_org("Department of Environmental Quality") == "government"
        assert _classify_org("U.S. Environmental Protection Agency") == "government"

    def test_industry(self) -> None:
        """Companies should be classified as industry."""
        assert _classify_org("Acme Manufacturing Corp.") == "industry"
        assert _classify_org("ExxonMobil Corporation") == "industry"
        assert _classify_org("Shell Energy LLC") == "industry"
        assert _classify_org("Power Solutions Inc.") == "industry"

    def test_nonprofit(self) -> None:
        """Nonprofits should be correctly classified."""
        assert _classify_org("Environmental Defense Fund") == "nonprofit"
        assert _classify_org("Sierra Club Foundation") == "nonprofit"
        assert _classify_org("Center for Biological Diversity") == "nonprofit"

    def test_academic(self) -> None:
        """Academic institutions should be correctly classified."""
        assert _classify_org("Harvard University") == "academic"
        assert _classify_org("MIT School of Engineering") == "academic"
        assert _classify_org("Stanford Research Laboratory") == "academic"

    def test_law_firm(self) -> None:
        """Law firms should be correctly classified."""
        assert _classify_org("Baker & McKenzie LLP") == "law_firm"
        assert _classify_org("Smith Law Office") == "law_firm"

    def test_unknown(self) -> None:
        """Empty or None orgs should be unknown; personal names without business indicators too."""
        assert _classify_org("") == "unknown"
        assert _classify_org(None) == "unknown"
        # Personal names without business indicators should be unknown, not industry
        assert _classify_org("Some Random Name") == "unknown"
        assert _classify_org("Van Der Berg") == "unknown"

    def test_industry_fallback_with_indicator(self) -> None:
        """Names with business indicators should fall back to industry."""
        assert _classify_org("Acme Corp") == "industry"
        assert _classify_org("XYZ Holdings") == "industry"
        # Acronyms (2+ uppercase letters) should also match
        assert _classify_org("Hayden AI") == "industry"

    def test_individual_no_org(self) -> None:
        """Empty organization should be classified as unknown (individual logic is in the caller)."""
        assert _classify_org("") == "unknown"


class TestSubstantivenessScoring:
    """Test substantiveness heuristic scoring."""

    def test_short_form_letter_low_score(self) -> None:
        """Short form letters should score low."""
        text = "I support this rule."
        score = _compute_substantiveness(text, is_form_letter=True, config={})
        assert score < 20

    def test_substantive_comment_high_score(self) -> None:
        """Comments with citations, data, and legal references should score high."""
        text = """
        We oppose this proposed rule. The compliance costs outlined in the
        Regulatory Impact Analysis (85 Fed. Reg. 12345) significantly
        underestimate the impact on small refineries. According to our
        analysis under 42 U.S.C. § 7411, the proposed emission standards
        of 25 ppm would require capital expenditures exceeding $50 million
        per facility. Studies show that 40% of affected facilities cannot
        meet the proposed timeline. See also NRDC v. EPA, 571 F.3d 1245
        (D.C. Cir. 2009). The Administrative Procedure Act requires the
        agency to consider the economic impact of its regulations.
        See https://example.com/technical-report.pdf for our detailed analysis.
        """
        score = _compute_substantiveness(text, is_form_letter=False, config={})
        assert score > 30  # Should be meaningfully higher than a form letter

    def test_medium_substantiveness(self) -> None:
        """Comments with some substance but no citations should score in the middle."""
        text = """
        I am writing to express my concerns about this proposed regulation.
        As a small business owner in the manufacturing sector, I believe the
        proposed emission standards are technically achievable but the timeline
        is too aggressive. I would recommend extending the compliance deadline
        by at least two years to allow for proper planning and implementation.
        """
        score = _compute_substantiveness(text, is_form_letter=False, config={})
        assert 10 <= score <= 60

    def test_long_comment_with_citations_exceeds_60(self) -> None:
        """A long comment with citations, legal refs, and data should score above 60."""
        text = """
        We respectfully submit these comments on the proposed rule published at
        85 Fed. Reg. 12345. Our organization has conducted extensive analysis of
        the proposed emission standards under 42 U.S.C. § 7411 and 40 C.F.R.
        Part 60 Subpart OOOOa.

        The Regulatory Impact Analysis significantly underestimates compliance
        costs. According to our study of 250 affected facilities, the proposed
        standard of 25 ppm would require capital expenditures exceeding
        $50 million per facility — a 340% increase over the agency's estimate.
        Data show that 40% of small refineries cannot meet the 2028 deadline.
        See Table 3 and Figure 7 in our attached technical report.

        The agency's cost-benefit analysis fails to account for job losses in
        rural communities. Our economic impact assessment projects that 15,000
        jobs would be eliminated in the first three years of implementation.
        This contradicts the agency's baseline assumptions (see Appendix B).

        Under the Administrative Procedure Act, the agency must provide a
        reasoned explanation for its methodology. See NRDC v. EPA, 571 F.3d
        1245 (D.C. Cir. 2009); see also Chevron U.S.A. v. NRDC, 467 U.S.
        837 (1984). The proposed threshold of 25 ppm is arbitrary and capricious
        absent adequate supporting data.

        We urge the agency to extend the compliance deadline by at least two
        years and revise the risk assessment framework to incorporate regional
        economic factors. See https://example.com/our-analysis.pdf for our
        complete technical submission.
        """
        score = _compute_substantiveness(
            text,
            is_form_letter=False,
            has_org=True,
            stakeholder_type="trade_association",
            is_representative=True,
            config={},
        )
        assert score > 60, f"Expected score > 60 for highly substantive comment, got {score}"

    def test_microsoft_corporation_classified_as_industry(self) -> None:
        """'Microsoft Corporation' should be classified as 'industry', not 'unknown'."""
        assert _classify_org("Microsoft Corporation") == "industry"

    def test_score_range(self) -> None:
        """All scores should be in 0-100 range."""
        for text in ["", "a", "hello world", "x" * 10000]:
            score = _compute_substantiveness(text, is_form_letter=False, config={})
            assert 0 <= score <= 100


def _setup_test_db() -> sqlite3.Connection:
    """Create an in-memory DB with schema for integration tests."""
    from regscope.db import SCHEMA_SQL

    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA_SQL)
    conn.execute("INSERT OR REPLACE INTO schema_version (version) VALUES (3)")
    conn.commit()
    return conn


def _insert_test_comments(db: sqlite3.Connection, docket_id: str) -> list[str]:
    """Insert test comments for classification integration tests."""
    db.execute(
        "INSERT OR IGNORE INTO dockets (docket_id, title) VALUES (?, ?)",
        (docket_id, "Test Docket"),
    )
    comments = [
        ("C-001", docket_id, "Sierra Club Foundation", "Jane Doe",
         "We strongly support the proposed emission standards under 42 U.S.C. § 7411. "
         "These regulations are essential for public health."),
        ("C-002", docket_id, "ExxonMobil Corporation", "John Smith",
         "We oppose this proposed rule. The compliance costs of $50 million per facility "
         "are excessive. See 85 Fed. Reg. 12345."),
        ("C-003", docket_id, "", "Anonymous Citizen",
         "I support this rule."),
    ]
    for cid, did, org, name, text in comments:
        db.execute(
            """INSERT INTO comments (comment_id, docket_id, organization,
               submitter_name, full_text, comment_text) VALUES (?, ?, ?, ?, ?, ?)""",
            (cid, did, org, name, text, text),
        )
    db.commit()
    return [c[0] for c in comments]


class TestClassifyStakeholdersDB:
    """Integration tests for _classify_stakeholders with a real database."""

    def test_stakeholder_classification_writes_to_db(self) -> None:
        """Stakeholder classification should write correct types to comment_classifications."""
        db = _setup_test_db()
        docket_id = "TEST-DOCKET-001"
        _insert_test_comments(db, docket_id)

        _classify_stakeholders(db, docket_id, {})

        rows = db.execute(
            "SELECT comment_id, stakeholder_type FROM comment_classifications ORDER BY comment_id"
        ).fetchall()
        result = {r[0]: r[1] for r in rows}
        assert result["C-001"] == "nonprofit"
        assert result["C-002"] == "industry"
        assert result["C-003"] == "individual"

    def test_stakeholder_idempotent(self) -> None:
        """Running stakeholder classification twice should not duplicate rows."""
        db = _setup_test_db()
        docket_id = "TEST-DOCKET-001"
        _insert_test_comments(db, docket_id)

        _classify_stakeholders(db, docket_id, {})
        _classify_stakeholders(db, docket_id, {})

        count = db.execute("SELECT COUNT(*) FROM comment_classifications").fetchone()[0]
        assert count == 3


class TestSubstantivenessScoringDB:
    """Integration tests for _score_substantiveness with a real database."""

    def test_scoring_writes_to_db(self) -> None:
        """Substantiveness scoring should write scores without destroying stakeholder_type."""
        db = _setup_test_db()
        docket_id = "TEST-DOCKET-001"
        _insert_test_comments(db, docket_id)

        # First run stakeholder classification
        _classify_stakeholders(db, docket_id, {})

        # Then run substantiveness scoring
        _score_substantiveness(db, docket_id, {})

        rows = db.execute(
            """SELECT comment_id, stakeholder_type, substantiveness_score
               FROM comment_classifications ORDER BY comment_id"""
        ).fetchall()
        result = {r[0]: (r[1], r[2]) for r in rows}

        # Stakeholder types should be preserved
        assert result["C-001"][0] == "nonprofit"
        assert result["C-002"][0] == "industry"
        assert result["C-003"][0] == "individual"

        # All should have scores
        for cid in ("C-001", "C-002", "C-003"):
            assert result[cid][1] is not None
            assert 0 <= result[cid][1] <= 100

    def test_scoring_handles_missing_dedup_group(self) -> None:
        """Scoring should not crash when dedup_group_id references a deleted group."""
        db = _setup_test_db()
        docket_id = "TEST-DOCKET-001"
        _insert_test_comments(db, docket_id)

        # Create a dedup group, assign it to C-001, then delete the group
        cursor = db.execute(
            """INSERT INTO dedup_groups (docket_id, group_type, group_size, representative_comment_id)
               VALUES (?, 'exact', 2, 'C-001')""",
            (docket_id,),
        )
        gid = cursor.lastrowid
        db.execute("UPDATE comments SET dedup_group_id = ? WHERE comment_id = 'C-001'", (gid,))
        db.commit()

        # Now delete the group row (simulates data inconsistency)
        db.execute("PRAGMA foreign_keys=OFF")
        db.execute("DELETE FROM dedup_groups WHERE dedup_group_id = ?", (gid,))
        db.execute("PRAGMA foreign_keys=ON")
        db.commit()

        _classify_stakeholders(db, docket_id, {})
        _score_substantiveness(db, docket_id, {})

        row = db.execute(
            "SELECT substantiveness_score FROM comment_classifications WHERE comment_id = 'C-001'"
        ).fetchone()
        assert row is not None
        assert row[0] is not None


class TestRunClassifyIntegration:
    """Integration tests for the full run_classify pipeline."""

    @patch("regscope.pipeline.classify._detect_stance")
    def test_all_stages_preserve_data(self, mock_stance: MagicMock) -> None:
        """All three sub-stages should preserve each other's data in comment_classifications.

        We mock _detect_stance to avoid loading the transformers model, but simulate
        its effect by manually inserting stance data before substantiveness scoring.
        """
        db = _setup_test_db()
        docket_id = "TEST-DOCKET-001"
        _insert_test_comments(db, docket_id)

        def fake_stance(db: sqlite3.Connection, docket_id: str, config: dict) -> None:
            """Simulate stance detection by inserting stance values."""
            for cid, stance, conf in [("C-001", "support", 0.85), ("C-002", "oppose", 0.92)]:
                db.execute(
                    """INSERT INTO comment_classifications
                       (comment_id, stance, stance_confidence)
                       VALUES (?, ?, ?)
                       ON CONFLICT(comment_id) DO UPDATE SET stance = ?, stance_confidence = ?""",
                    (cid, stance, conf, stance, conf),
                )
            db.commit()

        mock_stance.side_effect = fake_stance

        run_classify(db, docket_id, {})

        rows = db.execute(
            """SELECT comment_id, stakeholder_type, stance, stance_confidence, substantiveness_score
               FROM comment_classifications ORDER BY comment_id"""
        ).fetchall()
        result = {r[0]: {"type": r[1], "stance": r[2], "conf": r[3], "score": r[4]} for r in rows}

        # All three columns should be populated for C-001 and C-002
        assert result["C-001"]["type"] == "nonprofit"
        assert result["C-001"]["stance"] == "support"
        assert result["C-001"]["conf"] == 0.85
        assert result["C-001"]["score"] is not None

        assert result["C-002"]["type"] == "industry"
        assert result["C-002"]["stance"] == "oppose"
        assert result["C-002"]["conf"] == 0.92
        assert result["C-002"]["score"] is not None

        # C-003 has no stance (short text, wasn't set by fake_stance)
        assert result["C-003"]["type"] == "individual"
        assert result["C-003"]["score"] is not None
