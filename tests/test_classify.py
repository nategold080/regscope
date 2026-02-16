"""Tests for stakeholder classification patterns."""

import pytest

from regscope.pipeline.classify import _classify_org, _compute_substantiveness


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
        """Empty or None orgs should be unknown; capitalized names fall back to industry."""
        assert _classify_org("") == "unknown"
        assert _classify_org(None) == "unknown"
        # Capitalized multi-word names with no pattern match → industry heuristic
        assert _classify_org("Some Random Name") == "industry"

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
