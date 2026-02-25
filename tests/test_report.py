"""Tests for report generation pipeline — Markdown report output and helper functions."""

import json
import os
import sqlite3
from unittest.mock import patch

import pytest

from regscope.db import get_db
from regscope.pipeline.report import (
    _clean_excerpt,
    _generate_llm_topic_labels,
    _readable_topic_label,
    run_report,
)


@pytest.fixture
def test_db(tmp_path):
    """Create a test database with a docket and seed data for report generation."""
    config = {"data": {"data_dir": str(tmp_path)}}
    db = get_db("TEST-REPORT", config)
    db.execute(
        "INSERT INTO dockets (docket_id, title, agency, docket_type) "
        "VALUES ('TEST-REPORT', 'Report Test Docket', 'EPA', 'Rulemaking')"
    )
    db.commit()
    return db


@pytest.fixture
def populated_db(test_db):
    """A test database populated with comments, dedup groups, topics, and classifications."""
    db = test_db

    # Insert comments
    for i in range(1, 6):
        db.execute(
            """INSERT INTO comments
               (comment_id, docket_id, comment_text, full_text, submitter_name,
                organization, posted_date, detail_fetched)
               VALUES (?, 'TEST-REPORT', ?, ?, ?, ?, ?, 1)""",
            (
                f"CMT-{i}",
                f"Comment text {i} about the proposed regulation.",
                f"Full text of comment {i} discussing regulatory impact on the environment.",
                f"Submitter {i}",
                f"Org {i}" if i <= 3 else None,
                f"2025-01-{i:02d}",
            ),
        )

    # Insert a dedup group with two duplicated comments
    db.execute(
        """INSERT INTO dedup_groups
           (docket_id, group_type, group_size, representative_comment_id, template_text)
           VALUES ('TEST-REPORT', 'exact', 2, 'CMT-1',
                   'This is a template form letter about the regulation.')"""
    )
    dedup_group_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]

    # Assign first two comments to the dedup group
    db.execute(
        "UPDATE comments SET dedup_group_id = ? WHERE comment_id IN ('CMT-1', 'CMT-2')",
        (dedup_group_id,),
    )

    # Insert topics
    topic_keywords_1 = json.dumps(["environment", "pollution", "air quality", "emissions", "health"])
    topic_keywords_2 = json.dumps(["cost", "compliance", "small business", "economic", "burden"])
    rep_texts_1 = json.dumps(["This regulation will significantly improve air quality."])
    rep_texts_2 = json.dumps(["The compliance costs are too high for small businesses."])

    db.execute(
        """INSERT INTO topics
           (docket_id, bertopic_id, label, keywords, topic_size, representative_texts)
           VALUES ('TEST-REPORT', 0, '0_environment_pollution_air quality', ?, 3, ?)""",
        (topic_keywords_1, rep_texts_1),
    )
    topic_id_1 = db.execute("SELECT last_insert_rowid()").fetchone()[0]

    db.execute(
        """INSERT INTO topics
           (docket_id, bertopic_id, label, keywords, topic_size, representative_texts)
           VALUES ('TEST-REPORT', 1, '1_cost_compliance_small business', ?, 2, ?)""",
        (topic_keywords_2, rep_texts_2),
    )
    topic_id_2 = db.execute("SELECT last_insert_rowid()").fetchone()[0]

    # Assign comments to topics
    for i in range(1, 4):
        db.execute(
            "INSERT INTO comment_topics (comment_id, topic_id, relevance_score) VALUES (?, ?, ?)",
            (f"CMT-{i}", topic_id_1, 0.85),
        )
    for i in range(4, 6):
        db.execute(
            "INSERT INTO comment_topics (comment_id, topic_id, relevance_score) VALUES (?, ?, ?)",
            (f"CMT-{i}", topic_id_2, 0.90),
        )

    # Insert classifications
    stances = ["support", "oppose", "support", "conditional_oppose", "neutral_informational"]
    stakeholder_types = ["individual", "industry", "nonprofit", "trade_association", "individual"]
    for i in range(1, 6):
        db.execute(
            """INSERT INTO comment_classifications
               (comment_id, stakeholder_type, stance, stance_confidence, substantiveness_score)
               VALUES (?, ?, ?, ?, ?)""",
            (f"CMT-{i}", stakeholder_types[i - 1], stances[i - 1], 0.8, 50 + i * 10),
        )

    db.commit()
    return db


class TestCleanExcerpt:
    """Test the _clean_excerpt helper for Markdown-safe text."""

    def test_strips_html_tags(self) -> None:
        """HTML tags should be removed from excerpts."""
        result = _clean_excerpt("<p>Hello <strong>world</strong></p>")
        assert "<p>" not in result
        assert "<strong>" not in result
        assert "Hello" in result
        assert "world" in result

    def test_unescapes_html_entities(self) -> None:
        """HTML entities like &amp; should be converted to characters."""
        result = _clean_excerpt("Tom &amp; Jerry &lt;3")
        assert "Tom & Jerry <3" in result

    def test_converts_br_to_space(self) -> None:
        """<br> and <br/> tags should become spaces."""
        result = _clean_excerpt("Line one<br>Line two<br/>Line three")
        assert "Line one Line two Line three" in result

    def test_truncates_long_text(self) -> None:
        """Text exceeding max_len should be truncated with ellipsis."""
        long_text = "word " * 200  # 1000 chars
        result = _clean_excerpt(long_text, max_len=50)
        assert len(result) <= 54  # 50 + "..."
        assert result.endswith("...")

    def test_preserves_short_text(self) -> None:
        """Text shorter than max_len should not be truncated."""
        result = _clean_excerpt("Short text", max_len=500)
        assert result == "Short text"
        assert "..." not in result

    def test_normalizes_whitespace(self) -> None:
        """Multiple whitespace characters should be collapsed."""
        result = _clean_excerpt("too   many    spaces   here")
        assert "too many spaces here" in result


class TestReadableTopicLabel:
    """Test BERTopic label formatting into readable display names."""

    def test_bertopic_style_label(self) -> None:
        """BERTopic-style numeric prefix labels should be converted to readable form."""
        keywords = json.dumps(["environment", "pollution", "health"])
        result = _readable_topic_label("0_environment_pollution_health", keywords)
        assert "Environment" in result
        assert "Pollution" in result
        assert "Health" in result

    def test_known_acronyms_uppercased(self) -> None:
        """Known acronyms like EPA, BOEM should be uppercased."""
        keywords = json.dumps(["epa", "regulation", "compliance"])
        result = _readable_topic_label("0_epa_regulation_compliance", keywords)
        assert "EPA" in result

    def test_deduplication_of_overlapping_keywords(self) -> None:
        """Overlapping keywords (e.g., 'project' and 'project help') should be deduplicated."""
        keywords = json.dumps(["project help", "project", "communities", "pollution"])
        result = _readable_topic_label("0_project help_project_communities", keywords)
        # "project" should not appear separately when "project help" is already included
        parts = [p.strip() for p in result.split("/")]
        # Count how many parts contain "project" — should only be one
        project_parts = [p for p in parts if "project" in p.lower()]
        assert len(project_parts) == 1

    def test_three_keyword_limit(self) -> None:
        """At most 3 keywords should appear in the label."""
        keywords = json.dumps(["alpha", "beta", "gamma", "delta", "epsilon"])
        result = _readable_topic_label("0_alpha_beta_gamma_delta", keywords)
        parts = [p.strip() for p in result.split("/")]
        assert len(parts) <= 3

    def test_empty_keywords_returns_raw_label(self) -> None:
        """If no keywords are provided, the raw label should be returned."""
        result = _readable_topic_label("some_raw_label", None)
        assert result == "some_raw_label"

        result = _readable_topic_label("some_raw_label", "[]")
        assert result == "some_raw_label"

    def test_short_unknown_words_uppercased(self) -> None:
        """Short words (<=3 chars) that are not common words should be uppercased."""
        keywords = json.dumps(["osw", "wind", "energy"])
        result = _readable_topic_label("0_osw_wind_energy", keywords)
        assert "OSW" in result

    def test_common_short_words_title_cased(self) -> None:
        """Common short words like 'new' should be title-cased, not uppercased."""
        keywords = json.dumps(["new", "plan", "farm"])
        result = _readable_topic_label("0_new_plan_farm", keywords)
        assert "New" in result
        assert "NEW" not in result


class TestGenerateLlmTopicLabels:
    """Test LLM topic label generation with graceful degradation."""

    def test_no_api_key_returns_empty(self, test_db) -> None:
        """When OPENAI_API_KEY is not set, the function should return an empty dict."""
        # Ensure the env var is not set
        env_without_key = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
        with patch.dict(os.environ, env_without_key, clear=True):
            topics_data = [
                (1, "0_env_pollution", '["environment", "pollution"]', 10, '["sample text"]'),
            ]
            config = {"llm": {"enabled": True}}
            result = _generate_llm_topic_labels(test_db, "TEST-REPORT", topics_data, config)
            assert result == {}

    def test_llm_disabled_in_config(self, test_db) -> None:
        """When LLM is disabled in config, should return empty dict immediately."""
        topics_data = [
            (1, "0_env_pollution", '["environment", "pollution"]', 10, '["sample text"]'),
        ]
        config = {"llm": {"enabled": False}}
        result = _generate_llm_topic_labels(test_db, "TEST-REPORT", topics_data, config)
        assert result == {}

    def test_openai_not_installed_returns_empty(self, test_db) -> None:
        """When the openai package is not installed, should return empty dict."""
        # Set a fake API key so we get past the env check
        with patch.dict(os.environ, {"OPENAI_API_KEY": "fake-key"}):
            with patch("builtins.__import__", side_effect=_mock_import_no_openai):
                topics_data = [
                    (1, "0_env_pollution", '["environment", "pollution"]', 10, '["sample text"]'),
                ]
                config = {"llm": {"enabled": True}}
                result = _generate_llm_topic_labels(test_db, "TEST-REPORT", topics_data, config)
                assert result == {}


def _mock_import_no_openai(name, *args, **kwargs):
    """Mock import that raises ImportError for openai."""
    if name == "openai":
        raise ImportError("No module named 'openai'")
    return original_import(name, *args, **kwargs)


# Capture the original import for fallback
import builtins
original_import = builtins.__import__


class TestRunReport:
    """Test end-to-end report generation."""

    def test_report_contains_all_sections(self, populated_db, tmp_path) -> None:
        """The generated report should contain all expected section headings."""
        output_path = str(tmp_path / "test_report.md")
        config = {"data": {"data_dir": str(tmp_path)}, "report": {}}

        run_report(populated_db, "TEST-REPORT", config, output_path=output_path)

        with open(output_path) as f:
            report = f.read()

        # Check top-level title
        assert "# RegScope Analysis Report: TEST-REPORT" in report

        # Check all section headings
        assert "## 1. Docket Overview" in report
        assert "## 2. Comment Landscape" in report
        assert "## 3. Topic Analysis" in report
        assert "## 4. Stakeholder Breakdown" in report
        assert "## 5. Stance Analysis" in report
        assert "## 6. Substantive Comment Highlights" in report
        assert "## 7. Data Quality Notes" in report

    def test_docket_overview_contains_metadata(self, populated_db, tmp_path) -> None:
        """The Docket Overview section should contain docket metadata."""
        output_path = str(tmp_path / "overview_report.md")
        config = {"data": {"data_dir": str(tmp_path)}, "report": {}}

        run_report(populated_db, "TEST-REPORT", config, output_path=output_path)

        with open(output_path) as f:
            report = f.read()

        assert "TEST-REPORT" in report
        assert "Report Test Docket" in report
        assert "EPA" in report
        # Should show comment count
        assert "5" in report

    def test_comment_landscape_shows_dedup_info(self, populated_db, tmp_path) -> None:
        """The Comment Landscape section should show duplicate group info."""
        output_path = str(tmp_path / "landscape_report.md")
        config = {"data": {"data_dir": str(tmp_path)}, "report": {}}

        run_report(populated_db, "TEST-REPORT", config, output_path=output_path)

        with open(output_path) as f:
            report = f.read()

        # Should mention duplicates
        assert "Duplicate groups" in report or "duplicate" in report.lower()

    def test_topic_analysis_shows_topics(self, populated_db, tmp_path) -> None:
        """The Topic Analysis section should list the topics with keywords."""
        output_path = str(tmp_path / "topics_report.md")
        config = {"data": {"data_dir": str(tmp_path)}, "report": {}}

        run_report(populated_db, "TEST-REPORT", config, output_path=output_path)

        with open(output_path) as f:
            report = f.read()

        # Should contain topic keywords
        assert "environment" in report.lower()
        assert "cost" in report.lower()
        # Should have representative quotes section
        assert "Representative Quotes" in report

    def test_stakeholder_breakdown_shows_types(self, populated_db, tmp_path) -> None:
        """The Stakeholder Breakdown section should list stakeholder types with counts."""
        output_path = str(tmp_path / "stakeholder_report.md")
        config = {"data": {"data_dir": str(tmp_path)}, "report": {}}

        run_report(populated_db, "TEST-REPORT", config, output_path=output_path)

        with open(output_path) as f:
            report = f.read()

        assert "individual" in report
        assert "industry" in report
        assert "nonprofit" in report

    def test_stance_analysis_shows_distribution(self, populated_db, tmp_path) -> None:
        """The Stance Analysis section should list stance categories."""
        output_path = str(tmp_path / "stance_report.md")
        config = {"data": {"data_dir": str(tmp_path)}, "report": {}}

        run_report(populated_db, "TEST-REPORT", config, output_path=output_path)

        with open(output_path) as f:
            report = f.read()

        assert "support" in report.lower()
        assert "oppose" in report.lower()

    def test_substantive_highlights_shows_scores(self, populated_db, tmp_path) -> None:
        """The Substantive Highlights section should show top comments with scores."""
        output_path = str(tmp_path / "substantive_report.md")
        config = {"data": {"data_dir": str(tmp_path)}, "report": {}}

        run_report(populated_db, "TEST-REPORT", config, output_path=output_path)

        with open(output_path) as f:
            report = f.read()

        # Should show scores
        assert "/100" in report
        # Should show submitter names
        assert "Submitter" in report

    def test_data_quality_notes_present(self, populated_db, tmp_path) -> None:
        """The Data Quality Notes section should contain quality metrics."""
        output_path = str(tmp_path / "quality_report.md")
        config = {"data": {"data_dir": str(tmp_path)}, "report": {}}

        run_report(populated_db, "TEST-REPORT", config, output_path=output_path)

        with open(output_path) as f:
            report = f.read()

        assert "Data Quality" in report
        assert "Total comments" in report

    def test_report_creates_output_directory(self, populated_db, tmp_path) -> None:
        """The report should create the output directory if it does not exist."""
        output_path = str(tmp_path / "nested" / "dir" / "report.md")
        config = {"data": {"data_dir": str(tmp_path)}, "report": {}}

        run_report(populated_db, "TEST-REPORT", config, output_path=output_path)

        assert os.path.exists(output_path)

    def test_report_with_empty_docket(self, test_db, tmp_path) -> None:
        """A report on a docket with no comments should still generate without errors."""
        output_path = str(tmp_path / "empty_report.md")
        config = {"data": {"data_dir": str(tmp_path)}, "report": {}}

        run_report(test_db, "TEST-REPORT", config, output_path=output_path)

        with open(output_path) as f:
            report = f.read()

        assert "# RegScope Analysis Report: TEST-REPORT" in report
        assert "## 1. Docket Overview" in report
