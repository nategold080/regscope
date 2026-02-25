"""Tests for topic modeling pipeline."""

import json
import sqlite3

import numpy as np
import pytest

from regscope.db import get_db
from regscope.pipeline.topics import _clean_for_topics, _propagate_topics_to_groups


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def test_db(tmp_path):
    """Create a test database with the full schema and a test docket."""
    config = {"data": {"data_dir": str(tmp_path)}}
    db = get_db("TEST-TOPICS", config)
    db.execute(
        "INSERT INTO dockets (docket_id, title) VALUES ('TEST-TOPICS', 'Topic Modeling Test')"
    )
    db.commit()
    return db


def _insert_comment(db: sqlite3.Connection, comment_id: str, text: str,
                     docket_id: str = "TEST-TOPICS",
                     dedup_group_id: int | None = None) -> None:
    """Helper to insert a comment with full_text."""
    db.execute(
        """INSERT INTO comments
           (comment_id, docket_id, full_text, comment_text, detail_fetched, dedup_group_id)
           VALUES (?, ?, ?, ?, 1, ?)""",
        (comment_id, docket_id, text, text, dedup_group_id),
    )


def _insert_embedding(db: sqlite3.Connection, comment_id: str,
                       embedding: np.ndarray) -> None:
    """Helper to insert a pre-computed embedding."""
    db.execute(
        "INSERT INTO embeddings (comment_id, embedding) VALUES (?, ?)",
        (comment_id, embedding.astype(np.float32).tobytes()),
    )


# ---------------------------------------------------------------------------
# Original tests (kept as-is)
# ---------------------------------------------------------------------------

class TestTopicConfiguration:
    """Test topic modeling configuration and parameters."""

    def test_default_config_values(self) -> None:
        """Default config should have sensible topic modeling defaults."""
        from regscope.config import DEFAULTS

        topic_cfg = DEFAULTS["topics"]
        assert topic_cfg["min_topic_size"] == 10
        assert topic_cfg["nr_topics"] == "auto"
        assert topic_cfg["top_n_words"] == 10
        assert topic_cfg["umap_n_neighbors"] == 15
        assert topic_cfg["umap_n_components"] == 5
        assert topic_cfg["hdbscan_min_cluster_size"] == 10

    def test_topic_model_imports(self) -> None:
        """All required topic modeling libraries should be importable."""
        from umap import UMAP
        from hdbscan import HDBSCAN
        from bertopic import BERTopic
        from sklearn.feature_extraction.text import CountVectorizer

        # Just verify imports don't raise
        assert UMAP is not None
        assert HDBSCAN is not None
        assert BERTopic is not None
        assert CountVectorizer is not None


# ---------------------------------------------------------------------------
# Test 1: Small corpus topic modeling with real embeddings
# ---------------------------------------------------------------------------

class TestSmallCorpusTopicModeling:
    """Run BERTopic on a small corpus with clearly separable topics."""

    # Documents with three distinct themes
    ENVIRONMENT_DOCS = [
        "Air pollution from factories is causing respiratory diseases in nearby communities.",
        "Carbon dioxide emissions must be reduced to combat global warming and climate change.",
        "Industrial waste water is contaminating rivers and harming aquatic ecosystems.",
        "Deforestation in the Amazon is accelerating biodiversity loss at an alarming rate.",
        "Toxic chemical spills from oil refineries pollute groundwater and soil.",
        "Greenhouse gas emissions from power plants are a leading cause of acid rain.",
        "Ocean acidification caused by carbon emissions threatens coral reef ecosystems.",
        "Smog and particulate matter in urban areas increase asthma rates among children.",
    ]

    HEALTHCARE_DOCS = [
        "Rising prescription drug prices are making essential medications unaffordable.",
        "Hospitals are facing nursing shortages that compromise patient care quality.",
        "Health insurance premiums have increased dramatically over the past decade.",
        "Mental health services are severely underfunded in rural communities.",
        "The cost of emergency room visits discourages low-income patients from seeking care.",
        "Medicare reimbursement rates for physicians need to be updated to reflect inflation.",
        "Telehealth services expanded during the pandemic and should remain accessible.",
        "Preventive healthcare screenings can reduce long-term medical treatment costs.",
    ]

    EDUCATION_DOCS = [
        "Student loan debt is preventing young graduates from buying homes and starting families.",
        "Public school funding disparities between wealthy and poor districts must be addressed.",
        "Teacher salaries are too low to attract and retain qualified educators in classrooms.",
        "Early childhood education programs like Head Start improve long-term academic outcomes.",
        "Standardized testing does not accurately measure student learning and critical thinking.",
        "College tuition increases have far outpaced wage growth over the past twenty years.",
        "School curriculum should include financial literacy and practical life skills courses.",
        "Special education students need more resources and individualized support in schools.",
    ]

    def test_bertopic_finds_distinct_topics(self, test_db) -> None:
        """BERTopic should find at least 2 topics from 24 documents in 3 clear themes."""
        from sentence_transformers import SentenceTransformer
        from regscope.pipeline.topics import run_topics

        all_docs = self.ENVIRONMENT_DOCS + self.HEALTHCARE_DOCS + self.EDUCATION_DOCS

        # Generate real embeddings
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(all_docs, convert_to_numpy=True)

        # Insert comments and embeddings into the test database
        for i, text in enumerate(all_docs):
            cid = f"TOPIC-{i:03d}"
            _insert_comment(test_db, cid, text)
            _insert_embedding(test_db, cid, embeddings[i])
        test_db.commit()

        # Run the full topic pipeline
        config = {"topics": {"nr_topics": "auto", "top_n_words": 10}}
        run_topics(test_db, "TEST-TOPICS", config)

        # Check that topics were created (excluding the outlier topic with bertopic_id = -1)
        topic_rows = test_db.execute(
            "SELECT topic_id, bertopic_id, label, keywords FROM topics WHERE docket_id = 'TEST-TOPICS'"
        ).fetchall()

        real_topics = [r for r in topic_rows if r[1] != -1]
        assert len(real_topics) >= 2, (
            f"Expected at least 2 real topics from 3 clear themes, got {len(real_topics)}"
        )

        # Check that most comments got a topic assignment
        assigned = test_db.execute(
            "SELECT COUNT(DISTINCT comment_id) FROM comment_topics"
        ).fetchone()[0]
        assert assigned >= len(all_docs) * 0.5, (
            f"Expected at least half of {len(all_docs)} comments to get topic assignments, "
            f"but only {assigned} were assigned"
        )

    def test_topic_keywords_are_stored(self, test_db) -> None:
        """Each topic should have a JSON-encoded keywords list in the database."""
        from sentence_transformers import SentenceTransformer
        from regscope.pipeline.topics import run_topics

        all_docs = self.ENVIRONMENT_DOCS + self.HEALTHCARE_DOCS + self.EDUCATION_DOCS
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(all_docs, convert_to_numpy=True)

        for i, text in enumerate(all_docs):
            cid = f"KW-{i:03d}"
            _insert_comment(test_db, cid, text)
            _insert_embedding(test_db, cid, embeddings[i])
        test_db.commit()

        config = {"topics": {"nr_topics": "auto", "top_n_words": 5}}
        run_topics(test_db, "TEST-TOPICS", config)

        topic_rows = test_db.execute(
            "SELECT keywords FROM topics WHERE docket_id = 'TEST-TOPICS' AND bertopic_id != -1"
        ).fetchall()

        for (kw_json,) in topic_rows:
            keywords = json.loads(kw_json)
            assert isinstance(keywords, list)
            assert len(keywords) > 0, "Each topic should have at least one keyword"
            assert all(isinstance(k, str) for k in keywords)


# ---------------------------------------------------------------------------
# Test 2: Topic propagation to dedup group members
# ---------------------------------------------------------------------------

class TestTopicPropagation:
    """Test _propagate_topics_to_groups assigns representative's topic to members."""

    def test_members_receive_representative_topic(self, test_db) -> None:
        """Non-representative group members should get the representative's topic."""
        # Insert comments first (FK: dedup_groups.representative_comment_id -> comments)
        _insert_comment(test_db, "C-REP", "Form letter text about regulation.")
        _insert_comment(test_db, "C-M1", "Form letter text about regulation.")
        _insert_comment(test_db, "C-M2", "Form letter text about regulation.")

        # Create the dedup group referencing the now-existing representative
        test_db.execute(
            """INSERT INTO dedup_groups
               (dedup_group_id, docket_id, group_type, group_size, representative_comment_id)
               VALUES (1, 'TEST-TOPICS', 'exact', 3, 'C-REP')"""
        )

        # Set dedup_group_id on the comments
        for cid in ("C-REP", "C-M1", "C-M2"):
            test_db.execute(
                "UPDATE comments SET dedup_group_id = 1 WHERE comment_id = ?", (cid,)
            )

        # Create a real topic (not the outlier)
        test_db.execute(
            """INSERT INTO topics (topic_id, docket_id, bertopic_id, label, keywords, topic_size)
               VALUES (10, 'TEST-TOPICS', 0, 'Regulation Topic', '["regulation","policy"]', 5)"""
        )

        # Assign the representative to that topic
        test_db.execute(
            "INSERT INTO comment_topics (comment_id, topic_id, relevance_score) VALUES ('C-REP', 10, 0.85)"
        )
        test_db.commit()

        # Propagate
        _propagate_topics_to_groups(test_db, "TEST-TOPICS")
        test_db.commit()

        # Verify members got the same topic
        for member_id in ("C-M1", "C-M2"):
            row = test_db.execute(
                "SELECT topic_id, relevance_score FROM comment_topics WHERE comment_id = ?",
                (member_id,),
            ).fetchone()
            assert row is not None, f"{member_id} should have a topic assignment after propagation"
            assert row[0] == 10, f"{member_id} should have topic_id 10"
            assert row[1] == pytest.approx(0.85), f"{member_id} should have relevance_score 0.85"

    def test_outlier_representative_falls_back_to_member_topic(self, test_db) -> None:
        """If the representative is an outlier, use a member's non-outlier topic instead."""
        # Insert comments first, then the dedup group, then set dedup_group_id
        _insert_comment(test_db, "OUT-REP", "Outlier representative text.")
        _insert_comment(test_db, "OUT-M1", "Member one text.")
        _insert_comment(test_db, "OUT-M2", "Member two text.")

        test_db.execute(
            """INSERT INTO dedup_groups
               (dedup_group_id, docket_id, group_type, group_size, representative_comment_id)
               VALUES (2, 'TEST-TOPICS', 'exact', 3, 'OUT-REP')"""
        )
        for cid in ("OUT-REP", "OUT-M1", "OUT-M2"):
            test_db.execute(
                "UPDATE comments SET dedup_group_id = 2 WHERE comment_id = ?", (cid,)
            )

        # Create an outlier topic (bertopic_id = -1) and a real topic
        test_db.execute(
            """INSERT INTO topics (topic_id, docket_id, bertopic_id, label, keywords, topic_size)
               VALUES (20, 'TEST-TOPICS', -1, 'Miscellaneous / Outliers', '[]', 5)"""
        )
        test_db.execute(
            """INSERT INTO topics (topic_id, docket_id, bertopic_id, label, keywords, topic_size)
               VALUES (21, 'TEST-TOPICS', 1, 'Real Topic', '["keyword"]', 10)"""
        )

        # Representative is an outlier
        test_db.execute(
            "INSERT INTO comment_topics (comment_id, topic_id, relevance_score) VALUES ('OUT-REP', 20, 0.1)"
        )
        # Member M1 already has a real topic assignment
        test_db.execute(
            "INSERT INTO comment_topics (comment_id, topic_id, relevance_score) VALUES ('OUT-M1', 21, 0.75)"
        )
        test_db.commit()

        _propagate_topics_to_groups(test_db, "TEST-TOPICS")
        test_db.commit()

        # M2 should get the real topic (from M1), not the outlier topic
        m2_row = test_db.execute(
            "SELECT topic_id FROM comment_topics WHERE comment_id = 'OUT-M2'"
        ).fetchone()
        assert m2_row is not None, "OUT-M2 should have a topic assignment"
        assert m2_row[0] == 21, "OUT-M2 should get the real topic 21, not the outlier 20"

    def test_no_propagation_without_representative_topic(self, test_db) -> None:
        """If the representative has no topic at all, members should stay unassigned."""
        _insert_comment(test_db, "NOREP", "Representative without topic.")
        _insert_comment(test_db, "NOREP-M1", "Member without topic.")

        test_db.execute(
            """INSERT INTO dedup_groups
               (dedup_group_id, docket_id, group_type, group_size, representative_comment_id)
               VALUES (3, 'TEST-TOPICS', 'exact', 2, 'NOREP')"""
        )
        for cid in ("NOREP", "NOREP-M1"):
            test_db.execute(
                "UPDATE comments SET dedup_group_id = 3 WHERE comment_id = ?", (cid,)
            )
        test_db.commit()

        _propagate_topics_to_groups(test_db, "TEST-TOPICS")
        test_db.commit()

        row = test_db.execute(
            "SELECT topic_id FROM comment_topics WHERE comment_id = 'NOREP-M1'"
        ).fetchone()
        assert row is None, "Member should have no topic when representative has none"

    def test_propagation_across_multiple_groups(self, test_db) -> None:
        """Propagation should work independently for each dedup group."""
        # Insert all comments first
        _insert_comment(test_db, "GA-REP", "Group A text.")
        _insert_comment(test_db, "GA-M1", "Group A text.")
        _insert_comment(test_db, "GB-REP", "Group B text.")
        _insert_comment(test_db, "GB-M1", "Group B text.")

        # Group A
        test_db.execute(
            """INSERT INTO dedup_groups
               (dedup_group_id, docket_id, group_type, group_size, representative_comment_id)
               VALUES (4, 'TEST-TOPICS', 'exact', 2, 'GA-REP')"""
        )
        for cid in ("GA-REP", "GA-M1"):
            test_db.execute(
                "UPDATE comments SET dedup_group_id = 4 WHERE comment_id = ?", (cid,)
            )

        # Group B
        test_db.execute(
            """INSERT INTO dedup_groups
               (dedup_group_id, docket_id, group_type, group_size, representative_comment_id)
               VALUES (5, 'TEST-TOPICS', 'exact', 2, 'GB-REP')"""
        )
        for cid in ("GB-REP", "GB-M1"):
            test_db.execute(
                "UPDATE comments SET dedup_group_id = 5 WHERE comment_id = ?", (cid,)
            )

        # Two distinct topics
        test_db.execute(
            """INSERT INTO topics (topic_id, docket_id, bertopic_id, label, keywords, topic_size)
               VALUES (30, 'TEST-TOPICS', 0, 'Topic A', '["a"]', 5)"""
        )
        test_db.execute(
            """INSERT INTO topics (topic_id, docket_id, bertopic_id, label, keywords, topic_size)
               VALUES (31, 'TEST-TOPICS', 1, 'Topic B', '["b"]', 5)"""
        )

        test_db.execute(
            "INSERT INTO comment_topics (comment_id, topic_id, relevance_score) VALUES ('GA-REP', 30, 0.9)"
        )
        test_db.execute(
            "INSERT INTO comment_topics (comment_id, topic_id, relevance_score) VALUES ('GB-REP', 31, 0.8)"
        )
        test_db.commit()

        _propagate_topics_to_groups(test_db, "TEST-TOPICS")
        test_db.commit()

        ga_m1 = test_db.execute(
            "SELECT topic_id FROM comment_topics WHERE comment_id = 'GA-M1'"
        ).fetchone()
        gb_m1 = test_db.execute(
            "SELECT topic_id FROM comment_topics WHERE comment_id = 'GB-M1'"
        ).fetchone()

        assert ga_m1[0] == 30, "Group A member should get Topic A"
        assert gb_m1[0] == 31, "Group B member should get Topic B"


# ---------------------------------------------------------------------------
# Test 3: Parameter scaling logic
# ---------------------------------------------------------------------------

class TestParameterScaling:
    """Verify HDBSCAN/UMAP parameter selection for different dataset sizes."""

    @staticmethod
    def _get_scaled_params(n_docs: int, topic_cfg: dict | None = None) -> dict:
        """Extract the parameter scaling logic from run_topics and return computed values.

        This mirrors the parameter selection code in run_topics without running
        the full BERTopic pipeline.
        """
        if topic_cfg is None:
            topic_cfg = {}

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

        return {
            "min_cluster": min_cluster,
            "min_samples": min_samples,
            "n_neighbors": n_neighbors,
            "n_components": n_components,
            "min_topic_size": min_topic_size,
        }

    def test_small_dataset_under_30(self) -> None:
        """For n<30, parameters should be minimized for tiny datasets."""
        params = self._get_scaled_params(10)
        assert params["min_cluster"] == 2
        assert params["min_samples"] == 1
        assert params["n_neighbors"] <= 9, "n_neighbors must be < n_docs"
        assert params["n_components"] <= 3
        assert params["min_topic_size"] == 2

    def test_small_dataset_edge_case_5(self) -> None:
        """For n=5 (minimum viable corpus), n_neighbors and n_components stay valid."""
        params = self._get_scaled_params(5)
        assert params["n_neighbors"] <= 4, "n_neighbors must be < n_docs (5)"
        assert params["n_components"] <= 3, "n_components must be < n_docs - 1"
        assert params["n_components"] >= 2, "n_components should be at least 2 for UMAP"

    def test_small_dataset_n_components_floor(self) -> None:
        """For n_docs <= 3, n_components should fall back to 2."""
        params = self._get_scaled_params(3)
        assert params["n_components"] == 2

    def test_medium_dataset_30_to_100(self) -> None:
        """For 30 <= n < 100, parameters should be moderately sized."""
        params = self._get_scaled_params(50)
        assert params["min_cluster"] == 3
        assert params["min_samples"] == 1
        assert params["n_neighbors"] == 8
        assert params["n_components"] == 5
        assert params["min_topic_size"] == 3

    def test_medium_dataset_edge_case_30(self) -> None:
        """At n=30 (boundary), should use medium parameters not small."""
        params = self._get_scaled_params(30)
        assert params["min_cluster"] == 3, "n=30 should hit the medium branch (>= 30)"
        assert params["min_samples"] == 1

    def test_large_dataset_100_to_500(self) -> None:
        """For 100 <= n < 500, use config defaults with sensible floors."""
        params = self._get_scaled_params(200)
        assert params["min_cluster"] >= 3
        assert params["min_samples"] == 2
        assert params["n_neighbors"] <= 12
        assert params["min_topic_size"] >= 3

    def test_very_large_dataset_over_500(self) -> None:
        """For n >= 500, use full config defaults."""
        params = self._get_scaled_params(1000)
        assert params["min_cluster"] == 10
        assert params["min_samples"] == 5
        assert params["n_neighbors"] == 15
        assert params["n_components"] == 5
        assert params["min_topic_size"] == 10

    def test_large_dataset_respects_config_overrides(self) -> None:
        """For n >= 500, user-provided config should be used directly."""
        custom_cfg = {
            "hdbscan_min_cluster_size": 20,
            "hdbscan_min_samples": 10,
            "umap_n_neighbors": 25,
            "umap_n_components": 8,
            "min_topic_size": 15,
        }
        params = self._get_scaled_params(1000, topic_cfg=custom_cfg)
        assert params["min_cluster"] == 20
        assert params["min_samples"] == 10
        assert params["n_neighbors"] == 25
        assert params["n_components"] == 8
        assert params["min_topic_size"] == 15

    def test_n_neighbors_never_exceeds_n_docs(self) -> None:
        """n_neighbors should always be strictly less than n_docs."""
        for n_docs in [5, 6, 7, 10, 15, 20, 29]:
            params = self._get_scaled_params(n_docs)
            assert params["n_neighbors"] < n_docs, (
                f"n_neighbors ({params['n_neighbors']}) must be < n_docs ({n_docs})"
            )


# ---------------------------------------------------------------------------
# Test 4: _clean_for_topics HTML stripping
# ---------------------------------------------------------------------------

class TestCleanForTopics:
    """Test that _clean_for_topics properly strips HTML tags and entities."""

    def test_strips_html_tags(self) -> None:
        """HTML tags should be removed from text."""
        result = _clean_for_topics("<p>This is a <strong>bold</strong> statement.</p>")
        assert "<p>" not in result
        assert "<strong>" not in result
        assert "</strong>" not in result
        assert "bold" in result
        assert "statement" in result

    def test_unescapes_html_entities(self) -> None:
        """HTML entities like &amp; should be converted to plain text."""
        result = _clean_for_topics("Fish &amp; Wildlife Service")
        assert "&amp;" not in result
        assert "Fish & Wildlife Service" in result

    def test_lt_gt_entities_stripped_as_tags(self) -> None:
        """&lt;...&gt; entities get unescaped to <...> then stripped as tags."""
        # This is the expected behavior: unescape first, then strip tags.
        # So &lt;Agency&gt; -> <Agency> -> "" (removed as a tag)
        result = _clean_for_topics("Before &lt;Agency&gt; After")
        assert "&lt;" not in result
        assert "&gt;" not in result
        assert "Before" in result
        assert "After" in result

    def test_br_tags_become_spaces(self) -> None:
        """<br> and <br/> tags should be converted to spaces."""
        result = _clean_for_topics("Line one<br>Line two<br/>Line three")
        assert "<br>" not in result
        assert "<br/>" not in result
        assert "Line one" in result
        assert "Line two" in result

    def test_numeric_entities(self) -> None:
        """Numeric HTML entities should be decoded."""
        result = _clean_for_topics("Copyright &#169; 2024")
        assert "&#169;" not in result
        # The copyright symbol should be present after decoding
        assert "Copyright" in result

    def test_collapses_whitespace(self) -> None:
        """Multiple whitespace characters should be collapsed to a single space."""
        result = _clean_for_topics("Hello    world   test")
        assert "  " not in result
        assert "Hello world test" == result

    def test_plain_text_unchanged(self) -> None:
        """Plain text without HTML should pass through essentially unchanged."""
        text = "This is a normal comment about environmental policy."
        result = _clean_for_topics(text)
        assert result == text

    def test_empty_string(self) -> None:
        """Empty string should return empty string."""
        assert _clean_for_topics("") == ""

    def test_complex_html_document(self) -> None:
        """A more complex HTML fragment should be fully cleaned."""
        html_text = (
            '<div class="comment">'
            '<h2>My Comment</h2>'
            '<p>I believe the rule&#39;s impact on <em>small businesses</em> '
            'is &quot;significant&quot;.</p>'
            '<ul><li>Point one</li><li>Point two</li></ul>'
            '</div>'
        )
        result = _clean_for_topics(html_text)
        assert "<" not in result
        assert ">" not in result
        assert "&quot;" not in result
        assert "&#39;" not in result
        assert "My Comment" in result
        assert "small businesses" in result
        assert "significant" in result
        assert "Point one" in result
