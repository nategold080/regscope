"""Tests for topic modeling pipeline."""

import pytest


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
