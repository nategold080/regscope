"""Tests for Regulations.gov API client and response parsing."""

import json
from pathlib import Path

import pytest

# Load sample comments fixture
FIXTURES_DIR = Path(__file__).parent / "fixtures"


def load_sample_comments() -> list[dict]:
    """Load sample comments from fixture file."""
    with open(FIXTURES_DIR / "sample_comments.json") as f:
        return json.load(f)


class TestAPIResponseParsing:
    """Test parsing of API response structures."""

    def test_parse_comment_attributes(self) -> None:
        """Test that comment attributes are correctly extracted."""
        comments = load_sample_comments()
        comment = comments[0]

        assert comment["id"] == "TEST-COMMENT-001"
        attrs = comment["attributes"]
        assert attrs["title"] == "Comment from John Doe"
        assert attrs["firstName"] == "John"
        assert attrs["lastName"] == "Doe"
        assert attrs["organization"] == ""
        assert "support" in attrs["comment"].lower()

    def test_parse_all_fixtures(self) -> None:
        """Test that all fixture comments parse without errors."""
        comments = load_sample_comments()
        assert len(comments) == 8

        for comment in comments:
            assert "id" in comment
            assert "attributes" in comment
            attrs = comment["attributes"]
            assert "comment" in attrs
            assert "firstName" in attrs
            assert "lastName" in attrs

    def test_comment_with_organization(self) -> None:
        """Test parsing comments that have organizations."""
        comments = load_sample_comments()
        acme = next(c for c in comments if c["id"] == "TEST-COMMENT-003")
        assert acme["attributes"]["organization"] == "Acme Manufacturing Corp."

    def test_list_response_structure(self) -> None:
        """Test the expected structure of a list API response."""
        # Simulate the list response wrapper
        comments = load_sample_comments()
        response = {
            "data": comments[:3],
            "meta": {"totalElements": 3},
        }
        assert len(response["data"]) == 3
        assert response["meta"]["totalElements"] == 3

    def test_detail_response_structure(self) -> None:
        """Test the expected structure of a detail API response."""
        comments = load_sample_comments()
        comment = comments[3]  # EDF with attachments mention
        response = {
            "data": comment,
            "included": [
                {
                    "id": "ATT-001",
                    "type": "attachments",
                    "attributes": {
                        "fileUrl": "https://example.com/file.pdf",
                        "format": "pdf",
                        "title": "Technical Analysis",
                        "size": 1024000,
                    },
                }
            ],
        }

        assert response["data"]["id"] == "TEST-COMMENT-004"
        assert len(response["included"]) == 1
        assert response["included"][0]["attributes"]["format"] == "pdf"
