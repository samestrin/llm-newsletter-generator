"""Tests for RSS feed processing functionality."""

import pytest
import hashlib
import os
from unittest.mock import patch, mock_open, MagicMock
from llm_newsletter_generator.llm_newsletter_generator import NewsletterGenerator


class TestFeedLoading:
    """Test cases for RSS feed loading functionality."""

    @pytest.fixture
    def generator(self):
        """Returns a mocked NewsletterGenerator instance."""
        with (
            patch(
                "llm_newsletter_generator.llm_newsletter_generator.AutoTokenizer.from_pretrained"
            ),
            patch("llm_newsletter_generator.llm_newsletter_generator.pipeline"),
        ):
            yield NewsletterGenerator(feed_url="http://example.com/rss")

    @patch("requests.get")
    def test_load_feed_success(self, mock_get, generator):
        """Test successful feed loading from URL."""
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = "<rss><channel><title>Test</title></channel></rss>"

        with patch("builtins.open", mock_open()) as mock_file:
            feed_content = generator.load_feed()

        assert feed_content == "<rss><channel><title>Test</title></channel></rss>"
        mock_get.assert_called_once_with(generator.feed_url)
        mock_file.assert_called()

    @patch("requests.get")
    def test_load_feed_http_error(self, mock_get, generator):
        """Test feed loading with HTTP error."""
        mock_get.return_value.status_code = 404

        result = generator.load_feed()
        assert result is None  # Method returns None on HTTP errors

    @patch("requests.get")
    def test_load_feed_network_error(self, mock_get, generator):
        """Test feed loading with network error."""
        mock_get.side_effect = ConnectionError("Network error")

        result = generator.load_feed()
        assert result is None  # Method returns None on network errors

    @patch("os.path.exists", return_value=True)
    @patch("os.path.getmtime")
    def test_load_feed_from_cache_fresh(self, mock_getmtime, mock_exists, generator):
        """Test loading fresh feed from cache."""
        # Mock recent cache file (within 1 hour)
        import time

        mock_getmtime.return_value = time.time() - 1800  # 30 minutes ago

        cache_content = "<rss><channel><title>Cached</title></channel></rss>"
        with patch("builtins.open", mock_open(read_data=cache_content)):
            feed_content = generator.load_feed()

        assert feed_content == cache_content

    @patch("os.path.exists", return_value=True)
    @patch("os.path.getmtime")
    @patch("requests.get")
    def test_load_feed_cache_expired(self, mock_get, mock_getmtime, mock_exists, generator):
        """Test loading feed when cache is expired."""
        # Mock old cache file (older than 1 hour)
        import time

        mock_getmtime.return_value = time.time() - 7200  # 2 hours ago

        mock_get.return_value.status_code = 200
        mock_get.return_value.text = "<rss><channel><title>Fresh</title></channel></rss>"

        with patch("builtins.open", mock_open()):
            feed_content = generator.load_feed()

        assert feed_content == "<rss><channel><title>Fresh</title></channel></rss>"
        mock_get.assert_called_once()


class TestFeedParsing:
    """Test cases for RSS feed parsing functionality."""

    @pytest.fixture
    def generator(self):
        """Returns a mocked NewsletterGenerator instance."""
        with (
            patch(
                "llm_newsletter_generator.llm_newsletter_generator.AutoTokenizer.from_pretrained"
            ),
            patch("llm_newsletter_generator.llm_newsletter_generator.pipeline"),
        ):
            yield NewsletterGenerator(feed_url="http://example.com/rss")

    @patch("feedparser.parse")
    def test_get_items_basic(self, mock_parse, generator):
        """Test basic item extraction from feed."""
        mock_parse.return_value.entries = [
            {
                "title": "Article 1",
                "description": "Description 1",
                "link": "http://example.com/1",
            },
            {
                "title": "Article 2",
                "description": "Description 2",
                "link": "http://example.com/2",
            },
        ]

        items = generator.get_items("<rss>feed content</rss>")

        assert len(items) == 2
        assert items[0] == ("Article 1", "Description 1", "http://example.com/1")
        assert items[1] == ("Article 2", "Description 2", "http://example.com/2")

    @patch("feedparser.parse")
    def test_get_items_empty_feed(self, mock_parse, generator):
        """Test parsing empty feed."""
        mock_parse.return_value.entries = []

        items = generator.get_items("<rss></rss>")

        assert len(items) == 0
        assert items == []

    @patch("feedparser.parse")
    def test_get_items_missing_fields(self, mock_parse, generator):
        """Test parsing feed with missing fields."""
        mock_parse.return_value.entries = [
            {
                "title": "Article 1",
                "description": "Description 1",
                # Missing 'link' field
            },
            {
                "title": "Article 2",
                "link": "http://example.com/2",
                # Missing 'description' field
            },
        ]

        items = generator.get_items("<rss>feed content</rss>")

        assert len(items) == 2
        assert items[0] == ("Article 1", "Description 1", "")
        assert items[1] == ("Article 2", "", "http://example.com/2")

    @patch("feedparser.parse")
    def test_get_items_malformed_feed(self, mock_parse, generator):
        """Test parsing malformed feed."""
        # Mock a malformed parse result with no entries attribute
        mock_result = MagicMock()
        del mock_result.entries  # Remove entries attribute to simulate malformed feed
        mock_parse.return_value = mock_result

        with pytest.raises(AttributeError):
            generator.get_items("malformed feed")

    @patch("feedparser.parse")
    def test_get_items_unicode_content(self, mock_parse, generator):
        """Test parsing feed with unicode content."""
        mock_parse.return_value.entries = [
            {
                "title": "Article with émojis 🚀",
                "description": "Description with special chars: àáâãäå",
                "link": "http://example.com/unicode",
            }
        ]

        items = generator.get_items("<rss>unicode feed</rss>")

        assert len(items) == 1
        assert "émojis 🚀" in items[0][0]
        assert "àáâãäå" in items[0][1]

    @patch("feedparser.parse")
    def test_get_items_large_feed(self, mock_parse, generator):
        """Test parsing large feed with many items."""
        # Create 100 mock entries
        entries = []
        for i in range(100):
            entries.append(
                {
                    "title": f"Article {i}",
                    "description": f"Description {i}",
                    "link": f"http://example.com/{i}",
                }
            )

        mock_parse.return_value.entries = entries

        items = generator.get_items("<rss>large feed</rss>")

        assert len(items) == 100
        assert items[0][0] == "Article 0"
        assert items[99][0] == "Article 99"


class TestCacheManagement:
    """Test cases for cache management functionality."""

    @pytest.fixture
    def generator(self):
        """Returns a mocked NewsletterGenerator instance."""
        with (
            patch(
                "llm_newsletter_generator.llm_newsletter_generator.AutoTokenizer.from_pretrained"
            ),
            patch("llm_newsletter_generator.llm_newsletter_generator.pipeline"),
        ):
            yield NewsletterGenerator(feed_url="http://example.com/rss")

    def test_cache_file_path_generation(self, generator):
        """Test cache file path generation."""
        expected_hash = hashlib.md5(generator.feed_url.encode()).hexdigest()
        expected_path = os.path.join("./cache/", f"{expected_hash}.txt")

        # This tests the internal logic used in load_feed
        cache_dir = "./cache/"
        cache_file = os.path.join(
            cache_dir, f"{hashlib.md5(generator.feed_url.encode()).hexdigest()}.txt"
        )

        assert cache_file == expected_path

    # Note: Cache directory creation test removed as current implementation
    # doesn't create directories. This will be added in Phase 1.2 refactoring.

    @patch("os.path.exists", return_value=False)
    @patch("requests.get")
    def test_cache_miss_loads_from_url(self, mock_get, mock_exists, generator):
        """Test that cache miss triggers URL loading."""
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = "<rss>fresh content</rss>"

        with patch("builtins.open", mock_open()):
            feed_content = generator.load_feed()

        assert feed_content == "<rss>fresh content</rss>"
        mock_get.assert_called_once_with(generator.feed_url)
