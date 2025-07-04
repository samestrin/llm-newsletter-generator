"""Test configuration and fixtures for the newsletter generator tests."""

import pytest
import os
from unittest.mock import MagicMock, patch
from llm_newsletter_generator.llm_newsletter_generator import NewsletterGenerator


@pytest.fixture
def sample_feed_content():
    """Reads and returns the content of the sample RSS feed."""
    with open("tests/fixtures/sample_feed.xml", "r") as f:
        return f.read()


@pytest.fixture
def mock_tokenizer():
    """Returns a mock tokenizer for testing."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.side_effect = lambda x, add_special_tokens=False: [1] * len(x)
    return mock_tokenizer


@pytest.fixture
def mock_text_generation_pipeline():
    """Returns a mock text generation pipeline."""
    mock_pipeline = MagicMock()
    mock_pipeline.return_value = [{"generated_text": "prompt + generated content"}]
    return mock_pipeline


@pytest.fixture
def mock_summarizer_pipeline():
    """Returns a mock summarization pipeline."""
    mock_summarizer = MagicMock()
    mock_summarizer.return_value = [{"summary_text": "Summarized content"}]
    return mock_summarizer


@pytest.fixture
def mocked_generator(mock_tokenizer, mock_text_generation_pipeline, mock_summarizer_pipeline):
    """Returns a fully mocked NewsletterGenerator instance."""
    with (
        patch(
            "llm_newsletter_generator.llm_newsletter_generator.AutoTokenizer.from_pretrained",
            return_value=mock_tokenizer,
        ),
        patch(
            "llm_newsletter_generator.llm_newsletter_generator.pipeline",
            side_effect=[mock_text_generation_pipeline, mock_summarizer_pipeline],
        ),
    ):
        generator = NewsletterGenerator(feed_url="http://example.com/rss")
        yield generator


@pytest.fixture
def sample_feed_items():
    """Returns sample feed items for testing."""
    return [
        ("First Article", "Description of first article", "http://example.com/1"),
        ("Second Article", "Description of second article", "http://example.com/2"),
        ("Third Article", "Description of third article", "http://example.com/3"),
    ]


@pytest.fixture
def large_feed_items():
    """Returns a large set of feed items for performance testing."""
    items = []
    for i in range(100):
        items.append(
            (
                f"Article {i}",
                f"This is the description for article {i}. " * 10,
                f"http://example.com/article/{i}",
            )
        )
    return items


@pytest.fixture
def sample_rss_feed():
    """Returns a sample RSS feed XML content."""
    return """
    <?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
    <channel>
        <title>Test Feed</title>
        <link>http://example.com/</link>
        <description>A test feed for unit tests.</description>
        <item>
            <title>Breaking News</title>
            <link>http://example.com/breaking</link>
            <description>This is breaking news content.</description>
            <pubDate>Mon, 01 Jan 2024 12:00:00 GMT</pubDate>
            <category>News</category>
            <author>Test Author</author>
        </item>
        <item>
            <title>Technology Update</title>
            <link>http://example.com/tech</link>
            <description>Latest technology developments.</description>
            <pubDate>Mon, 01 Jan 2024 13:00:00 GMT</pubDate>
            <category>Technology</category>
            <author>Tech Writer</author>
        </item>
    </channel>
    </rss>
    """


@pytest.fixture
def malformed_rss_feed():
    """Returns a malformed RSS feed for error testing."""
    return """
    <?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
    <channel>
        <title>Malformed Feed</title>
        <item>
            <title>Incomplete Item</title>
            <!-- Missing description and link -->
        </item>
        <item>
            <!-- Missing title -->
            <description>Description without title</description>
            <link>http://example.com/notitle</link>
        </item>
    </channel>
    <!-- Missing closing rss tag -->
    """


@pytest.fixture
def unicode_rss_feed():
    """Returns an RSS feed with unicode content."""
    return """
    <?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
    <channel>
        <title>Unicode Test Feed 🌍</title>
        <link>http://example.com/</link>
        <description>Feed with unicode characters</description>
        <item>
            <title>Article with émojis 🚀</title>
            <link>http://example.com/unicode</link>
            <description>Content with special characters: àáâãäå ñ ç</description>
            <pubDate>Mon, 01 Jan 2024 12:00:00 GMT</pubDate>
            <category>Unicode</category>
        </item>
    </channel>
    </rss>
    """


@pytest.fixture
def mock_requests_response():
    """Returns a mock requests response object."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "<rss><channel><title>Mock Feed</title></channel></rss>"
    return mock_response


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Returns a temporary cache directory for testing."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return str(cache_dir)


@pytest.fixture
def mock_file_operations():
    """Returns mock file operation context managers."""

    class MockFileOperations:
        def __init__(self):
            self.written_files = {}
            self.read_files = {}

        def mock_open_write(self, filename, mode):
            if "w" in mode:
                mock_file = MagicMock()
                mock_file.write = lambda content: self.written_files.update({filename: content})
                return mock_file

        def mock_open_read(self, filename, mode):
            if "r" in mode and filename in self.read_files:
                mock_file = MagicMock()
                mock_file.read = lambda: self.read_files[filename]
                return mock_file
            raise FileNotFoundError(f"File {filename} not found")

        def set_file_content(self, filename, content):
            self.read_files[filename] = content

        def get_written_content(self, filename):
            return self.written_files.get(filename)

    return MockFileOperations()


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment before each test."""
    # Ensure cache directory exists for tests
    os.makedirs("./cache/", exist_ok=True)
    yield
    # Cleanup can be added here if needed


@pytest.fixture
def performance_threshold():
    """Returns performance thresholds for testing."""
    return {
        "initialization_time": 1.0,  # seconds
        "feed_parsing_time": 0.1,  # seconds per 100 items
        "text_generation_time": 0.1,  # seconds
        "cache_read_time": 0.01,  # seconds
        "cache_write_time": 0.05,  # seconds
        "memory_increase_limit": 50,  # MB
    }


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "performance: marks tests as performance benchmarks")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add markers based on test file names
        if "test_performance" in item.nodeid:
            item.add_marker(pytest.mark.performance)
        if "test_integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        if (
            "test_newsletter_generator" in item.nodeid
            or "test_feed_processing" in item.nodeid
            or "test_content_generation" in item.nodeid
        ):
            item.add_marker(pytest.mark.unit)

        # Mark slow tests
        if "large_feed" in item.name or "performance" in item.name or "benchmark" in item.name:
            item.add_marker(pytest.mark.slow)
