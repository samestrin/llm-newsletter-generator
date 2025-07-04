import pytest

@pytest.fixture
def sample_feed_content():
    """Reads and returns the content of the sample RSS feed."""
    with open("tests/fixtures/sample_feed.xml", "r") as f:
        return f.read()
