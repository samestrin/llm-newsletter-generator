import pytest
from unittest.mock import patch, mock_open
from llm_newsletter_generator.llm_newsletter_generator import NewsletterGenerator

@pytest.fixture
def generator():
    """Returns a NewsletterGenerator instance for testing."""
    return NewsletterGenerator(feed_url="http://example.com/rss")

def test_newsletter_generator_instantiation(generator):
    """Tests if the NewsletterGenerator class can be instantiated."""
    assert generator is not None
