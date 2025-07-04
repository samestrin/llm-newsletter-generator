import pytest
from unittest.mock import patch
from llm_newsletter_generator.llm_newsletter_generator import NewsletterGenerator


def test_integration_load_and_parse_feed():
    """Tests loading and parsing a real RSS feed."""
    feed_url = "http://feeds.arstechnica.com/arstechnica/index/"
    with patch("llm_newsletter_generator.llm_newsletter_generator.AutoTokenizer.from_pretrained"), \
         patch("llm_newsletter_generator.llm_newsletter_generator.pipeline") as mock_pipeline:
        generator = NewsletterGenerator(feed_url=feed_url)
        feed_content = generator.load_feed()
        assert feed_content is not None
        items = generator.get_items(feed_content)
        assert len(items) > 0

