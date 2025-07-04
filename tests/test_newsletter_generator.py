import pytest
import os
import hashlib
from unittest.mock import patch, mock_open, MagicMock, call
from llm_newsletter_generator.llm_newsletter_generator import NewsletterGenerator

@pytest.fixture
def generator():
    """Returns a NewsletterGenerator instance for testing."""
    with patch("llm_newsletter_generator.llm_newsletter_generator.AutoTokenizer.from_pretrained") as mock_tokenizer_from_pretrained, \
         patch("llm_newsletter_generator.llm_newsletter_generator.pipeline") as mock_pipeline:
        # Configure the mock tokenizer to return a mock object with an encode method
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.encode.side_effect = lambda x, add_special_tokens=False: [1] * len(x) # Default for generate_prompt
        mock_tokenizer_from_pretrained.return_value = mock_tokenizer_instance

        yield NewsletterGenerator(feed_url="http://example.com/rss")

def test_newsletter_generator_instantiation(generator):
    """Tests if the NewsletterGenerator class can be instantiated."""
    assert generator is not None

@patch("llm_newsletter_generator.llm_newsletter_generator.AutoTokenizer.from_pretrained")
@patch("llm_newsletter_generator.llm_newsletter_generator.pipeline")
def test_newsletter_generator_init_model_configs(mock_pipeline, mock_tokenizer_from_pretrained):
    """Tests if the NewsletterGenerator initializes with correct model configs."""
    generator = NewsletterGenerator(feed_url="http://example.com/rss", model_name="mistral")
    assert generator.model == "mistralai/Mistral-7B-Instruct-v0.2"

    # Check calls to AutoTokenizer.from_pretrained
    mock_tokenizer_from_pretrained.assert_called_once_with("mistralai/Mistral-7B-Instruct-v0.2")

    # Check calls to pipeline
    expected_calls = [
        call("text-generation", model="mistralai/Mistral-7B-Instruct-v0.2", tokenizer=mock_tokenizer_from_pretrained.return_value, trust_remote_code=True),
        call("summarization", model="sshleifer/distilbart-cnn-12-6")
    ]
    mock_pipeline.assert_has_calls(expected_calls, any_order=True)

@patch("requests.get")
def test_load_feed_from_url(mock_get, generator):
    """
    Tests loading the feed from a URL.
    """
    mock_get.return_value.status_code = 200
    mock_get.return_value.text = "<rss></rss>"
    cache_dir = "./cache/"
    expected_cache_file = os.path.join(
        cache_dir, f"{hashlib.md5(generator.feed_url.encode()).hexdigest()}.txt"
    )
    with patch("builtins.open", mock_open()) as mock_file:
        feed_content = generator.load_feed()
        assert feed_content == "<rss></rss>"
        mock_file.assert_called_with(expected_cache_file, "w")

@patch("os.path.exists", return_value=True)
@patch("os.path.getmtime", return_value=9999999999)
def test_load_feed_from_cache(mock_getmtime, mock_exists, generator):
    """Tests loading the feed from the cache."""
    cache_dir = "./cache/"
    expected_cache_file = os.path.join(
        cache_dir, f"{hashlib.md5(generator.feed_url.encode()).hexdigest()}.txt"
    )
    with patch("builtins.open", mock_open(read_data="<rss></rss>")) as mock_file:
        feed_content = generator.load_feed()
        assert feed_content == "<rss></rss>"
        mock_file.assert_called_with(expected_cache_file, "r")

@patch("feedparser.parse")
def test_get_items(mock_feedparser_parse, generator, sample_feed_content):
    """Tests parsing feed items."""
    mock_feedparser_parse.return_value.entries = [
        {"title": "Test Article 1", "description": "Desc 1", "link": "Link 1"},
        {"title": "Test Article 2", "description": "Desc 2", "link": "Link 2"},
    ]
    items = generator.get_items(sample_feed_content)
    assert len(items) == 2
    assert items[0][0] == "Test Article 1"
    mock_feedparser_parse.assert_called_once_with(sample_feed_content)

def test_generate_text(generator):
    """
    Tests text generation.
    """
    prompt = "Test prompt"
    cache_dir = "./cache/"
    cache_key = generator.model + " " + prompt
    expected_cache_file = os.path.join(
        cache_dir, f"{hashlib.md5(cache_key.encode()).hexdigest()}.txt"
    )
    with patch.object(generator, 'text_generation', return_value=[{"generated_text": "Test promptGenerated text"}]) as mock_text_generation, \
         patch("builtins.open", mock_open()) as mock_file:
        generated_text = generator.generate_text(prompt)
        assert generated_text == "Generated text"
        mock_text_generation.assert_called_with(prompt, max_new_tokens=2046, do_sample=True)
        mock_file.assert_called_with(expected_cache_file, "w")

def test_load_template(generator):
    """Tests loading template content from a file."""
    template_path = "prompts/introduction.md"
    with patch("builtins.open", mock_open(read_data="Template content")) as mock_file:
        content = generator.load_template(template_path)
        assert content == "Template content"
        mock_file.assert_called_with(template_path, "r")

@patch("llm_newsletter_generator.llm_newsletter_generator.os.path.join", return_value="prompts/introduction.md")
@patch("llm_newsletter_generator.llm_newsletter_generator.NewsletterGenerator.load_template", return_value="{{ title }} {{ topic }} {{ row_titles }}")
def test_generate_prompt(mock_load_template, mock_os_path_join, generator):
    """Tests prompt generation for introduction/closing sections."""
    generator.tokenizer.encode.side_effect = lambda x, add_special_tokens=False: [1] * len(x)

    title = "Newsletter Title"
    topic = "Technology"
    row_titles = ["Article 1", "Article 2"]
    section = "introduction"
    prompt = generator.generate_prompt(title, topic, row_titles, section)
    assert "Newsletter Title" in prompt
    assert "Technology" in prompt
    assert "Article 1" in prompt
    assert "Article 2" in prompt

@patch("llm_newsletter_generator.llm_newsletter_generator.os.path.join", return_value="prompts/item.md")
@patch("llm_newsletter_generator.llm_newsletter_generator.NewsletterGenerator.load_template", return_value="{{ item_title }} {{ item_description }} {{ topic }}")
@patch("llm_newsletter_generator.llm_newsletter_generator.BeautifulSoup")
def test_generate_prompt_for_item(mock_beautifulsoup, mock_load_template, mock_os_path_join, generator):
    """Tests prompt generation for individual news items."""
    generator.tokenizer.encode.side_effect = lambda x: [1] * len(x)
    generator.summarizer.return_value = [{'summary_text': 'Summarized description'}]

    mock_beautifulsoup.return_value.get_text.return_value = "Cleaned description"

    item = ("Item Title", "<p>Item Description</p>", "http://example.com/item")
    topic = "Technology"
    prompt = generator.generate_prompt_for_item(item, topic)
    assert "Item Title" in prompt
    assert "Cleaned description" in prompt or "Summarized description" in prompt
    assert "Technology" in prompt

@patch.object(NewsletterGenerator, 'generate_prompt')
@patch.object(NewsletterGenerator, 'generate_prompt_for_item')
@patch.object(NewsletterGenerator, 'generate_text')
@patch("llm_newsletter_generator.llm_newsletter_generator.Progress")
def test_create_newsletter(mock_progress, mock_generate_text, mock_generate_prompt_for_item, mock_generate_prompt, generator):
    """
    Tests the end-to-end newsletter creation process.
    """
    mock_generate_text.side_effect = ["Intro", "Story 1", "Story 2", "Closing"]
    mock_generate_prompt.side_effect = ["Intro Prompt", "Closing Prompt"]
    mock_generate_prompt_for_item.side_effect = ["Story 1 Prompt", "Story 2 Prompt"]

    items = [
        ("Title 1", "Desc 1", "Link 1"),
        ("Title 2", "Desc 2", "Link 2"),
    ]
    newsletter = generator.create_newsletter("Test Title", "Test Topic", items)

    assert "Intro" in newsletter
    assert "Story 1" in newsletter
    assert "Story 2" in newsletter
    assert "Closing" in newsletter
    assert mock_generate_prompt.call_count == 2
    assert mock_generate_prompt_for_item.call_count == 2
    assert mock_generate_text.call_count == 4
    mock_progress.assert_called_once()
