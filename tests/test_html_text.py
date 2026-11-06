from pathlib import Path

from newsletter.sources.html_text import Article, extract_article, html_to_text


def test_strips_tags_and_unescapes_entities():
    assert html_to_text("<p>The model is <b>open</b> &amp; small.</p>") == (
        "The model is open & small."
    )


def test_drops_script_and_style_contents():
    html = "<p>Kept.</p><script>alert('x')</script><style>p{color:red}</style>"
    assert html_to_text(html) == "Kept."


def test_block_elements_become_separate_lines():
    assert html_to_text("<h2>Title</h2><p>One</p><p>Two<br>Three</p>") == "Title\nOne\nTwo\nThree"


def test_collapses_whitespace():
    assert html_to_text("  <p>a\n\n   b\t c</p>  ") == "a b c"


def test_plain_text_passes_through():
    assert html_to_text("Plain text summary.") == "Plain text summary."


def test_empty_input():
    assert html_to_text("") == ""


PAGES = Path(__file__).parent / "fixtures" / "pages"


def page(name: str) -> str:
    return (PAGES / name).read_text()


def test_the_article_element_is_the_main_text():
    article = extract_article(page("article.html"))
    assert article.title == "Open model ships & tops the board | Example News"
    assert article.text == (
        "Open model ships\n"
        "An open-weights model was released today under the Apache 2.0 license.\n"
        "It scores 81.2 on the benchmark, ahead of last month's leader."
    )


def test_the_main_element_is_used_when_there_is_no_article():
    assert extract_article(page("main.html")).text == (
        "Version 2.0\nThe parser is twice as fast.\nNew streaming API\nDropped Python 3.9"
    )


def test_otherwise_the_largest_text_block_is_the_main_text():
    article = extract_article(page("blog.html"))
    assert article.title == "Why caches go stale"
    assert article.text == (
        "Why caches go stale\n"
        "Every cache is a bet that the past predicts the future. "
        "When the bet goes wrong, readers see yesterday's answer.\n"
        "The fix is rarely a shorter TTL. It is a clear owner for invalidation.\n"
        "Start by listing who writes the data, then who reads it.\n"
        "Unclosed paragraphs are common in hand-written HTML."
    )


def test_of_several_articles_the_longest_is_the_main_text():
    assert extract_article(page("two-articles.html")).text.startswith("Lead story\n")


def test_the_first_heading_is_the_title_when_there_is_no_title_element():
    article = extract_article(page("no-title.html"))
    assert article.title == "Heading as title"
    assert article.text == "Heading as title\nBody text with no <title> element."


def test_a_page_with_no_blocks_is_its_whole_text():
    assert extract_article("Just words, <b>no</b> blocks.") == Article("", "Just words, no blocks.")


def test_an_empty_page_has_no_text():
    assert extract_article("") == Article("", "")


def test_a_nested_article_inside_main_still_wins():
    html = "<main><p>Intro</p><article><p>Story</p></article></main>"
    assert extract_article(html).text == "Story"


def test_a_stray_end_tag_does_not_close_the_article():
    html = "<article><p>One</p></div><p>Two</p></article><p>Outside</p>"
    assert extract_article(html).text == "One\nTwo"


def test_an_unclosed_article_runs_to_the_end_of_the_page():
    assert extract_article("<p>x</p><article><p>Story</p><p>More").text == "Story\nMore"
