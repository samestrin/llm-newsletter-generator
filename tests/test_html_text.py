from newsletter.sources.html_text import html_to_text


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
