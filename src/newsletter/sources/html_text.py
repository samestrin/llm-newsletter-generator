"""Turn the HTML found in feed entries into plain text."""

import re
from html.parser import HTMLParser

# Elements whose contents are never text a reader should see.
_SKIP = {"script", "style", "template", "noscript", "iframe", "svg"}

# Elements that start a new line of text.
_BLOCK = {
    "address", "article", "aside", "blockquote", "br", "dd", "div", "dl", "dt", "figcaption",
    "figure", "footer", "h1", "h2", "h3", "h4", "h5", "h6", "header", "hr", "li", "ol", "p",
    "pre", "section", "table", "td", "th", "tr", "ul",
}  # fmt: skip


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP:
            self.skip_depth += 1
        elif tag in _BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP:
            self.skip_depth = max(0, self.skip_depth - 1)
        elif tag in _BLOCK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.skip_depth:
            # Newlines in the source are just whitespace; only block elements break lines.
            self.parts.append(re.sub(r"\s+", " ", data))


def html_to_text(html: str) -> str:
    """Return the visible text of `html`: one line per block, whitespace collapsed."""
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    lines = (" ".join(line.split()) for line in "".join(parser.parts).split("\n"))
    return "\n".join(line for line in lines if line)
