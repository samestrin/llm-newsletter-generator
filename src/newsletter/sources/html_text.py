"""Turn the HTML found in feed entries into plain text, and find a web page's main text."""

import re
from dataclasses import dataclass
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


# Elements with no end tag, so never open.
_VOID = {
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source",
    "track", "wbr",
}  # fmt: skip

# Elements that hold an article's paragraphs, and the paragraphs that score them.
_CONTAINER = {"article", "blockquote", "body", "div", "main", "section", "td"}
_PARAGRAPH = {"p", "pre"}


@dataclass(frozen=True)
class Article:
    title: str
    """The page's <title>, else its first <h1>; empty when it has neither."""
    text: str
    """The main text, as `html_to_text` gives it."""


@dataclass
class _Element:
    tag: str
    start: int
    """Offset in the page just after the start tag."""
    end: int = 0
    """Offset of the end tag (or of the end of the page, when it never closes)."""
    text: int = 0
    """Characters of visible text inside."""
    score: int = 0
    """Characters of text in the paragraphs whose nearest container this is."""


class _Outline(HTMLParser):
    """Records where each element starts and ends in the page and how much text it holds."""

    def __init__(self, html: str) -> None:
        super().__init__(convert_charrefs=True)
        self.html = html
        self.line_starts = [0]
        for line in html.split("\n")[:-1]:
            self.line_starts.append(self.line_starts[-1] + len(line) + 1)
        self.stack: list[_Element] = []
        self.closed: list[_Element] = []
        self.skip_depth = 0
        self.title: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP:
            self.skip_depth += 1
            return
        if tag in _VOID:
            return
        if tag in _BLOCK and self.stack and self.stack[-1].tag == "p":
            # A block element ends an open paragraph, as browsers do with "<p>a<p>b".
            self._close(len(self.stack) - 1, self._offset())
        start = self._offset() + len(self.get_starttag_text() or "")
        self.stack.append(_Element(tag, start))

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP:
            self.skip_depth = max(0, self.skip_depth - 1)
            return
        # A stray end tag with nothing open to match it is ignored.
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index].tag == tag:
                self._close(index, self._offset())
                return

    def handle_data(self, data: str) -> None:
        if self.skip_depth or not self.stack:
            return
        if self.stack[-1].tag == "title":
            self.title.append(data)
        self.stack[-1].text += len(" ".join(data.split()))

    def close(self) -> None:
        super().close()
        if self.stack:
            self._close(0, len(self.html))

    def _close(self, index: int, end: int) -> None:
        """Close the element at `index` and every element still open inside it."""
        while len(self.stack) > index:
            element = self.stack.pop()
            element.end = end
            self.closed.append(element)
            if self.stack:
                self.stack[-1].text += element.text
            if element.tag in _PARAGRAPH:
                container = next((e for e in reversed(self.stack) if e.tag in _CONTAINER), None)
                if container:
                    container.score += element.text

    def _offset(self) -> int:
        line, column = self.getpos()
        return self.line_starts[line - 1] + column

    def inner(self, element: _Element) -> str:
        return self.html[element.start : element.end]


def extract_article(html: str) -> Article:
    """The page's title and main text: its longest `article`, else its longest `main`, else
    the element holding the most paragraph text, else the whole page."""
    outline = _Outline(html)
    outline.feed(html)
    outline.close()
    elements = outline.closed

    main = _longest(e for e in elements if e.tag == "article") or _longest(
        e for e in elements if e.tag == "main"
    )
    if main is None:
        scored = [e for e in elements if e.score]
        main = max(scored, key=lambda e: e.score) if scored else None
    text = html_to_text(outline.inner(main) if main else html)

    title = " ".join("".join(outline.title).split())
    if not title:
        heading = next((e for e in sorted(elements, key=lambda e: e.start) if e.tag == "h1"), None)
        title = " ".join(html_to_text(outline.inner(heading)).split()) if heading else ""
    return Article(title, text)


def _longest(elements) -> _Element | None:
    found = list(elements)
    # max() keeps the first of equals, but `closed` lists inner elements first: sort by start.
    found.sort(key=lambda e: e.start)
    return max(found, key=lambda e: e.text) if found else None
