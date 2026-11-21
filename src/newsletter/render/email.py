"""The draft's Markdown as an HTML email preview, in the shell `issue.html.j2` uses.

The input is the draft's current text: the format `issue.md.j2` writes, possibly edited
by hand, so it is untrusted. It is parsed as CommonMark (plus the run report's tables)
with the raw-HTML rules off, so any HTML in it is shown escaped, as text. A link is
written only for a URL `safe_url` accepts — an absolute http or https URL — and anything
else is shown as its text with no link, the same rule the issue templates follow. An
image is shown as its alt text, so opening the preview fetches nothing either. Nothing
here fetches anything or sends mail.
"""

from functools import cache
from html import escape

from markdown_it import MarkdownIt
from markdown_it.token import Token
from markupsafe import Markup

from newsletter.render.render import _environments, safe_url


def render_email(markdown: str) -> str:
    """The Markdown as HTML in the issue's email shell. The page title is the first
    `# heading`, or "Newsletter" when there is none."""
    parser = _parser()
    tokens = parser.parse(markdown)
    content = parser.renderer.render(tokens, parser.options, {})
    return (
        _environments()[0]
        .get_template("email_preview.html.j2")
        .render(title=_title(tokens), content=Markup(content))
    )


@cache
def _parser() -> MarkdownIt:
    parser = MarkdownIt("commonmark").enable("table")
    # Raw HTML, inline or block, is never passed through: it renders as escaped text.
    parser.disable(["html_block", "html_inline"])
    parser.validateLink = lambda url: safe_url(url) is not None
    # A refused destination still parses as a link with an empty href; drop the anchor
    # so only its text shows, never a link to somewhere else.
    parser.core.ruler.push("drop_unsafe_links", _drop_unsafe_links)
    parser.renderer.rules["image"] = _image_as_text
    return parser


def _drop_unsafe_links(state) -> None:
    def unlink(tokens: list[Token]) -> None:
        opens: list[Token] = []
        for token in tokens:
            if token.type == "link_open":
                opens.append(token)
            elif token.type == "link_close" and opens:
                opener = opens.pop()
                if safe_url(opener.attrGet("href") or "") is None:
                    for blank in (opener, token):
                        blank.type = "text"
                        blank.tag = ""
                        blank.content = ""
                        blank.attrs = None
                        blank.children = None
            if token.children:
                unlink(token.children)

    unlink(state.tokens)


def _image_as_text(tokens: list[Token], idx: int, options, env) -> str:
    """An image as its escaped alt text: no <img>, so the viewer fetches nothing."""
    return escape(tokens[idx].content)


def _plain_text(tokens: list[Token]) -> str:
    parts = []
    for token in tokens:
        if token.type in {"text", "code_inline"}:
            parts.append(token.content)
        elif token.type in {"softbreak", "hardbreak"}:
            parts.append(" ")
        if token.children:
            parts.append(_plain_text(token.children))
    return "".join(parts)


def _title(tokens: list[Token]) -> str:
    for index, token in enumerate(tokens[:-1]):
        if token.type == "heading_open" and token.tag == "h1":
            inline = tokens[index + 1]
            if inline.type == "inline":
                return _plain_text(inline.children or []) or "Newsletter"
    return "Newsletter"
