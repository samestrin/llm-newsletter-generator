"""One connector per email platform, each creating a draft and never sending one.

Each connector makes a single create call whose body asks for a draft, and nothing else:
no send call, no schedule. A connector is set up only when its key is in the
environment, so each one is optional.

Environment:

    BUTTONDOWN_API_KEY          Buttondown: `POST /v1/emails` with `status: draft`
    BEEHIIV_API_KEY             Beehiiv: `POST /v2/publications/{id}/posts` with
    BEEHIIV_PUBLICATION_ID        `status: draft`
    RESEND_API_KEY              Resend: `POST /broadcasts` with `send: false`; the
    RESEND_SEGMENT_ID             `/broadcasts/{id}/send` call is never made
    RESEND_FROM                   (the sender, e.g. `Name <you@example.org>`)
    NEWSLETTER_EXPORT_TIMEOUT   seconds per request, default 30
"""

import html
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

from newsletter.llm import ConfigError, _number

DEFAULT_TIMEOUT = 30.0
DRAFT = "draft"


class ExportError(RuntimeError):
    """The platform did not confirm a draft. The message never holds a key."""


@dataclass(frozen=True)
class Connector:
    """A platform's draft endpoint: the request to make, and how to read its answer."""

    name: ClassVar[str]
    draft_only: ClassVar[bool] = True
    """False for a platform with no endpoint that only creates a draft; it is dropped."""

    api_key: str = field(repr=False)
    timeout: float = DEFAULT_TIMEOUT

    def request(self, title: str, text: str) -> tuple[str, dict[str, str], dict[str, Any]]:
        """The URL, headers and JSON body of the one create call."""
        raise NotImplementedError

    def draft_id(self, data: Any) -> str:
        """The new draft's id from the answer; raises ExportError if it is not a draft."""
        raise NotImplementedError


@dataclass(frozen=True)
class Buttondown(Connector):
    name: ClassVar[str] = "buttondown"
    base_url: str = "https://api.buttondown.com/v1"

    def request(self, title, text):
        headers = {"Authorization": f"Token {self.api_key}"}
        return f"{self.base_url}/emails", headers, {"subject": title, "body": text, "status": DRAFT}

    def draft_id(self, data):
        return _checked(data if isinstance(data, dict) else {})


@dataclass(frozen=True)
class Beehiiv(Connector):
    name: ClassVar[str] = "beehiiv"
    publication_id: str = ""
    base_url: str = "https://api.beehiiv.com/v2"

    def request(self, title, text):
        url = f"{self.base_url}/publications/{self.publication_id}/posts"
        body = {"title": title, "body_content": to_html(text), "status": DRAFT}
        return url, {"Authorization": f"Bearer {self.api_key}"}, body

    def draft_id(self, data):
        post = data.get("data") if isinstance(data, dict) else None
        return _checked(post if isinstance(post, dict) else {})


@dataclass(frozen=True)
class Resend(Connector):
    name: ClassVar[str] = "resend"
    segment_id: str = ""
    sender: str = ""
    base_url: str = "https://api.resend.com"

    def request(self, title, text):
        body = {
            "segment_id": self.segment_id,
            "from": self.sender,
            "subject": title,
            "name": title,
            "text": text,
            "html": to_html(text),
            # Resend creates a draft unless told to send; say so rather than rely on it.
            "send": False,
        }
        return f"{self.base_url}/broadcasts", {"Authorization": f"Bearer {self.api_key}"}, body

    def draft_id(self, data):
        return _checked(data if isinstance(data, dict) else {})


def _checked(created: dict[str, Any]) -> str:
    status = created.get("status")
    if status is not None and status != DRAFT:
        raise ExportError(f"the platform reports status {status!r}, not {DRAFT!r}")
    draft_id = created.get("id")
    if not isinstance(draft_id, str) or not draft_id:
        raise ExportError("no draft id in the response")
    return draft_id


def to_html(text: str) -> str:
    """The text as escaped HTML paragraphs, so no feed or model text becomes markup."""
    paragraphs = [p.strip() for p in text.replace("\r\n", "\n").split("\n\n") if p.strip()]
    return "\n".join(f"<p>{html.escape(p).replace(chr(10), '<br>')}</p>" for p in paragraphs)


def connectors_from_env(env: Mapping[str, str] | None = None) -> list[Connector]:
    """The connectors whose key is set, in a fixed order; raises ConfigError if one is
    missing a setting it needs."""
    env = os.environ if env is None else env
    timeout = _number(env, "NEWSLETTER_EXPORT_TIMEOUT", float, minimum=0, exclusive=True)
    timeout = DEFAULT_TIMEOUT if timeout is None else timeout
    connectors: list[Connector] = []
    if key := env.get("BUTTONDOWN_API_KEY"):
        connectors.append(Buttondown(api_key=key, timeout=timeout))
    if key := env.get("BEEHIIV_API_KEY"):
        publication = _required(env, "BEEHIIV_PUBLICATION_ID", "BEEHIIV_API_KEY")
        connectors.append(Beehiiv(api_key=key, timeout=timeout, publication_id=publication))
    if key := env.get("RESEND_API_KEY"):
        segment = _required(env, "RESEND_SEGMENT_ID", "RESEND_API_KEY")
        sender = _required(env, "RESEND_FROM", "RESEND_API_KEY")
        connectors.append(Resend(api_key=key, timeout=timeout, segment_id=segment, sender=sender))
    return connectors


def _required(env: Mapping[str, str], name: str, because: str) -> str:
    value = (env.get(name) or "").strip()
    if not value:
        raise ConfigError(f"{name} must be set when {because} is")
    return value
