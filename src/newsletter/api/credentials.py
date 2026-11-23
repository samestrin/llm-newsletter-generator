"""The model a request may use: a provider from `providers.json` and the visitor's key or
a demo code. The header contract is `llm-pdf-chatbot`'s (`backend/app/credentials.py`).

A request names a provider by id (`X-Model-Preset`), never a base URL, so it cannot point
the server at another address. With it comes either the visitor's own key
(`X-Model-Key`) or a demo code (`X-Demo-Code`, for the `demo` provider), never both. No
header at all means the server's own key, as before.

A bad combination is a 422 and a malformed key or code a 401. Every detail names a
header, never a value: a visitor may paste a key into any field. The secret is held in
memory only and kept out of every repr.
"""

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import Header, HTTPException

PROVIDERS_FILE = Path(__file__).parent / "providers.json"
DEMO = "demo"
# A key or code: printable ASCII with no space, as every provider's keys are.
SECRET = re.compile(r"[\x21-\x7e]{1,512}")
FIELDS = (
    "id",
    "label",
    "kind",
    "base_url",
    "fast_model",
    "reasoning_model",
    "embedding_model",
)


@dataclass(frozen=True)
class Provider:
    id: str
    label: str
    kind: Literal["key", "demo"]
    base_url: str | None
    """`None` for the demo provider only: a demo code goes to the demo gateway."""
    fast_model: str
    reasoning_model: str
    embedding_model: str | None
    """`None` when the provider has no embedding model (Synthetic)."""


@dataclass(frozen=True)
class Credentials:
    provider: Provider
    secret: str = field(repr=False)
    """The visitor's key, or the demo code for the demo provider."""


def load_providers(path: Path = PROVIDERS_FILE) -> dict[str, Provider]:
    """The providers by id, in file order; a malformed file raises `ValueError`."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = raw.get("providers") if isinstance(raw, dict) else None
    if not isinstance(items, list) or not items:
        raise ValueError(f"{path.name} holds no providers")
    providers: dict[str, Provider] = {}
    for item in items:
        provider = _provider(item)
        if provider.id in providers:
            raise ValueError(f"{path.name}: provider {provider.id!r} appears twice")
        providers[provider.id] = provider
    return providers


def _provider(item: Any) -> Provider:
    if not isinstance(item, dict) or set(item) != set(FIELDS):
        raise ValueError(f"a provider has exactly the fields {list(FIELDS)}")
    provider = Provider(**item)
    names = (provider.id, provider.label, provider.fast_model, provider.reasoning_model)
    if not all(isinstance(v, str) and v for v in names):
        raise ValueError("a provider's id, label and model names are non-empty text")
    if provider.embedding_model is not None and not (
        isinstance(provider.embedding_model, str) and provider.embedding_model
    ):
        raise ValueError(f"provider {provider.id!r}: embedding_model is text or null")
    is_demo = provider.id == DEMO
    if provider.kind != (DEMO if is_demo else "key"):
        raise ValueError(f"provider {provider.id!r}: kind must be {DEMO if is_demo else 'key'!r}")
    if is_demo and provider.base_url is not None:
        raise ValueError("the demo provider has no base URL of its own; codes go to the gateway")
    if not is_demo and not (
        isinstance(provider.base_url, str) and provider.base_url.startswith("https://")
    ):
        raise ValueError(f"provider {provider.id!r}: base_url must be an https:// URL")
    return provider


PROVIDERS = load_providers()


def public_providers(providers: dict[str, Provider]) -> list[dict[str, Any]]:
    """Every provider as the page sees it. The file holds no secret, so this is all of it."""
    return [asdict(provider) for provider in providers.values()]


def request_credentials(
    x_model_preset: Annotated[
        str | None,
        Header(description=f"A provider id from `GET /api/providers`; `{DEMO}` for a demo code."),
    ] = None,
    x_model_key: Annotated[
        str | None,
        Header(description="The visitor's own key for the provider; never stored or logged."),
    ] = None,
    x_demo_code: Annotated[
        str | None,
        Header(description="A demo code, sent to the demo gateway; never stored or logged."),
    ] = None,
) -> Credentials | None:
    """The request's provider and secret, or `None` when it sends neither (the server's
    own key). Blank headers count as absent."""
    preset = (x_model_preset or "").strip() or None
    key = (x_model_key or "").strip() or None
    code = (x_demo_code or "").strip() or None

    if key and code:
        raise _unprocessable("send X-Model-Key or X-Demo-Code, not both")
    if code:
        if preset not in (None, DEMO):
            raise _unprocessable(f"X-Demo-Code is accepted only with X-Model-Preset: {DEMO}")
        if not SECRET.fullmatch(code):
            raise _unauthorized("X-Demo-Code is not a valid demo code")
        return Credentials(PROVIDERS[DEMO], code)
    if preset is None:
        if key:
            raise _unprocessable("X-Model-Preset is required with X-Model-Key")
        return None
    if preset not in PROVIDERS:
        raise _unprocessable(
            f"X-Model-Preset is not a provider; use one of: {', '.join(PROVIDERS)}"
        )
    if preset == DEMO:
        raise _unprocessable(f"X-Demo-Code is required with X-Model-Preset: {DEMO}")
    if not key:
        raise _unprocessable("X-Model-Key is required with X-Model-Preset")
    if not SECRET.fullmatch(key):
        raise _unauthorized("X-Model-Key is not a valid key")
    return Credentials(PROVIDERS[preset], key)


def _unprocessable(detail: str) -> HTTPException:
    return HTTPException(status_code=422, detail=detail)


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(status_code=401, detail=detail)
