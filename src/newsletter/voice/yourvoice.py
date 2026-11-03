"""The writer's voice profile, built by the yourvoice.dev API from samples of their writing.

yourvoice.dev analyzes text; it does not rewrite it. `POST /analyze/voice-profile` takes
the samples and returns one profile per sample, each with a `style_prompt` that
describes how the writer writes. The rewrite itself is done by the reasoning model. The
profile is cached by the samples' content, so a run pays for it only when they change.
The key goes in the `X-API-Key` header, as the API's spec names it, and is never logged
or reported.

Environment:

    YOURVOICE_API_KEY            turns the voice on; without it the text stays neutral
    YOURVOICE_BASE_URL           default https://api.yourvoice.dev
    YOURVOICE_TIMEOUT            seconds per request, default 60
    NEWSLETTER_VOICE_SAMPLES     folder of the writer's own .md and .txt files
    NEWSLETTER_VOICE_CACHE_DIR   where profiles are kept, default
                                 $XDG_CACHE_HOME/newsletter/voice (~/.cache/...)
"""

import hashlib
import json
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from newsletter.llm import ERROR_BODY_LIMIT, _number

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.yourvoice.dev"
PROFILE_PATH = "/analyze/voice-profile"
SAMPLE_SUFFIXES = (".md", ".txt")


class VoiceError(RuntimeError):
    """The voice profile could not be built. The message never holds the key."""


@dataclass(frozen=True)
class VoiceSettings:
    api_key: str | None = field(default=None, repr=False)
    base_url: str = DEFAULT_BASE_URL
    timeout: float = 60.0
    samples_dir: Path | None = None
    cache_dir: Path = field(default_factory=lambda: _default_cache_dir({}))

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "VoiceSettings":
        """Read settings from `YOURVOICE_*` and `NEWSLETTER_VOICE_*` variables."""
        env = os.environ if env is None else env
        timeout = _number(env, "YOURVOICE_TIMEOUT", float, minimum=0, exclusive=True)
        samples = env.get("NEWSLETTER_VOICE_SAMPLES")
        cache = env.get("NEWSLETTER_VOICE_CACHE_DIR")
        return cls(
            api_key=env.get("YOURVOICE_API_KEY") or None,
            base_url=(env.get("YOURVOICE_BASE_URL") or DEFAULT_BASE_URL).rstrip("/"),
            timeout=60.0 if timeout is None else timeout,
            samples_dir=Path(samples).expanduser() if samples else None,
            cache_dir=Path(cache).expanduser() if cache else _default_cache_dir(env),
        )


def _default_cache_dir(env: Mapping[str, str]) -> Path:
    base = env.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "newsletter" / "voice"


def read_samples(settings: VoiceSettings) -> list[str]:
    """The writer's samples, in file-name order; empty when there are none to read."""
    folder = settings.samples_dir
    if folder is None or not folder.is_dir():
        return []
    texts = []
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in SAMPLE_SUFFIXES:
            try:
                text = path.read_text(encoding="utf-8").strip()
            except (OSError, UnicodeDecodeError) as error:
                log.warning("skipping voice sample %s: %s", path.name, error)
                continue
            if text:
                texts.append(text)
    return texts


def voice_style(
    settings: VoiceSettings,
    samples: list[str],
    *,
    transport: httpx.BaseTransport | None = None,
) -> str:
    """The writer's style, from the cache or else from the API; raises `VoiceError`."""
    digest = hashlib.sha256(json.dumps([settings.base_url, samples]).encode()).hexdigest()
    cached = settings.cache_dir / f"profile-{digest}.json"
    style = _read_cache(cached)
    if style:
        return style
    style = _fetch_style(settings, samples, transport)
    try:
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps({"style": style}), encoding="utf-8")
    except OSError as error:
        # The profile is still good for this run; the next run builds it again.
        log.warning("could not cache the voice profile in %s: %s", cached.parent, error)
    return style


def _read_cache(path: Path) -> str | None:
    try:
        style = json.loads(path.read_text(encoding="utf-8")).get("style")
    except (OSError, ValueError, AttributeError):
        return None
    return style if isinstance(style, str) and style.strip() else None


def _fetch_style(settings, samples, transport) -> str:
    redact = _redactor(settings.api_key)
    with httpx.Client(
        base_url=settings.base_url,
        headers={"X-API-Key": settings.api_key or ""},
        timeout=settings.timeout,
        transport=transport,
    ) as client:
        try:
            response = client.post(PROFILE_PATH, json={"texts": samples})
        except httpx.TimeoutException:
            raise VoiceError(f"timed out after {settings.timeout:g}s") from None
        except httpx.TransportError as exc:
            raise VoiceError(redact(f"request failed: {type(exc).__name__}: {exc}")) from None
    if response.status_code >= 400:
        body = redact(response.text)[:ERROR_BODY_LIMIT]
        raise VoiceError(f"HTTP {response.status_code}: {body}")
    try:
        data = response.json()
    except ValueError:
        raise VoiceError("response is not JSON") from None
    return _style_of(data)


def _style_of(data: Any) -> str:
    """Each sample's style prompt, once each, in sample order."""
    results = data.get("results") if isinstance(data, dict) else None
    if not isinstance(results, list):
        raise VoiceError("response has no list of profiles")
    styles: list[str] = []
    for result in results:
        if not isinstance(result, dict) or result.get("error"):
            continue
        style = result.get("style_prompt")
        if isinstance(style, str) and style.strip() and style.strip() not in styles:
            styles.append(style.strip())
    if not styles:
        raise VoiceError("no usable voice profile in the response")
    return "\n\n".join(styles)


def _redactor(key: str | None):
    return lambda text: text.replace(key, "[redacted]") if key else text
