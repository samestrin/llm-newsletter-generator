"""One client for any OpenAI-compatible endpoint, with models set from the environment.

The `fast` model does the bulk work, the `reasoning` model does the editing, and the
`embedding` model compares stories. Every call is recorded as a step with its usage
and cost, a failed call raises `LLMError`, and the API key is never logged or
reported.

Environment:

    NEWSLETTER_BASE_URL          endpoint, default https://api.openai.com/v1
    NEWSLETTER_API_KEY           or OPENAI_API_KEY; optional for local endpoints
    NEWSLETTER_FAST_MODEL        model per tier; required only when the tier is used
    NEWSLETTER_REASONING_MODEL
    NEWSLETTER_EMBEDDING_MODEL
    NEWSLETTER_<TIER>_PRICE_INPUT, NEWSLETTER_<TIER>_PRICE_OUTPUT
                                 USD per million tokens; without them the cost the
                                 server reports is used, else the cost is unknown
    NEWSLETTER_TIMEOUT           seconds per request, default 60
    NEWSLETTER_MAX_RETRIES       retries after a timeout, connection error, 408, 409,
                                 429 or 5xx, default 2
"""

import logging
import os
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx

log = logging.getLogger(__name__)

TIERS = ("fast", "reasoning", "embedding")
CHAT_TIERS = ("fast", "reasoning")
DEFAULT_BASE_URL = "https://api.openai.com/v1"
RETRY_STATUSES = {408, 409, 429}
ERROR_BODY_LIMIT = 500


class ConfigError(ValueError):
    """A setting is missing or invalid."""


class LLMError(RuntimeError):
    """A model call failed: no usable result came back."""

    def __init__(self, step: str, message: str):
        super().__init__(f"{step}: {message}")
        self.step = step


@dataclass(frozen=True)
class Price:
    """USD per million tokens."""

    input: float = 0.0
    output: float = 0.0


@dataclass(frozen=True)
class Settings:
    base_url: str = DEFAULT_BASE_URL
    api_key: str | None = field(default=None, repr=False)
    models: dict[str, str] = field(default_factory=dict)
    prices: dict[str, Price] = field(default_factory=dict)
    timeout: float = 60.0
    max_retries: int = 2

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        """Read settings from `NEWSLETTER_*` variables (and `OPENAI_API_KEY`)."""
        env = os.environ if env is None else env
        models = {}
        prices = {}
        for tier in TIERS:
            prefix = f"NEWSLETTER_{tier.upper()}"
            if env.get(f"{prefix}_MODEL"):
                models[tier] = env[f"{prefix}_MODEL"]
            price_in = _number(env, f"{prefix}_PRICE_INPUT", float, minimum=0)
            price_out = _number(env, f"{prefix}_PRICE_OUTPUT", float, minimum=0)
            if price_in is not None or price_out is not None:
                prices[tier] = Price(price_in or 0.0, price_out or 0.0)
        timeout = _number(env, "NEWSLETTER_TIMEOUT", float, minimum=0, exclusive=True)
        max_retries = _number(env, "NEWSLETTER_MAX_RETRIES", int, minimum=0)
        return cls(
            base_url=(env.get("NEWSLETTER_BASE_URL") or DEFAULT_BASE_URL).rstrip("/"),
            api_key=env.get("NEWSLETTER_API_KEY") or env.get("OPENAI_API_KEY") or None,
            models=models,
            prices=prices,
            timeout=60.0 if timeout is None else timeout,
            max_retries=2 if max_retries is None else max_retries,
        )


def _number(env, name, kind, *, minimum, exclusive=False):
    raw = env.get(name)
    if raw is None or raw == "":
        return None
    try:
        value = kind(raw)
    except ValueError:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from None
    if value < minimum or (exclusive and value == minimum):
        bound = "greater than" if exclusive else "at least"
        raise ConfigError(f"{name} must be {bound} {minimum}, got {raw!r}")
    return value


@dataclass
class StepUsage:
    """One model call as it appears in the run report."""

    step: str
    tier: str
    model: str
    ok: bool
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float | None = None
    attempts: int = 0
    seconds: float = 0.0
    error: str | None = None


class _Failure(Exception):
    def __init__(self, message: str, retry: bool):
        super().__init__(message)
        self.retry = retry


class LLMClient:
    def __init__(
        self,
        settings: Settings,
        *,
        retry_backoff: float = 1.0,
        transport: httpx.BaseTransport | None = None,
    ):
        self.settings = settings
        self.retry_backoff = retry_backoff
        self.records: list[StepUsage] = []
        headers = {}
        if settings.api_key:
            headers["Authorization"] = f"Bearer {settings.api_key}"
        self._http = httpx.Client(
            base_url=settings.base_url,
            headers=headers,
            timeout=settings.timeout,
            transport=transport,
        )

    def __repr__(self) -> str:
        return f"LLMClient(base_url={self.settings.base_url!r}, steps={len(self.records)})"

    def close(self) -> None:
        self._http.close()

    def complete(
        self,
        step: str,
        messages: list[dict[str, Any]],
        *,
        tier: str = "fast",
        **params: Any,
    ) -> str:
        """Return the reply text from the `fast` or `reasoning` model."""
        if tier not in CHAT_TIERS:
            raise ConfigError(f"tier must be one of {CHAT_TIERS}, got {tier!r}")
        model = self._model(tier)
        payload = {"model": model, "messages": messages, **params}
        return self._call(step, tier, model, "/chat/completions", payload, _parse_completion)

    def embed(self, step: str, inputs: list[str]) -> list[list[float]]:
        """Return one embedding per input, in input order."""
        model = self._model("embedding")
        payload = {"model": model, "input": inputs}
        return self._call(
            step,
            "embedding",
            model,
            "/embeddings",
            payload,
            lambda data: _parse_embeddings(data, len(inputs)),
        )

    def report(self) -> dict[str, Any]:
        """Usage and cost per step, for the run report. Unknown costs are not counted as 0."""
        return {
            "steps": [asdict(record) for record in self.records],
            "total_cost": sum(r.cost for r in self.records if r.cost is not None),
            "cost_complete": all(r.cost is not None for r in self.records),
            "failed_steps": [r.step for r in self.records if not r.ok],
        }

    def _model(self, tier: str) -> str:
        model = self.settings.models.get(tier)
        if not model:
            raise ConfigError(f"no {tier} model: set NEWSLETTER_{tier.upper()}_MODEL")
        return model

    def _call(self, step, tier, model, path, payload, parse):
        record = StepUsage(step=step, tier=tier, model=model, ok=False)
        self.records.append(record)
        started = time.monotonic()
        try:
            data = self._post(record, path, payload)
            # Tokens spent on a reply we cannot use still count toward the cost.
            usage = data.get("usage")
            usage = usage if isinstance(usage, dict) else {}
            record.prompt_tokens = _int(usage.get("prompt_tokens"))
            record.completion_tokens = _int(usage.get("completion_tokens"))
            record.cost = self._cost(tier, record, usage)
            result = parse(data)
        except _Failure as failure:
            record.error = self._redact(str(failure))
            log.error("step %s (%s) failed: %s", step, model, record.error)
            raise LLMError(step, record.error) from None
        finally:
            record.seconds = round(time.monotonic() - started, 3)
        record.ok = True
        log.info(
            "step %s (%s): %d in, %d out, cost %s",
            step,
            model,
            record.prompt_tokens,
            record.completion_tokens,
            "unknown" if record.cost is None else f"${record.cost:.6f}",
        )
        return result

    def _post(self, record: StepUsage, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        attempts = 1 + self.settings.max_retries
        for attempt in range(1, attempts + 1):
            record.attempts = attempt
            try:
                return self._attempt(path, payload)
            except _Failure as failure:
                if not failure.retry or attempt == attempts:
                    raise
                log.warning(
                    "step %s attempt %d/%d failed, retrying: %s",
                    record.step,
                    attempt,
                    attempts,
                    self._redact(str(failure)),
                )
                time.sleep(self.retry_backoff * 2 ** (attempt - 1))
        raise AssertionError("unreachable")

    def _attempt(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self._http.post(path, json=payload)
        except httpx.TimeoutException:
            raise _Failure(f"timed out after {self.settings.timeout}s", retry=True) from None
        except httpx.TransportError as exc:
            raise _Failure(f"request failed: {type(exc).__name__}: {exc}", retry=True) from None
        if response.status_code >= 400:
            retry = response.status_code in RETRY_STATUSES or response.status_code >= 500
            # Some providers echo the key back in the error body.
            body = self._redact(response.text)[:ERROR_BODY_LIMIT]
            raise _Failure(f"HTTP {response.status_code}: {body}", retry=retry)
        try:
            data = response.json()
        except ValueError:
            raise _Failure("response is not JSON", retry=False) from None
        if not isinstance(data, dict):
            raise _Failure("response is not a JSON object", retry=False)
        return data

    def _cost(self, tier: str, record: StepUsage, usage: dict[str, Any]) -> float | None:
        price = self.settings.prices.get(tier)
        if price is not None:
            return (
                record.prompt_tokens * price.input + record.completion_tokens * price.output
            ) / 1_000_000
        cost = usage.get("cost")
        if isinstance(cost, int | float) and not isinstance(cost, bool):
            return float(cost)
        return None

    def _redact(self, text: str) -> str:
        key = self.settings.api_key
        return text.replace(key, "[redacted]") if key else text


def _int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _parse_completion(data: dict[str, Any]) -> str:
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise _Failure("response has no message", retry=False) from None
    if not isinstance(content, str) or not content.strip():
        raise _Failure("response message is empty", retry=False)
    return content


def _parse_embeddings(data: dict[str, Any], count: int) -> list[list[float]]:
    items = data.get("data")
    if not isinstance(items, list) or len(items) != count:
        raise _Failure(f"expected {count} embeddings", retry=False)
    vectors: list[list[float] | None] = [None] * count
    for position, item in enumerate(items):
        index = item.get("index", position) if isinstance(item, dict) else None
        vector = item.get("embedding") if isinstance(item, dict) else None
        if (
            not isinstance(index, int)
            or not 0 <= index < count
            or vectors[index] is not None
            or not isinstance(vector, list)
            or not all(isinstance(x, int | float) for x in vector)
        ):
            raise _Failure("response has a malformed embedding", retry=False)
        vectors[index] = vector
    return vectors
