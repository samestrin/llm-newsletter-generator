"""The API's settings, read from the environment.

Environment:

    NEWSLETTER_API_TOKEN     the token every request but the health check must send, as
                             `Authorization: Bearer <token>`; without it, writes are refused
    NEWSLETTER_API_HOST      bind address, default 127.0.0.1
    NEWSLETTER_API_PORT      port, default 8000
    NEWSLETTER_API_ORIGINS   comma-separated origins a write may come from, default
                             http://127.0.0.1:<port> and http://localhost:<port>
    NEWSLETTER_DATA_DIR      where drafts and runs are kept, default
                             $XDG_DATA_HOME/newsletter (~/.local/share/newsletter)
    NEWSLETTER_RUN_TIMEOUT   seconds a run may take before it is reported as failed,
                             default 900
    NEWSLETTER_DEMO          1 for demo mode: a run takes at most 6 feeds and 3 items per
                             feed, and its `date` is ignored, and each draft belongs to the
                             browser that made it (`X-Owner-Token`); off when unset
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from newsletter.llm import ConfigError, _number

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_RUN_TIMEOUT = 900.0
SECRET_NAMES = ("NEWSLETTER_API_TOKEN", "NEWSLETTER_API_KEY", "OPENAI_API_KEY", "YOURVOICE_API_KEY")
"""Variables whose values are redacted from everything the API stores, returns or logs."""


@dataclass(frozen=True)
class ApiSettings:
    token: str | None = field(default=None, repr=False)
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    origins: tuple[str, ...] = ()
    data_dir: Path = Path("data")
    run_timeout: float = DEFAULT_RUN_TIMEOUT
    demo: bool = False
    """Demo mode (`NEWSLETTER_DEMO=1`): the demo limits on a run apply, and drafts and runs
    are scoped to the owner token their run answered."""
    env: Mapping[str, str] = field(default_factory=dict, repr=False)
    """The environment the pipeline reads its model and voice settings from."""

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "ApiSettings":
        env = dict(os.environ if env is None else env)
        port = _number(env, "NEWSLETTER_API_PORT", int, minimum=1)
        if port is not None and port > 65535:
            raise ConfigError(f"NEWSLETTER_API_PORT must be a port number, got {port}")
        port = DEFAULT_PORT if port is None else port
        timeout = _number(env, "NEWSLETTER_RUN_TIMEOUT", float, minimum=0, exclusive=True)
        origins = [o.strip().rstrip("/") for o in env.get("NEWSLETTER_API_ORIGINS", "").split(",")]
        data_dir = env.get("NEWSLETTER_DATA_DIR")
        return cls(
            token=env.get("NEWSLETTER_API_TOKEN") or None,
            host=env.get("NEWSLETTER_API_HOST") or DEFAULT_HOST,
            port=port,
            origins=tuple(o for o in origins if o)
            or (f"http://127.0.0.1:{port}", f"http://localhost:{port}"),
            data_dir=Path(data_dir).expanduser() if data_dir else _default_data_dir(env),
            run_timeout=DEFAULT_RUN_TIMEOUT if timeout is None else timeout,
            demo=env.get("NEWSLETTER_DEMO", "").strip() == "1",
            env=env,
        )

    @property
    def secrets(self) -> list[str]:
        """Every secret value set, longest first so one inside another is redacted whole."""
        values = {self.token, *(self.env.get(name) for name in SECRET_NAMES)}
        return sorted((v for v in values if v), key=len, reverse=True)


def _default_data_dir(env: Mapping[str, str]) -> Path:
    base = env.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "newsletter"
