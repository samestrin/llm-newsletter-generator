"""The embassy review compose file keeps its shape: no host port, no secret in
the file, settings only from `.env.embassy`, the api seeded with the sample.
fleet-deploy's receiver builds it as-is and waits for `web`'s healthcheck, so
the shape is part of the deploy contract (epic 07 T4)."""

from pathlib import Path

import yaml

from newsletter.api.settings import SECRET_NAMES

ROOT = Path(__file__).resolve().parents[1]
COMPOSE_FILE = ROOT / "docker-compose.embassy.yml"


def _env_names(service: dict) -> set[str]:
    env = service.get("environment") or {}
    if isinstance(env, dict):
        return set(env)
    return {entry.split("=", 1)[0] for entry in env}


def problems(doc: dict) -> list[str]:
    """Every way the file breaks the review-deploy contract; empty when it holds."""
    found = []
    services = doc.get("services") or {}
    for name, service in services.items():
        if "ports" in service:
            found.append(f"{name} publishes a host port")
        if "image" in service:
            found.append(f"{name} sets image: (the receiver tags compose's default names)")
        for secret in SECRET_NAMES:
            if secret in _env_names(service):
                found.append(f"{name} sets {secret} in the file")
    api = services.get("api") or {}
    if api.get("env_file") != [".env.embassy"]:
        found.append("api does not read its settings only from .env.embassy")
    command = api.get("command") or []
    if isinstance(command, str):
        command = command.split()
    if "serve" not in command or "--seed-sample" not in command:
        found.append("api is not started with serve --seed-sample")
    web = services.get("web") or {}
    if not web.get("healthcheck"):
        found.append("web has no healthcheck (the receiver waits for it)")
    if not (doc.get("networks") or {}).get("proxy", {}).get("external"):
        found.append("the proxy network is not external")
    return found


def _doc() -> dict:
    return yaml.safe_load(COMPOSE_FILE.read_text())


def test_the_shipped_file_holds_the_contract() -> None:
    assert problems(_doc()) == []


def test_the_env_file_is_gitignored() -> None:
    lines = (ROOT / ".gitignore").read_text().splitlines()
    assert ".env.embassy" in lines


def test_the_check_catches_a_host_port() -> None:
    doc = _doc()
    doc["services"]["web"]["ports"] = ["8080:80"]
    assert any("host port" in p for p in problems(doc))


def test_the_check_catches_a_secret_in_the_file() -> None:
    for secret in SECRET_NAMES:
        doc = _doc()
        doc["services"]["api"]["environment"] = {secret: "literal"}
        assert any(secret in p for p in problems(doc)), secret
    # The list form of environment is caught too.
    doc = _doc()
    doc["services"]["api"]["environment"] = ["NEWSLETTER_API_KEY=literal"]
    assert any("NEWSLETTER_API_KEY" in p for p in problems(doc))


def test_the_check_catches_an_image_name() -> None:
    # The receiver tags compose's default <name>-<service> images for rollback;
    # an image: line would break that tagging.
    doc = _doc()
    doc["services"]["api"]["image"] = "x:y"
    assert any("image" in p for p in problems(doc))


def test_the_check_reads_a_string_command() -> None:
    # Compose also accepts command as a plain string.
    doc = _doc()
    doc["services"]["api"]["command"] = "newsletter serve --seed-sample"
    assert problems(doc) == []
    doc = _doc()
    doc["services"]["api"]["command"] = "newsletter serve"
    assert any("--seed-sample" in p for p in problems(doc))
    # Whole tokens, not substrings: neither flag below is the real one.
    doc = _doc()
    doc["services"]["api"]["command"] = "newsletter serve --seed-samples"
    assert any("--seed-sample" in p for p in problems(doc))
    doc = _doc()
    doc["services"]["api"]["command"] = "newsletter preserve --seed-sample"
    assert any("serve" in p for p in problems(doc))
    doc = _doc()
    doc["services"]["api"]["command"] = ["--seed-sample"]
    assert any("serve" in p for p in problems(doc))


def test_the_check_catches_a_lost_seed_or_settings_file() -> None:
    doc = _doc()
    doc["services"]["api"]["command"] = ["/app/.venv/bin/newsletter", "serve"]
    assert any("--seed-sample" in p for p in problems(doc))
    doc = _doc()
    del doc["services"]["api"]["env_file"]
    assert any(".env.embassy" in p for p in problems(doc))


def test_the_check_catches_a_lost_healthcheck_or_network() -> None:
    doc = _doc()
    del doc["services"]["web"]["healthcheck"]
    assert any("healthcheck" in p for p in problems(doc))
    doc = _doc()
    doc["networks"]["proxy"]["external"] = False
    assert any("proxy" in p for p in problems(doc))
