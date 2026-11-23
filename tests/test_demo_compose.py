"""The demo compose file keeps its shape: no host port, no secret in the file,
demo mode on, the gateway key only from `.env.demo`, the demo gateway and model
names set, an https origin, timeouts that fit, a data volume, the api seeded with
the sample. It is this repo's half of the public demo on the `demos` droplet
(epic 09.1 T5); fleet-deploy's entry builds and routes it, so the shape is part
of the deploy contract, as for the embassy review deploy."""

from pathlib import Path

import yaml

from newsletter.api.settings import SECRET_NAMES

ROOT = Path(__file__).resolve().parents[1]
COMPOSE_FILE = ROOT / "docker-compose.demo.yml"
DEMO_MODELS = (
    "NEWSLETTER_DEMO_FAST_MODEL",
    "NEWSLETTER_DEMO_REASONING_MODEL",
    "NEWSLETTER_DEMO_EMBEDDING_MODEL",
)


def _env_names(service: dict) -> set[str]:
    env = service.get("environment") or {}
    if isinstance(env, dict):
        return set(env)
    return {entry.split("=", 1)[0] for entry in env}


def _env_values(service: dict) -> dict[str, str]:
    """The environment as name -> value text, in either of compose's two forms."""
    env = service.get("environment") or {}
    if isinstance(env, dict):
        return {name: str(value) for name, value in env.items()}
    values = {}
    for entry in env:
        name, _, value = entry.partition("=")
        values[name] = value
    return values


def _env_files(service: dict) -> list[str]:
    files = service.get("env_file") or []
    return [files] if isinstance(files, str) else list(files)


def _seconds(value: str | None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def problems(doc: dict) -> list[str]:
    """Every way the file breaks the demo-deploy contract; empty when it holds."""
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
    env = _env_values(api)
    if ".env.demo" not in _env_files(api):
        found.append("api does not read the gateway key from .env.demo")
    if env.get("NEWSLETTER_DEMO") != "1":
        found.append("demo mode is not on (NEWSLETTER_DEMO=1)")
    if not env.get("NEWSLETTER_DEMO_BASE_URL"):
        found.append("NEWSLETTER_DEMO_BASE_URL (the droplet's gateway) is not set")
    for name in DEMO_MODELS:
        if not env.get(name):
            found.append(f"{name} is not set")
    if not env.get("NEWSLETTER_API_ORIGINS", "").startswith("https://"):
        found.append("NEWSLETTER_API_ORIGINS is not the demo's https origin")
    reasoning = _seconds(env.get("NEWSLETTER_REASONING_TIMEOUT"))
    if reasoning != 300:
        found.append("NEWSLETTER_REASONING_TIMEOUT is not 300")
    run_timeout = _seconds(env.get("NEWSLETTER_RUN_TIMEOUT"))
    if run_timeout is None or run_timeout <= (reasoning or 0):
        found.append("NEWSLETTER_RUN_TIMEOUT does not fit NEWSLETTER_REASONING_TIMEOUT")
    command = api.get("command") or []
    if isinstance(command, str):
        command = command.split()
    if "serve" not in command or "--seed-sample" not in command:
        found.append("api is not started with serve --seed-sample")
    volumes = api.get("volumes") or []
    data = [v.split(":", 1)[0] for v in volumes if isinstance(v, str) and v.endswith(":/data")]
    if not data:
        found.append("api has no data volume mounted at /data")
    elif not any(name in (doc.get("volumes") or {}) for name in data):
        found.append("the data volume is not declared under volumes:")
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
    assert ".env.demo" in lines


def test_the_check_catches_a_host_port() -> None:
    doc = _doc()
    doc["services"]["web"]["ports"] = ["8080:80"]
    assert any("host port" in p for p in problems(doc))


def test_the_check_catches_a_secret_in_the_file() -> None:
    for secret in SECRET_NAMES:
        doc = _doc()
        doc["services"]["api"]["environment"][secret] = "literal"
        assert any(secret in p for p in problems(doc)), secret
    # The list form of environment is caught too.
    doc = _doc()
    doc["services"]["api"]["environment"] = ["NEWSLETTER_API_KEY=literal"]
    assert any("NEWSLETTER_API_KEY" in p for p in problems(doc))


def test_the_check_reads_the_list_form_of_environment() -> None:
    # Compose also accepts environment as a list of NAME=value strings; the
    # shipped settings in that form still hold the contract.
    doc = _doc()
    env = doc["services"]["api"]["environment"]
    doc["services"]["api"]["environment"] = [f"{name}={value}" for name, value in env.items()]
    assert problems(doc) == []
    # A bare pass-through name carries no inline value: demo mode is then not on.
    doc = _doc()
    doc["services"]["api"]["environment"] = ["NEWSLETTER_DEMO"]
    assert any("demo mode" in p for p in problems(doc))


def test_the_check_catches_an_image_name() -> None:
    doc = _doc()
    doc["services"]["api"]["image"] = "x:y"
    assert any("image" in p for p in problems(doc))


def test_the_check_catches_demo_mode_off_or_lost() -> None:
    doc = _doc()
    doc["services"]["api"]["environment"]["NEWSLETTER_DEMO"] = "0"
    assert any("demo mode" in p for p in problems(doc))
    doc = _doc()
    del doc["services"]["api"]["environment"]["NEWSLETTER_DEMO"]
    assert any("demo mode" in p for p in problems(doc))


def test_the_check_catches_a_lost_gateway_or_model_name() -> None:
    doc = _doc()
    del doc["services"]["api"]["environment"]["NEWSLETTER_DEMO_BASE_URL"]
    assert any("NEWSLETTER_DEMO_BASE_URL" in p for p in problems(doc))
    for name in DEMO_MODELS:
        doc = _doc()
        del doc["services"]["api"]["environment"][name]
        assert any(name in p for p in problems(doc)), name


def test_the_check_catches_a_wrong_origin_or_timeouts() -> None:
    doc = _doc()
    doc["services"]["api"]["environment"]["NEWSLETTER_API_ORIGINS"] = (
        "http://newsletter.samestrin.com"
    )
    assert any("https origin" in p for p in problems(doc))
    doc = _doc()
    doc["services"]["api"]["environment"]["NEWSLETTER_REASONING_TIMEOUT"] = "60"
    assert any("NEWSLETTER_REASONING_TIMEOUT" in p for p in problems(doc))
    # A run timeout at or under the reasoning timeout cannot fit a whole run.
    for value in ("300", "60", ""):
        doc = _doc()
        doc["services"]["api"]["environment"]["NEWSLETTER_RUN_TIMEOUT"] = value
        assert any("NEWSLETTER_RUN_TIMEOUT" in p for p in problems(doc)), value
    doc = _doc()
    del doc["services"]["api"]["environment"]["NEWSLETTER_RUN_TIMEOUT"]
    assert any("NEWSLETTER_RUN_TIMEOUT" in p for p in problems(doc))


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


def test_the_check_catches_a_lost_seed_or_key_file() -> None:
    doc = _doc()
    doc["services"]["api"]["command"] = ["/app/.venv/bin/newsletter", "serve"]
    assert any("--seed-sample" in p for p in problems(doc))
    doc = _doc()
    del doc["services"]["api"]["env_file"]
    assert any(".env.demo" in p for p in problems(doc))


def test_the_check_catches_a_lost_data_volume() -> None:
    doc = _doc()
    doc["services"]["api"]["volumes"] = []
    assert any("data volume" in p for p in problems(doc))
    # A mount whose volume is not declared does not survive a deploy either.
    doc = _doc()
    del doc["volumes"]["drafts"]
    assert any("not declared" in p for p in problems(doc))


def test_the_check_catches_a_lost_healthcheck_or_network() -> None:
    doc = _doc()
    del doc["services"]["web"]["healthcheck"]
    assert any("healthcheck" in p for p in problems(doc))
    doc = _doc()
    doc["networks"]["proxy"]["external"] = False
    assert any("proxy" in p for p in problems(doc))
