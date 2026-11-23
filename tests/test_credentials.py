"""Providers and credentials: the committed `providers.json`, `GET /api/providers`, and
the `X-Model-Preset` / `X-Model-Key` / `X-Demo-Code` header contract. A bad combination
is a 422 and a malformed key or code a 401; every detail names a header, never a value.
"""

import json
from typing import Annotated

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from newsletter.api import create_app
from newsletter.api.credentials import (
    DEMO,
    PROVIDERS,
    PROVIDERS_FILE,
    Credentials,
    load_providers,
    public_providers,
    request_credentials,
)
from spec_completeness import spec_gaps
from test_api import LLM_KEY, TOKEN, VOICE_KEY, FakeRunner, headers, make_settings

KEY = "sk-visitor-marker-key-0123456789"
CODE = "demo-marker-code-0123456789"
FIELDS = {
    "id",
    "label",
    "kind",
    "base_url",
    "fast_model",
    "reasoning_model",
    "embedding_model",
}


@pytest.fixture
def creds_client():
    """A bare app whose one route reports what `request_credentials` resolved."""
    app = FastAPI()

    @app.get("/creds")
    def creds(credentials: Annotated[Credentials | None, Depends(request_credentials)]):
        if credentials is None:
            return {"provider": None}
        return {"provider": credentials.provider.id, "is_marker": credentials.secret in (KEY, CODE)}

    return TestClient(app)


@pytest.fixture
def api(tmp_path):
    with TestClient(create_app(make_settings(tmp_path), runner=FakeRunner())) as client:
        yield client


def assert_header_only(response, status: int, header: str) -> None:
    assert response.status_code == status, response.text
    detail = response.json()["detail"]
    assert isinstance(detail, str)
    assert header in detail
    for value in (KEY, CODE, "nope"):
        assert value not in response.text


# The committed file.


def test_the_committed_file_lists_the_five_providers_in_order():
    assert list(PROVIDERS) == ["demo", "openai", "openrouter", "alibaba", "synthetic"]
    raw = json.loads(PROVIDERS_FILE.read_text(encoding="utf-8"))
    for item in raw["providers"]:
        assert set(item) == FIELDS


@pytest.mark.parametrize(
    "provider_id, fast, reasoning, embedding",
    [
        ("openai", "gpt-5-nano", "gpt-5-mini", "text-embedding-3-small"),
        ("openrouter", "qwen/qwen3.7-flash", "qwen/qwen3.7-plus", "thenlper/gte-base"),
        ("alibaba", "qwen-flash", "qwen-plus", "text-embedding-v4"),
        ("synthetic", "hf:zai-org/GLM-5.3-Flash", "hf:zai-org/GLM-5.3", None),
    ],
)
def test_each_key_provider_ships_the_planned_models(provider_id, fast, reasoning, embedding):
    provider = PROVIDERS[provider_id]
    assert provider.kind == "key"
    assert provider.base_url.startswith("https://")
    assert (provider.fast_model, provider.reasoning_model, provider.embedding_model) == (
        fast,
        reasoning,
        embedding,
    )


def test_the_demo_provider_has_no_base_url_of_its_own():
    demo = PROVIDERS[DEMO]
    assert demo.kind == "demo"
    assert demo.base_url is None


def write(tmp_path, providers) -> object:
    path = tmp_path / "providers.json"
    path.write_text(json.dumps({"providers": providers}), encoding="utf-8")
    return path


def good(**changes) -> dict:
    return {
        "id": "openai",
        "label": "OpenAI",
        "kind": "key",
        "base_url": "https://api.openai.com/v1",
        "fast_model": "f",
        "reasoning_model": "r",
        "embedding_model": None,
        **changes,
    }


def test_a_good_file_loads(tmp_path):
    providers = load_providers(write(tmp_path, [good()]))
    assert providers["openai"].embedding_model is None


@pytest.mark.parametrize(
    "providers, message",
    [
        ([], "no providers"),
        ([good(), good()], "twice"),
        ([{k: v for k, v in good().items() if k != "label"}], "exactly the fields"),
        ([good(extra=1)], "exactly the fields"),
        ([good(label="")], "non-empty"),
        ([good(reasoning_model=None)], "non-empty"),
        ([good(embedding_model="")], "embedding_model"),
        ([good(kind="demo")], "kind"),
        ([good(base_url="http://api.openai.com/v1")], "https://"),
        ([good(base_url=None)], "https://"),
        ([good(id=DEMO, kind="demo", base_url="https://gateway.example")], "base URL"),
        ([good(id=DEMO, kind="key", base_url=None)], "kind"),
    ],
)
def test_a_malformed_file_is_refused(tmp_path, providers, message):
    with pytest.raises(ValueError, match=message):
        load_providers(write(tmp_path, providers))


def test_a_file_that_is_not_an_object_is_refused(tmp_path):
    path = tmp_path / "providers.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="no providers"):
        load_providers(path)


# GET /api/providers.


def test_providers_are_served_with_every_field_and_no_secret(api):
    response = api.get("/api/providers", headers=headers())
    assert response.status_code == 200
    served = response.json()
    assert [p["id"] for p in served] == list(PROVIDERS)
    assert served == public_providers(PROVIDERS)
    for item in served:
        assert set(item) == FIELDS
    for secret in (TOKEN, LLM_KEY, VOICE_KEY):
        assert secret not in response.text


def test_providers_need_no_token(api):
    # The page asks for the list before it holds any credential (web/src/api.ts
    # listProviders); a 401 here would hide the visitor key and demo-code panel.
    open_answer = api.get("/api/providers")
    assert open_answer.status_code == 200
    assert open_answer.json() == api.get("/api/providers", headers=headers()).json()


def test_providers_are_in_the_spec_and_the_spec_is_complete(api):
    spec = api.get("/api/openapi.json").json()
    operation = spec["paths"]["/api/providers"]["get"]
    assert operation["responses"]["200"]["content"]["application/json"]["schema"]
    assert spec_gaps(api.app) == []


def test_the_page_may_send_the_credential_headers_from_an_allowed_origin(api):
    response = api.options(
        "/api/runs",
        headers={
            "Origin": "http://127.0.0.1:8000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "x-model-preset, x-model-key, x-demo-code",
        },
    )
    assert response.status_code == 200
    allowed = response.headers["access-control-allow-headers"].lower()
    for name in ("x-model-preset", "x-model-key", "x-demo-code"):
        assert name in allowed


# The header contract.


def test_no_headers_means_the_servers_own_key(creds_client):
    response = creds_client.get("/creds")
    assert response.json() == {"provider": None}


def test_blank_headers_count_as_absent(creds_client):
    blank = {"X-Model-Preset": " ", "X-Model-Key": "", "X-Demo-Code": "  "}
    assert creds_client.get("/creds", headers=blank).json() == {"provider": None}


@pytest.mark.parametrize("provider_id", ["openai", "openrouter", "alibaba", "synthetic"])
def test_a_preset_and_a_key_resolve_to_that_provider(creds_client, provider_id):
    response = creds_client.get(
        "/creds", headers={"X-Model-Preset": provider_id, "X-Model-Key": KEY}
    )
    assert response.json() == {"provider": provider_id, "is_marker": True}


@pytest.mark.parametrize("preset", [None, DEMO])
def test_a_demo_code_resolves_to_the_demo_provider(creds_client, preset):
    sent = {"X-Demo-Code": CODE, **({"X-Model-Preset": preset} if preset else {})}
    response = creds_client.get("/creds", headers=sent)
    assert response.json() == {"provider": DEMO, "is_marker": True}


def test_both_a_key_and_a_demo_code_are_refused(creds_client):
    response = creds_client.get(
        "/creds",
        headers={"X-Model-Preset": "openrouter", "X-Model-Key": KEY, "X-Demo-Code": CODE},
    )
    assert_header_only(response, 422, "X-Demo-Code")
    assert "X-Model-Key" in response.json()["detail"]


def test_a_key_without_a_preset_is_refused(creds_client):
    response = creds_client.get("/creds", headers={"X-Model-Key": KEY})
    assert_header_only(response, 422, "X-Model-Preset")


def test_a_demo_code_with_a_provider_preset_is_refused(creds_client):
    response = creds_client.get(
        "/creds", headers={"X-Model-Preset": "openrouter", "X-Demo-Code": CODE}
    )
    assert_header_only(response, 422, "X-Demo-Code")


def test_an_unknown_preset_is_refused_without_echoing_it(creds_client):
    response = creds_client.get("/creds", headers={"X-Model-Preset": "nope", "X-Model-Key": KEY})
    assert_header_only(response, 422, "X-Model-Preset")
    assert "openai" in response.json()["detail"]


def test_a_provider_preset_without_a_key_is_refused(creds_client):
    response = creds_client.get("/creds", headers={"X-Model-Preset": "openai"})
    assert_header_only(response, 422, "X-Model-Key")


def test_the_demo_preset_without_a_code_is_refused(creds_client):
    response = creds_client.get("/creds", headers={"X-Model-Preset": DEMO})
    assert_header_only(response, 422, "X-Demo-Code")


def test_the_demo_preset_with_a_key_is_refused(creds_client):
    response = creds_client.get("/creds", headers={"X-Model-Preset": DEMO, "X-Model-Key": KEY})
    assert_header_only(response, 422, "X-Demo-Code")


@pytest.mark.parametrize("bad", ["sk-has a space", "x" * 513, "sk-é"])
def test_a_malformed_key_is_a_401_naming_the_header_only(creds_client, bad):
    sent = {"X-Model-Preset": "openai", "X-Model-Key": bad}
    if not bad.isascii():  # a header goes on the wire as bytes; the server reads latin-1
        sent = {k: v.encode("utf-8") for k, v in sent.items()}
    response = creds_client.get("/creds", headers=sent)
    assert_header_only(response, 401, "X-Model-Key")
    assert bad not in response.text


@pytest.mark.parametrize("bad", ["code with spaces", "c" * 513])
def test_a_malformed_demo_code_is_a_401_naming_the_header_only(creds_client, bad):
    response = creds_client.get("/creds", headers={"X-Demo-Code": bad})
    assert_header_only(response, 401, "X-Demo-Code")
    assert bad not in response.text


def test_the_secret_is_kept_out_of_the_credentials_repr():
    credentials = Credentials(PROVIDERS["openai"], KEY)
    assert KEY not in repr(credentials)
    assert KEY not in str(credentials)
