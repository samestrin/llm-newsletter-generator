import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from newsletter.llm import ConfigError, LLMClient, LLMError, Settings

API_KEY = "sk-test-SECRET-0123456789abcdef"


class FakeServer:
    """A local OpenAI-compatible server whose replies each test scripts.

    `replies` is a list of (status, body) tuples consumed one per request; the
    last one repeats. The status "hang" blocks until the test ends.
    """

    def __init__(self):
        self.replies = [(200, {})]
        self.requests = []
        self.release = threading.Event()
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                fake.requests.append(
                    {"path": self.path, "headers": dict(self.headers), "body": body}
                )
                index = min(len(fake.requests), len(fake.replies)) - 1
                status, reply = fake.replies[index]
                if status == "hang":
                    fake.release.wait(10)
                    return
                if callable(reply):
                    reply = reply(self)
                data = reply if isinstance(reply, bytes) else json.dumps(reply).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def close(self):
        self.release.set()
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def server():
    fake = FakeServer()
    yield fake
    fake.close()


def chat_reply(text="A summary.", prompt=10, completion=5, **usage):
    return {
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion, **usage},
    }


def embed_reply(vectors, prompt=7):
    return {
        "data": [{"index": i, "embedding": v} for i, v in enumerate(vectors)],
        "usage": {"prompt_tokens": prompt, "total_tokens": prompt},
    }


def make_env(server, **extra):
    env = {
        "NEWSLETTER_BASE_URL": server.url,
        "NEWSLETTER_API_KEY": API_KEY,
        "NEWSLETTER_FAST_MODEL": "fast-model",
        "NEWSLETTER_REASONING_MODEL": "reasoning-model",
        "NEWSLETTER_EMBEDDING_MODEL": "embedding-model",
        "NEWSLETTER_TIMEOUT": "0.5",
        "NEWSLETTER_MAX_RETRIES": "2",
    }
    env.update(extra)
    return env


def make_client(server, **extra):
    return LLMClient(Settings.from_env(make_env(server, **extra)), retry_backoff=0)


# --- settings from the environment ---


def test_settings_read_models_and_endpoint_from_env(server):
    settings = Settings.from_env(make_env(server))
    assert settings.base_url == server.url
    assert settings.models == {
        "fast": "fast-model",
        "reasoning": "reasoning-model",
        "embedding": "embedding-model",
    }
    assert settings.timeout == 0.5
    assert settings.max_retries == 2


def test_settings_defaults():
    settings = Settings.from_env({})
    assert settings.base_url == "https://api.openai.com/v1"
    assert settings.api_key is None
    assert settings.timeout == 60.0
    assert settings.max_retries == 2


def test_settings_fall_back_to_openai_api_key():
    assert Settings.from_env({"OPENAI_API_KEY": API_KEY}).api_key == API_KEY


@pytest.mark.parametrize(
    "name, value",
    [
        ("NEWSLETTER_TIMEOUT", "soon"),
        ("NEWSLETTER_TIMEOUT", "0"),
        ("NEWSLETTER_MAX_RETRIES", "-1"),
        ("NEWSLETTER_MAX_RETRIES", "many"),
        ("NEWSLETTER_FAST_PRICE_INPUT", "cheap"),
    ],
)
def test_bad_setting_is_a_config_error_naming_the_variable(name, value):
    with pytest.raises(ConfigError, match=name):
        Settings.from_env({name: value})


def test_settings_repr_hides_the_api_key():
    assert API_KEY not in repr(Settings.from_env({"NEWSLETTER_API_KEY": API_KEY}))


def test_missing_model_is_a_config_error_naming_the_variable(server):
    env = make_env(server)
    del env["NEWSLETTER_REASONING_MODEL"]
    client = LLMClient(Settings.from_env(env), retry_backoff=0)
    with pytest.raises(ConfigError, match="NEWSLETTER_REASONING_MODEL"):
        client.complete("edit", [{"role": "user", "content": "hi"}], tier="reasoning")
    assert server.requests == []


# --- calls ---


def test_complete_uses_the_tier_model_and_returns_text(server):
    server.replies = [(200, chat_reply("Fast text."))]
    client = make_client(server)
    text = client.complete("summarize", [{"role": "user", "content": "hi"}])
    assert text == "Fast text."
    request = server.requests[0]
    assert request["path"] == "/v1/chat/completions"
    assert request["body"]["model"] == "fast-model"
    assert request["body"]["messages"] == [{"role": "user", "content": "hi"}]
    assert request["headers"]["Authorization"] == f"Bearer {API_KEY}"


def test_complete_reasoning_tier(server):
    server.replies = [(200, chat_reply("Edited."))]
    client = make_client(server)
    assert client.complete("edit", [{"role": "user", "content": "x"}], tier="reasoning")
    assert server.requests[0]["body"]["model"] == "reasoning-model"


def test_complete_passes_extra_parameters(server):
    server.replies = [(200, chat_reply())]
    make_client(server).complete("s", [{"role": "user", "content": "x"}], temperature=0.2)
    assert server.requests[0]["body"]["temperature"] == 0.2


def test_unknown_tier_is_a_config_error(server):
    with pytest.raises(ConfigError, match="tier"):
        make_client(server).complete("s", [], tier="embedding")


def test_no_api_key_sends_no_authorization_header(server):
    server.replies = [(200, chat_reply())]
    env = make_env(server)
    del env["NEWSLETTER_API_KEY"]
    LLMClient(Settings.from_env(env)).complete("s", [{"role": "user", "content": "x"}])
    assert "Authorization" not in server.requests[0]["headers"]


def test_embed_returns_vectors_in_input_order(server):
    server.replies = [
        (
            200,
            {
                "data": [
                    {"index": 1, "embedding": [0.0, 1.0]},
                    {"index": 0, "embedding": [1.0, 0.0]},
                ],
                "usage": {"prompt_tokens": 4, "total_tokens": 4},
            },
        )
    ]
    client = make_client(server)
    assert client.embed("dedupe", ["a", "b"]) == [[1.0, 0.0], [0.0, 1.0]]
    request = server.requests[0]
    assert request["path"] == "/v1/embeddings"
    assert request["body"] == {"model": "embedding-model", "input": ["a", "b"]}


# --- usage and cost per step ---


def test_usage_and_cost_are_recorded_per_step(server):
    server.replies = [
        (200, chat_reply(prompt=1000, completion=500)),
        (200, chat_reply(prompt=2000, completion=1000)),
        (200, embed_reply([[1.0]], prompt=3000)),
    ]
    client = make_client(
        server,
        NEWSLETTER_FAST_PRICE_INPUT="1.0",
        NEWSLETTER_FAST_PRICE_OUTPUT="2.0",
        NEWSLETTER_REASONING_PRICE_INPUT="10",
        NEWSLETTER_REASONING_PRICE_OUTPUT="20",
        NEWSLETTER_EMBEDDING_PRICE_INPUT="0.5",
    )
    client.complete("summarize", [{"role": "user", "content": "x"}])
    client.complete("edit", [{"role": "user", "content": "x"}], tier="reasoning")
    client.embed("dedupe", ["x"])

    steps = [(r.step, r.tier, r.model, r.ok) for r in client.records]
    assert steps == [
        ("summarize", "fast", "fast-model", True),
        ("edit", "reasoning", "reasoning-model", True),
        ("dedupe", "embedding", "embedding-model", True),
    ]
    assert [(r.prompt_tokens, r.completion_tokens) for r in client.records] == [
        (1000, 500),
        (2000, 1000),
        (3000, 0),
    ]
    # USD per million tokens.
    assert [r.cost for r in client.records] == pytest.approx([0.002, 0.04, 0.0015])

    report = client.report()
    assert report["total_cost"] == pytest.approx(0.0435)
    assert report["cost_complete"] is True
    assert [s["step"] for s in report["steps"]] == ["summarize", "edit", "dedupe"]
    json.dumps(report)


def test_cost_reported_by_server_is_used_when_no_price_is_set(server):
    server.replies = [(200, chat_reply(cost=0.0123))]
    client = make_client(server)
    client.complete("summarize", [{"role": "user", "content": "x"}])
    assert client.records[0].cost == pytest.approx(0.0123)


def test_non_numeric_cost_from_server_is_unknown(server):
    server.replies = [(200, chat_reply(cost=True))]
    client = make_client(server)
    client.complete("summarize", [{"role": "user", "content": "x"}])
    assert client.records[0].cost is None


def test_unknown_cost_is_none_not_zero(server):
    server.replies = [(200, chat_reply())]
    client = make_client(server)
    client.complete("summarize", [{"role": "user", "content": "x"}])
    assert client.records[0].cost is None
    report = client.report()
    assert report["total_cost"] == 0
    assert report["cost_complete"] is False


# --- failures are errors ---


@pytest.mark.parametrize(
    "reply",
    [
        (400, {"error": {"message": "bad request"}}),
        (200, b"not json"),
        (200, []),
        (200, "hello"),
        (200, {"choices": []}),
        (200, {"choices": [{"message": {"role": "assistant", "content": None}}]}),
        (200, {"choices": [{"message": {"role": "assistant", "content": "  "}}]}),
    ],
)
def test_failed_completion_is_an_error_and_is_recorded(server, reply):
    server.replies = [reply]
    client = make_client(server)
    with pytest.raises(LLMError) as exc:
        client.complete("summarize", [{"role": "user", "content": "x"}])
    assert exc.value.step == "summarize"
    record = client.records[-1]
    assert (record.step, record.ok) == ("summarize", False)
    assert record.error
    assert client.report()["failed_steps"] == ["summarize"]


@pytest.mark.parametrize(
    "reply",
    [
        (200, {"data": []}),
        (200, {"data": [{"index": 0, "embedding": "nope"}]}),
        (200, {"data": [{"index": 0, "embedding": [1.0]}, {"index": 0, "embedding": [1.0]}]}),
    ],
)
def test_failed_embedding_is_an_error(server, reply):
    server.replies = [reply]
    client = make_client(server)
    with pytest.raises(LLMError):
        client.embed("dedupe", ["a", "b"])
    assert client.records[-1].ok is False


def test_hung_endpoint_ends_in_an_error_after_bounded_retries(server):
    server.replies = [("hang", None)]
    client = make_client(server, NEWSLETTER_TIMEOUT="0.2", NEWSLETTER_MAX_RETRIES="1")
    with pytest.raises(LLMError, match="(?i)timed out"):
        client.complete("summarize", [{"role": "user", "content": "x"}])
    assert len(server.requests) == 2
    assert client.records[-1].ok is False


def test_server_errors_are_retried_then_succeed(server):
    server.replies = [(503, {"error": "busy"}), (429, {"error": "slow down"}), (200, chat_reply())]
    client = make_client(server)
    assert client.complete("summarize", [{"role": "user", "content": "x"}]) == "A summary."
    assert len(server.requests) == 3
    assert len(client.records) == 1
    assert client.records[0].ok is True


def test_retries_are_bounded(server):
    server.replies = [(500, {"error": "down"})]
    client = make_client(server, NEWSLETTER_MAX_RETRIES="2")
    with pytest.raises(LLMError, match="500"):
        client.complete("summarize", [{"role": "user", "content": "x"}])
    assert len(server.requests) == 3


def test_client_errors_are_not_retried(server):
    server.replies = [(401, {"error": "no"})]
    client = make_client(server)
    with pytest.raises(LLMError, match="401"):
        client.complete("summarize", [{"role": "user", "content": "x"}])
    assert len(server.requests) == 1


def test_unreachable_endpoint_is_an_error():
    env = {
        "NEWSLETTER_BASE_URL": "http://127.0.0.1:9/v1",
        "NEWSLETTER_FAST_MODEL": "m",
        "NEWSLETTER_MAX_RETRIES": "0",
    }
    client = LLMClient(Settings.from_env(env), retry_backoff=0)
    with pytest.raises(LLMError):
        client.complete("summarize", [{"role": "user", "content": "x"}])


# --- the API key never leaks ---


def echo_authorization(handler):
    # Some providers echo the key they were given in the error body.
    return {"error": {"message": f"Invalid key: {handler.headers.get('Authorization')}"}}


@pytest.mark.parametrize("status", [401, 500])
def test_failing_call_keeps_the_api_key_out_of_logs_error_and_report(server, caplog, status):
    caplog.set_level(logging.DEBUG)
    server.replies = [(status, echo_authorization)]
    client = make_client(server, NEWSLETTER_MAX_RETRIES="1")
    with pytest.raises(LLMError) as exc:
        client.complete("summarize", [{"role": "user", "content": "x"}])

    assert server.requests, "the fake server was not called"
    assert caplog.text, "nothing was logged"
    assert API_KEY not in caplog.text
    assert API_KEY not in str(exc.value)
    assert API_KEY not in repr(exc.value)
    assert API_KEY not in json.dumps(client.report())
    assert API_KEY not in repr(client)
    assert "Invalid key" in client.records[-1].error


def test_hung_call_keeps_the_api_key_out_of_logs_and_report(server, caplog):
    caplog.set_level(logging.DEBUG)
    server.replies = [("hang", None)]
    client = make_client(server, NEWSLETTER_TIMEOUT="0.2", NEWSLETTER_MAX_RETRIES="0")
    with pytest.raises(LLMError):
        client.complete("summarize", [{"role": "user", "content": "x"}])
    assert API_KEY not in caplog.text
    assert API_KEY not in json.dumps(client.report())
