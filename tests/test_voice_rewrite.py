import json
import logging

import httpx
import pytest

from newsletter.llm import ConfigError, LLMClient, Settings
from newsletter.voice import VoiceSettings, apply_voice

KEY = "yv-secret-key-123"
STYLE = "Short sentences. First person. Dry humor."
NEUTRAL = (
    'The lab released a 3B parameter model [s1.c1]. It scores 71 on the benchmark. "It runs '
    'on a laptop," the lead said. See <https://lab.example.org/small-model>.'
)
VOICED = (
    'So the lab shipped a 3B parameter model [s1.c1]. It scores 71 on the benchmark. "It runs '
    'on a laptop," the lead said. Details: <https://lab.example.org/small-model>.'
)


class FakeYourVoice:
    """The yourvoice.dev `/analyze/voice-profile` endpoint, answering with `reply`.

    `reply` is a dict sent as JSON, an int sent as that HTTP status with the key echoed
    back in the body, or an exception raised as the transport error.
    """

    def __init__(self, reply=None):
        self.reply = reply
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        reply = self.reply
        if reply is None:
            texts = json.loads(request.content)["texts"]
            reply = {
                "document_count": len(texts),
                "results": [{"style_prompt": STYLE} for _ in texts],
            }
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, int):
            return httpx.Response(reply, json={"error": f"bad key {KEY}"})
        return httpx.Response(200, json=reply)


class FakeModel:
    """An OpenAI-compatible endpoint replying with `reply`, or raising it when an exception."""

    def __init__(self, reply=VOICED):
        self.reply = reply
        self.requests: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        if isinstance(self.reply, Exception):
            raise self.reply
        return httpx.Response(200, json={"choices": [{"message": {"content": self.reply}}]})


@pytest.fixture
def samples(tmp_path):
    folder = tmp_path / "samples"
    folder.mkdir()
    (folder / "one.md").write_text("I write short. I like it.", encoding="utf-8")
    (folder / "two.txt").write_text("Another post of mine.", encoding="utf-8")
    (folder / "skip.json").write_text("{}", encoding="utf-8")
    return folder


def voice_settings(tmp_path, samples=None, **env):
    values = {"YOURVOICE_API_KEY": KEY, "NEWSLETTER_VOICE_CACHE_DIR": str(tmp_path / "cache")}
    if samples is not None:
        values["NEWSLETTER_VOICE_SAMPLES"] = str(samples)
    values.update(env)
    return VoiceSettings.from_env({k: v for k, v in values.items() if v is not None})


def make_llm(model, reasoning="reason-1"):
    env = {"NEWSLETTER_BASE_URL": "http://model.test/v1", "NEWSLETTER_MAX_RETRIES": "0"}
    if reasoning:
        env["NEWSLETTER_REASONING_MODEL"] = reasoning
    return LLMClient(Settings.from_env(env), transport=httpx.MockTransport(model))


def run(tmp_path, samples, api=None, model=None, texts=None, **env):
    api = api or FakeYourVoice()
    model = model or FakeModel()
    result = apply_voice(
        {"1": NEUTRAL} if texts is None else texts,
        make_llm(model),
        voice_settings(tmp_path, samples, **env),
        transport=httpx.MockTransport(api),
    )
    return result, api, model


def test_a_rewrite_that_keeps_every_fact_is_used(tmp_path, samples):
    result, api, model = run(tmp_path, samples)

    assert result.texts == {"1": VOICED}
    assert result.ok
    assert result.report()["kept_neutral"] == []
    assert result.report()["voiced"] == ["1"]

    [request] = api.requests
    assert request.url == "https://api.yourvoice.dev/analyze/voice-profile"
    assert request.headers["X-API-Key"] == KEY
    assert "authorization" not in request.headers
    assert json.loads(request.content) == {
        "texts": ["I write short. I like it.", "Another post of mine."]
    }

    [call] = model.requests
    assert call["model"] == "reason-1"
    prompt = "\n".join(m["content"] for m in call["messages"])
    assert STYLE in prompt
    assert NEUTRAL in prompt


@pytest.mark.parametrize(
    ("rewrite", "problem"),
    [
        (VOICED.replace("71", "95"), "number 71 was dropped"),
        (VOICED.replace(" [s1.c1]", ""), "claim id s1.c1 was dropped"),
        (VOICED.replace("on a laptop", "anywhere"), "was dropped"),
        (VOICED.replace("<https://lab.example.org/small-model>", "the post"), "link"),
    ],
)
def test_a_rewrite_that_alters_or_drops_a_fact_keeps_the_neutral_text(
    tmp_path, samples, rewrite, problem
):
    result, _, _ = run(tmp_path, samples, model=FakeModel(rewrite))

    assert result.texts == {"1": NEUTRAL}
    assert not result.ok
    [kept] = result.report()["kept_neutral"]
    assert kept["id"] == "1"
    assert kept["reason"].startswith("rewrite changed facts: ")
    assert problem in kept["reason"]


def test_a_profile_timeout_keeps_the_neutral_text_and_is_reported(tmp_path, samples):
    api = FakeYourVoice(httpx.ReadTimeout("slow"))
    result, _, model = run(tmp_path, samples, api=api, YOURVOICE_TIMEOUT="7")

    assert result.texts == {"1": NEUTRAL}
    assert not result.ok
    assert result.report()["kept_neutral"] == [
        {"id": "1", "reason": "voice profile failed: timed out after 7s"}
    ]
    assert model.requests == []


def test_a_rewrite_timeout_keeps_the_neutral_text_and_is_reported(tmp_path, samples):
    result, _, _ = run(tmp_path, samples, model=FakeModel(httpx.ReadTimeout("slow")))

    assert result.texts == {"1": NEUTRAL}
    [kept] = result.report()["kept_neutral"]
    assert "timed out" in kept["reason"]


def test_the_key_never_appears_in_a_log_or_report(tmp_path, samples, caplog):
    caplog.set_level(logging.DEBUG)
    settings = voice_settings(tmp_path, samples)

    result, _, _ = run(tmp_path, samples, api=FakeYourVoice(402))

    [kept] = result.report()["kept_neutral"]
    assert kept["reason"].startswith("voice profile failed: HTTP 402: ")
    assert "[redacted]" in kept["reason"]
    assert KEY not in json.dumps(result.report())
    assert KEY not in caplog.text
    assert KEY not in repr(settings)
    # The failure is logged, just without the key.
    assert "kept neutral: voice profile failed: HTTP 402" in caplog.text


def test_an_unreadable_sample_is_skipped_and_logged(tmp_path, samples, caplog):
    (samples / "bad.md").write_bytes(b"\xff\xfe not utf-8")

    result, api, _ = run(tmp_path, samples)

    assert result.texts == {"1": VOICED}
    assert json.loads(api.requests[0].content)["texts"] == [
        "I write short. I like it.",
        "Another post of mine.",
    ]
    assert "bad.md" in caplog.text


def test_without_a_key_nothing_is_called_and_the_text_stays_neutral(tmp_path, samples):
    result, api, model = run(tmp_path, samples, YOURVOICE_API_KEY=None)

    assert result.texts == {"1": NEUTRAL}
    assert result.ok
    assert result.report()["skipped"] == "YOURVOICE_API_KEY is not set"
    assert api.requests == [] and model.requests == []


@pytest.mark.parametrize("folder", [None, "missing", "empty"])
def test_with_a_key_but_no_samples_the_text_stays_neutral_and_is_reported(tmp_path, folder):
    if folder == "empty":
        (tmp_path / "empty").mkdir()
    samples = None if folder is None else tmp_path / folder

    result, api, model = run(tmp_path, samples)

    assert result.texts == {"1": NEUTRAL}
    assert not result.ok
    assert "NEWSLETTER_VOICE_SAMPLES" in result.report()["skipped"]
    assert api.requests == [] and model.requests == []


def test_without_a_reasoning_model_the_text_stays_neutral_and_is_reported(tmp_path, samples):
    api, model = FakeYourVoice(), FakeModel()
    result = apply_voice(
        {"1": NEUTRAL},
        make_llm(model, reasoning=None),
        voice_settings(tmp_path, samples),
        transport=httpx.MockTransport(api),
    )

    assert result.texts == {"1": NEUTRAL}
    assert not result.ok
    assert "NEWSLETTER_REASONING_MODEL" in result.report()["skipped"]
    assert api.requests == [] and model.requests == []


def test_no_texts_make_no_calls(tmp_path, samples):
    result, api, model = run(tmp_path, samples, texts={})

    assert result.texts == {}
    assert result.ok
    assert api.requests == [] and model.requests == []


def test_the_profile_is_cached_until_the_samples_change(tmp_path, samples):
    api = FakeYourVoice()
    run(tmp_path, samples, api=api)
    run(tmp_path, samples, api=api)
    assert len(api.requests) == 1

    (samples / "one.md").write_text("I changed my mind.", encoding="utf-8")
    result, _, _ = run(tmp_path, samples, api=api)
    assert len(api.requests) == 2
    assert result.texts == {"1": VOICED}


def test_an_unreadable_cache_is_rebuilt(tmp_path, samples):
    api = FakeYourVoice()
    run(tmp_path, samples, api=api)
    for path in (tmp_path / "cache").iterdir():
        path.write_text("not json", encoding="utf-8")

    result, _, _ = run(tmp_path, samples, api=api)

    assert len(api.requests) == 2
    assert result.texts == {"1": VOICED}


@pytest.mark.parametrize(
    "reply",
    [
        {"document_count": 1, "results": [{"style_prompt": "", "error": "input too large"}]},
        {"document_count": 0, "results": []},
        {"results": "nope"},
    ],
)
def test_a_profile_with_no_usable_style_is_a_failure(tmp_path, samples, reply):
    result, _, model = run(tmp_path, samples, api=FakeYourVoice(reply))

    assert result.texts == {"1": NEUTRAL}
    [kept] = result.report()["kept_neutral"]
    assert kept["reason"].startswith("voice profile failed: ")
    assert model.requests == []


def test_a_failed_rewrite_of_one_text_keeps_the_others(tmp_path, samples):
    texts = {"1": NEUTRAL, "2": "It costs $5."}
    model = FakeModel(VOICED)

    result, _, _ = run(tmp_path, samples, model=model, texts=texts)

    assert result.texts == {"1": VOICED, "2": "It costs $5."}
    assert result.report()["voiced"] == ["1"]
    assert [k["id"] for k in result.report()["kept_neutral"]] == ["2"]


def test_settings_defaults_and_a_bad_timeout(tmp_path):
    settings = VoiceSettings.from_env({})
    assert settings.api_key is None
    assert settings.base_url == "https://api.yourvoice.dev"
    assert settings.timeout == 60.0
    assert settings.samples_dir is None
    assert settings.cache_dir.parts[-2:] == ("newsletter", "voice")

    custom = VoiceSettings.from_env(
        {"YOURVOICE_BASE_URL": "http://voice.test/", "XDG_CACHE_HOME": str(tmp_path)}
    )
    assert custom.base_url == "http://voice.test"
    assert custom.cache_dir == tmp_path / "newsletter" / "voice"

    with pytest.raises(ConfigError, match="YOURVOICE_TIMEOUT"):
        VoiceSettings.from_env({"YOURVOICE_TIMEOUT": "0"})
