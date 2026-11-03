"""The voice pass on an edited issue: the editor's prose changes, nothing else does."""

import copy
import json
import re
from datetime import date

import httpx
import pytest

from newsletter.editor import Claim, Edition, Prose, Theme
from newsletter.editor.faithfulness import CheckedClaim, CheckedStory
from newsletter.llm import LLMClient, Settings
from newsletter.render import Newsletter, build_report, render_html, render_markdown
from newsletter.sources import Item
from newsletter.voice import VoiceRun, VoiceSettings, voice_edition

KEY = "yv-secret-key-123"
CLAIM_TEXT = (
    'The lab said "it runs on a laptop" and scored 71% on MMLU, per https://lab.example.org/a.'
)
FLAGGED_TEXT = "The model has 1 trillion parameters."
INTRO = "Two labs shipped small models this week, one at 3B parameters."
WHY = 'A "laptop-class" model scoring 71% matters for local use.'


def story_items():
    item = Item(
        title="Small model",
        link="https://lab.example.org/small-model",
        summary="It runs on a laptop.",
        published=None,
        feed_url="https://lab.example.org/atom.xml",
        feed_title="Example Lab",
        rights=None,
    )
    return {"s1": [item]}


def edition():
    supported = Claim("s1.c1", CLAIM_TEXT, "it runs on a laptop", "https://lab.example.org/a")
    flagged = Claim("s1.c2", FLAGGED_TEXT, "Second paragraph", "https://lab.example.org/b")
    story = CheckedStory(
        "s1",
        [
            CheckedClaim(supported, "supported", "the passage says so"),
            CheckedClaim(flagged, "unsupported", "the passage does not say so"),
        ],
    )
    return Edition(
        stories=[story],
        lead="s1",
        themes=[Theme("Small models", ["s1.c1"]), Theme("Local use", ["s1.c1"])],
        intro=Prose(INTRO, ["s1.c1"]),
        why_it_matters={"s1": Prose(WHY, ["s1.c1"])},
    )


def voiced(text):
    return f"Honestly? {text}"


class FakeModel:
    """The reasoning model: rewrites the text after `Text:` with `rewrite`."""

    def __init__(self, rewrite=voiced):
        self.rewrite = rewrite
        self.prompts: list[str] = []

    def __call__(self, request):
        prompt = json.loads(request.content)["messages"][-1]["content"]
        self.prompts.append(prompt)
        text = re.search(r"Text:\n\n(.*)\Z", prompt, re.S).group(1)
        reply = {"choices": [{"message": {"content": self.rewrite(text)}}]}
        return httpx.Response(200, json=reply)


def yourvoice(status=200):
    def reply(request):
        if status != 200:
            return httpx.Response(status, json={"error": f"bad key {KEY}"})
        texts = json.loads(request.content)["texts"]
        return httpx.Response(200, json={"results": [{"style_prompt": "Wry."} for _ in texts]})

    return httpx.MockTransport(reply)


@pytest.fixture
def settings(tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    (samples / "post.md").write_text("I write like this.", encoding="utf-8")
    return VoiceSettings.from_env(
        {
            "YOURVOICE_API_KEY": KEY,
            "NEWSLETTER_VOICE_SAMPLES": str(samples),
            "NEWSLETTER_VOICE_CACHE_DIR": str(tmp_path / "cache"),
        }
    )


def make_llm(model):
    env = {
        "NEWSLETTER_BASE_URL": "http://model.test/v1",
        "NEWSLETTER_REASONING_MODEL": "reason-1",
        "NEWSLETTER_MAX_RETRIES": "0",
    }
    return LLMClient(Settings.from_env(env), transport=httpx.MockTransport(model))


def render(ed, run):
    report = build_report([], ed, story_items(), {}, run.report())
    newsletter = Newsletter("Weekly", date(2026, 11, 3), ed, story_items(), report)
    return render_markdown(newsletter), render_html(newsletter)


def claim_lines(markdown):
    return [line for line in markdown.splitlines() if line.startswith(r"- \[")]


def citations(text):
    return re.findall(r"\[\d+\]\(<[^>]+>\)", text)


def test_the_editors_prose_is_rewritten_and_every_claim_stays_byte_for_byte(settings):
    ed = edition()
    stories_before = copy.deepcopy(ed.stories)
    neutral_md, _ = render(edition(), VoiceRun())

    run = voice_edition(ed, make_llm(FakeModel()), settings, transport=yourvoice())

    assert run.ok and sorted(run.voiced) == ["intro", "theme-1", "theme-2", "why-s1"]
    assert ed.intro == Prose(voiced(INTRO), ["s1.c1"])
    assert ed.themes == [
        Theme(voiced("Small models"), ["s1.c1"]),
        Theme(voiced("Local use"), ["s1.c1"]),
    ]
    assert ed.why_it_matters == {"s1": Prose(voiced(WHY), ["s1.c1"])}
    # The claims, their quotes, numbers, statuses and source links are untouched.
    assert ed.stories == stories_before
    assert ed.lead == "s1"

    markdown, html = render(ed, run)
    assert claim_lines(markdown) == claim_lines(neutral_md)
    assert citations(markdown) == citations(neutral_md)
    assert len(citations(markdown)) == 4  # intro, two themes, why it matters
    assert "it runs on a laptop" in html and "https://lab.example.org/a" in html
    assert "Writer's voice: 4 of 4 texts rewritten." in markdown


def test_no_claim_is_ever_sent_to_the_rewrite(settings):
    model = FakeModel()
    voice_edition(edition(), make_llm(model), settings, transport=yourvoice())

    assert len(model.prompts) == 4
    for prompt in model.prompts:
        assert "it runs on a laptop" not in prompt
        assert FLAGGED_TEXT not in prompt
        assert "s1.c1" not in prompt


def test_a_rewrite_that_changes_a_number_in_the_prose_keeps_that_text_neutral(settings):
    model = FakeModel(lambda text: voiced(text.replace("71%", "90%")))
    ed = edition()

    run = voice_edition(ed, make_llm(model), settings, transport=yourvoice())

    assert ed.why_it_matters["s1"] == Prose(WHY, ["s1.c1"])
    assert ed.intro.text == voiced(INTRO)
    assert not run.ok
    markdown, _ = render(ed, run)
    assert "Writer's voice: 3 of 4 texts rewritten." in markdown
    assert r"`why-s1`: rewrite changed facts: number 71% was dropped" in markdown


def test_without_a_key_the_prose_stays_neutral_and_the_report_says_so(tmp_path):
    model, ed = FakeModel(), edition()
    settings = VoiceSettings.from_env({"NEWSLETTER_VOICE_CACHE_DIR": str(tmp_path)})

    run = voice_edition(ed, make_llm(model), settings)

    assert model.prompts == []
    assert ed.intro == Prose(INTRO, ["s1.c1"]) and ed.why_it_matters["s1"].text == WHY
    assert run.ok
    markdown, html = render(ed, run)
    assert r"Neutral voice: YOURVOICE\_API\_KEY is not set." in markdown
    assert "Neutral voice: YOURVOICE_API_KEY is not set." in html


def test_a_failed_voice_call_keeps_the_neutral_prose_with_a_report_line(settings):
    model, ed = FakeModel(), edition()

    run = voice_edition(ed, make_llm(model), settings, transport=yourvoice(status=500))

    assert model.prompts == []
    assert ed.intro.text == INTRO and [t.name for t in ed.themes] == ["Small models", "Local use"]
    assert not run.ok
    markdown, html = render(ed, run)
    assert "Writer's voice: 0 of 4 texts rewritten." in markdown
    assert "`intro`: voice profile failed: HTTP 500" in markdown
    assert KEY not in markdown and KEY not in html


def test_an_edition_with_no_prose_makes_no_call(settings):
    model = FakeModel()
    ed = Edition(stories=edition().stories, error="editor output rejected twice")

    run = voice_edition(ed, make_llm(model), settings, transport=yourvoice())

    assert model.prompts == [] and run.ok
    assert ed.intro is None and ed.themes == [] and ed.why_it_matters == {}
