# Case study: a newsletter that never invents a fact

This is the story of building `llm-newsletter-generator`, a Python tool that drafts a newsletter issue from RSS feeds with LLMs — and of the design choices that keep it honest. Every number below comes from the project's own eval replay in [`evals/results.md`](../evals/results.md).

## The problem

An LLM can summarize a week's news in seconds. It can also invent a quote, a number or a whole story with the same confidence. For a newsletter that goes out under a person's name, "usually right" is not good enough, and reading every source article by hand defeats the point.

## The approach: claims, checks and a human gate

The pipeline treats accuracy as a property of the system, not of the model:

1. **Claims quote their sources.** The fast model writes each claim together with the passage it came from. No passage, no claim.
2. **Every claim is checked.** A separate check compares each claim against its passage. A claim the source does not support is not dropped and not passed — it stays in the draft, visibly flagged.
3. **The editor may only cite supported claims.** The reasoning model writes the lead, themes and "why it matters" from claims that passed the check.
4. **A person approves.** Nothing is sent anywhere without an explicit approval, and any edit clears it.

A run report lists every failure: feeds that did not load, claims that were dropped or flagged, and each model call with its tokens and cost. The rule "nothing reports success when it did not succeed" sits above every other rule in the project's vision.

## What the eval shows

The eval replays two recorded past feeds through the claims, check and edit steps. Each case includes *planted* false claims — true passages with a false claim written over them — so the check is graded against known lies, not against itself. A planted claim that gets through fails CI.

From `evals/results.md`:

| Case | Faithfulness | Flag rate | Planted false claims flagged | Reference coverage | Cost per issue |
| --- | --- | --- | --- | --- | --- |
| open-models-week | 100.0% | 33.3% | 3/3 | 100.0% | $0.0072 |
| science-weekly | 100.0% | 33.3% | 2/2 | 100.0% | $0.0128 |
| Total | 100.0% | 33.3% | 5/5 | 100.0% | $0.0100 |

Three numbers carry the argument:

- **5/5 planted false claims flagged.** Every lie the fixtures planted was caught. The safety mechanism is tested, not just hoped for.
- **100.0% reference coverage.** Catching lies did not cost content: the issues still cite every story the human-written reference issues cited.
- **$0.0100 per issue on average.** The whole pipeline — claims, checks and editing — costs about a cent per issue, so checking every claim is affordable on any schedule.

One honest caveat, kept from the README: these numbers are a replay of recorded responses, not a live run. They show that the scoring and the flagging work on known answers, not how often a given model makes things up.

## Reading the 33.3% flag rate

The 33.3% is exactly the planted share: the fixtures plant a false claim in a third of all claims (5 planted out of 15 checked), and every flagged claim was a planted one. So the rate is set by the fixtures, not by the model — it does not mean a live model invents a third of its claims. What it does show is the check's precision on real claims: no genuine, source-supported claim was flagged, so the mechanism catches the planted lies without false positives on the real ones. Flagged claims stay in the draft, marked, where the reviewer can see them and decide.

## What this cost to build around

- **Fast for the bulk, careful for the editing.** A cheap model extracts claims; a reasoning model edits. The eval's ~1¢ per issue shows the split keeps cost negligible.
- **Failure is visible, never silent.** Failed feeds, dropped claims, failed model calls and voice-rewrite problems all land in the run report, and the command exits non-zero when anything failed.
- **The human gate is structural.** Approval is a hash of the exact text read; any edit clears it. The tool drafts, a person decides.

## Try it

The project is MIT-licensed. `uv run python -m newsletter.evals` reproduces the table above from recorded responses, with no network and no API key. See the [README](../README.md) for installation, the daily public archive, and the review page where a draft, its flagged claims and its run report are shown together.
