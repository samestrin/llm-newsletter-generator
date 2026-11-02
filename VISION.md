# Vision

`llm-newsletter-generator` exists so that a person can publish a useful, accurate newsletter from the sources they follow, in their own voice, without writing it from scratch.
It serves creators, marketers and teams who send a regular digest of what happened in their field.
It turns feeds and pages into a draft issue that an editor approves: stories picked and grouped, summarized, with a lead and the "why it matters", every claim linked to its source.
It owns exactly one thing: drafting a newsletter issue from sources, ready for a human to approve.

## Every claim has a source

Every fact in an issue links to the article it came from.
A check compares each summary with its source before the draft is shown, and a claim the source does not support is flagged.
The newsletter never invents a story, a quote or a number.

## A human approves before anything is sent

The tool drafts; a person edits and approves.
Nothing is sent to a list without an explicit approval.
The draft can be sent to an email platform as a draft, never published directly.

## Fast for the bulk, careful for the editing

A fast model summarizes each item.
A reasoning model acts as the editor: it groups duplicate stories, picks the lead, names the themes and writes the "why it matters".
The person can see which model did which step, and what the issue cost.

## In the writer's voice

The issue can be written in the writer's own voice through the yourvoice.dev API when a key is set.
Without it, a plain, neutral voice is used.

## Nothing reports success when it did not succeed

This rule sits above every other rule here.
A source that failed to load, a model call that failed, or a story that was dropped is listed in the draft, never silently skipped.

## Any model, no lock-in

Models and embeddings come from any OpenAI-compatible endpoint, local or cloud, set from the environment.
Email platforms are optional connectors, and no vendor is required.

## Scope

It is not an email sending service, a subscriber list, a CMS or a social media scheduler.
It does not scrape sites that forbid it, and it respects robots.txt and feed terms.
It is a portfolio demo of how this should be built, so clarity of the code and the README matter as much as features; a public archive built daily from AI news feeds shows it working.

A change aligns when it makes issues more accurate, better edited, closer to the writer's voice, or the project easier to run and understand.
A change should be resisted when it lets a claim go unsourced, sends without approval, hides a failure, ties the project to one vendor, or adds work that is not drafting a newsletter.
