Summarize the story "{{ story_title }}" as a short list of claims, using only the sources below.

For each claim give:

- `text`: one sentence stating one fact from the sources.
- `passage`: the words in one source that state that fact, copied exactly, character for character. Do not change, shorten inside, or join passages, and do not copy across two sources.

Leave out any fact you cannot copy a passage for. Make no claims about the site that listed the story, such as its points, votes, comment counts or URLs: only about the story itself. Reply with JSON only: {"claims": [{"text": "...", "passage": "..."}]}

{{ sources }}
