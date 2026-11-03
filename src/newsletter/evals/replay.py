"""A model client that answers from recorded responses instead of calling a model.

It is an `LLMClient` whose network call is replaced, so prompts, retries, parsing, usage
and cost go through the same code as a live run. A step that was not recorded fails like
a failed call, so a change in the pipeline's steps shows up in the eval instead of being
answered with something made up.
"""

import json
from typing import Any

from newsletter.llm import LLMClient, Settings, StepUsage, _Failure


class ReplayClient(LLMClient):
    def __init__(self, settings: Settings, recorded: dict[str, dict[str, Any]]):
        super().__init__(settings)
        self.recorded = recorded

    def _post(self, record: StepUsage, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        record.attempts = 1
        entry = self.recorded.get(record.step)
        if entry is None:
            raise _Failure(f"step {record.step!r} was not recorded", retry=False)
        content = entry["content"]
        if not isinstance(content, str):
            content = json.dumps(content)
        return {
            "choices": [{"message": {"content": content}}],
            "usage": entry.get("usage", {}),
        }
