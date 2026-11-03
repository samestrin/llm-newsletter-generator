# llm-newsletter-generator

[![Star on GitHub](https://img.shields.io/github/stars/samestrin/llm-newsletter-generator?style=social)](https://github.com/samestrin/llm-newsletter-generator/stargazers) [![Fork on GitHub](https://img.shields.io/github/forks/samestrin/llm-newsletter-generator?style=social)](https://github.com/samestrin/llm-newsletter-generator/network/members) [![Watch on GitHub](https://img.shields.io/github/watchers/samestrin/llm-newsletter-generator?style=social)](https://github.com/samestrin/llm-newsletter-generator/watchers)

![Version 0.0.4](https://img.shields.io/badge/Version-0.0.4-blue) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Built with Python](https://img.shields.io/badge/Built%20with-Python-green)](https://www.python.org/)

**llm-newsletter-generator** is being rebuilt as a Python package that generates text-only newsletters from RSS feeds using LLMs. See [VISION.md](VISION.md) for the plan.

The rebuild is in progress: `newsletter issue` drafts a plain issue from RSS and Atom feeds; editing, grouping and the writer's voice arrive with later work.

## Installation

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/samestrin/llm-newsletter-generator/
cd llm-newsletter-generator
uv sync
```

Dependencies are declared in `pyproject.toml`, which is also where the version lives.

## Usage

```bash
uv run newsletter --help
uv run newsletter --version
NEWSLETTER_FAST_MODEL=gpt-4o-mini NEWSLETTER_API_KEY=... \
  uv run newsletter issue https://example.com/feed.xml --title "AI Weekly" --topic AI -o issue.md
```

`newsletter issue` summarizes each feed item with the fast model and writes Markdown with a link per item, then a run report listing any feed, entry or model call that failed and what each step cost. It exits 1 when anything failed, so a partial issue is never reported as a success.

## Development

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

## Contribute

Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes or improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Share

[![Twitter](https://img.shields.io/badge/X-Tweet-blue)](https://twitter.com/intent/tweet?text=Check%20out%20this%20awesome%20project!&url=https://github.com/samestrin/llm-newsletter-generator) [![Facebook](https://img.shields.io/badge/Facebook-Share-blue)](https://www.facebook.com/sharer/sharer.php?u=https://github.com/samestrin/llm-newsletter-generator) [![LinkedIn](https://img.shields.io/badge/LinkedIn-Share-blue)](https://www.linkedin.com/sharing/share-offsite/?url=https://github.com/samestrin/llm-newsletter-generator)
