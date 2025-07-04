---
applyTo: "**"
-----

# LLM Newsletter Generator - Agent Guidelines

These guidelines define the context, rules, and methods for developing the LLM Newsletter Generator. The goal is to build a modern, interactive command-line tool for creating newsletters from RSS feeds using various Language Models (LLMs).

-----

## 1. Project Goal & Scope

The LLM Newsletter Generator is a CLI tool that helps users create personalized newsletters from one or more RSS feeds. It leverages LLMs to generate introductions, summaries for feed items, and closing remarks, based on customizable prompt templates.

The project will be modernized from its initial version to include:
- A more user-friendly and interactive CLI using Typer.
- A robust dependency management system using Poetry.
- Expanded capabilities for integrating with both local and API-based LLMs (e.g., Google Gemini, OpenAI).
- Enhanced metadata extraction from RSS feeds to create more context-aware and compelling content.

This is a **personal tool for individual use**, not a commercial product. The focus is on a clean, maintainable codebase and a great user experience.

-----

## 2. Core Principles & Technology Stack

### Core Principles

  * **User Experience First:** Prioritize a clear, interactive, and intuitive command-line experience.
  * **Modularity & Extensibility:** Design components to be easily extended, allowing for new LLM integrations, data sources, and output formats.
  * **Intelligence & Control:** Use LLMs intelligently by providing them with rich, well-structured data. Give the user full control over prompts, models, and the final output.
  * **Safety:** Cache downloaded data and generated content to avoid redundant processing and API calls.

### Technology Stack

  * **Language:** Python 3.8+
  * **CLI Framework:** **Typer**
  * **Dependency Management:** **Poetry (use `poetry add`, `poetry install`, `poetry run`; never `pip`)**
  * **Core Libraries:**
    * `requests` (HTTP requests)
    * `feedparser` (RSS/Atom feed parsing)
    * `rich` (advanced CLI formatting and progress bars)
    * `python-dotenv` (environment variable management)
    * `torch` & `transformers` (for local Hugging Face models)
    * Google & OpenAI Python SDKs (for API-based LLMs)
  * **Testing:** `pytest` with `mypy` (type checking).
  * **Code Quality:** `flake8`, `black`, `isort` (with pre-commit hooks).
  * **Configuration:** `.env` for API keys and settings.

-----

## 3. Git Commit Strategy

**The agent MUST create commits at the following major events.** Use conventional commit format:

```
<type>: <brief summary>

<detailed description of what was changed and why>
```

**Mandatory Commit Points:**

  * **Feature Development:**
      * `feat:` **Initial Implementation**: When core functionality is implemented (before tests).
      * `test:` **Test Implementation**: When tests are written and passing for the component.
      * `feat:` **Feature Integration**: Final commit when a feature is fully integrated.
  * **Other Major Events:**
      * `fix:` **Bug Resolutions**: After fixing any bugs.
      * `chore:` **Configuration Changes**: Updates to `.env`, `pyproject.toml`, etc.
      * `refactor(cli):` **CLI Refactoring**: For significant changes to the Typer interface.
      * `refactor(core):` **Core Logic Refactoring**: For major changes to the newsletter generation logic.
      * `deps:` **Dependency Management**: Adding/removing dependencies via Poetry.
      * `docs:` **Documentation Updates**: For changes to user or developer documentation.

-----

## 4. Code Style & Best Practices

### General Code Standards:

  * **PEP 8 Compliance:** Strictly adhere to PEP 8.
  * **Modularity:** Break down functionality into small, focused functions and classes. The core generation logic should be separate from the CLI interface.
  * **Docstrings:** All public functions, classes, and modules require clear docstrings.
  * **Error Handling:** Implement robust `try-except` blocks for all external interactions (file I/O, network/API/LLM calls).
  * **Logging:** Use Python's `logging` module for clear output.
  * **Environment Variables:** Always use `python-dotenv` for managing secrets and configuration.
  * **Poetry Usage:** Never reference `pip`. Always use Poetry commands (`poetry add`, `poetry install`, `poetry run`).

### Naming Conventions:

  * **Filenames:** `snake_case`. E.g., `cli.py`, `generator.py`.
  * **Functions/Variables:** `snake_case`.
  * **Classes:** `PascalCase`.

-----

## 5. Testing Requirements

  * All tests must be in the `/tests` folder.
  * **Unit Tests:** Should mock all external API calls and file system interactions.
  * **Integration Tests:** Can use real API keys (loaded from `.env`) for testing integrations with LLM providers, but should be clearly separated from unit tests.
  * **Running Tests:** Always use `poetry run pytest`.

-----

## 6. Specific AI Agent Instructions

### Development Workflow:

1.  **Understand the Goal:** Clarify the requirements for any new feature or refactoring task.
2.  **Plan the Changes:** Outline the steps, including which files will be modified.
3.  **Implement Incrementally:** Make small, logical changes and commit at each milestone.
4.  **Refactor to Typer:** The first major task is to replace the `argparse` implementation in `llm_newsletter_generator.py` with a Typer-based CLI.
5.  **Switch to Poetry:** The second major task is to initialize a Poetry project, add all dependencies, and remove `requirements.txt`.
6.  **Write Tests:** Add `pytest` tests for new functionality.