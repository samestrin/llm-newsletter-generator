#!/usr/bin/env python3

"""
llm-newsletter-generator is an experimental Python script designed to generate
text-only newsletters from RSS feeds using AI via PyTorch and Transformers. AI
is used to create "compelling" newsletter content based on the provided feed,
title, and optional topic. llm-newsletter-generator currently processes
templated prompts using configurable LLMs and summarizes with
sshleifer/distilbart-cnn-12-6.

Copyright (c) 2024-PRESENT Sam Estrin
This script is licensed under the MIT License (see LICENSE for details)
GitHub: https://github.com/samestrin/newsletter-generator
"""

import logging
import requests
import feedparser
import hashlib
import os
import time
import typer
from transformers import pipeline, AutoTokenizer
from bs4 import BeautifulSoup
from rich.progress import Progress
from typing import Optional
from typing_extensions import Annotated

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Set the logging level to ERROR to suppress warnings and info messages from
# transformers and torch
logger = logging.getLogger("transformers")
logger.setLevel(logging.ERROR)

logger_torch = logging.getLogger("torch")
logger_torch.setLevel(logging.ERROR)

app = typer.Typer(help="Generate text-only newsletter from a feed")


class NewsletterGenerator:
    """
    A class designed to generate text-only newsletters from RSS feeds using AI.

    This class handles fetching RSS feed content, parsing it, generating text for
    newsletter sections (introduction, item stories, closing) using a specified
    Hugging Face model, and caching results to optimize performance.
    """

    def __init__(self, feed_url: str, cache_timeout: int = 3600, model_name: str = "default"):
        """
        Initializes the NewsletterGenerator with feed URL, cache timeout, and AI
        model configuration.

        Args:
            feed_url (str): The URL of the RSS feed to process.
            cache_timeout (int): The duration in seconds for which feed content is
                                 cached. Defaults to 3600 seconds (1 hour).
            model_name (str): The name of the Hugging Face model to use for text
                              generation. Supported models are defined in
                              `self.model_configs`. Defaults to 'microsoft'.
        """
        self.feed_url = feed_url
        self.cache_timeout = cache_timeout
        self.model_configs = {
            "microsoft": (
                "microsoft/Phi-3-mini-128k-instruct",
                "microsoft/Phi-3-mini-128k-instruct",
            ),
            "mistral": (
                "mistralai/Mistral-7B-Instruct-v0.2",
                "mistralai/Mistral-7B-Instruct-v0.2",
            ),
            "meta-llama": (
                "meta-llama/Meta-Llama-3-8B-Instruct",
                "meta-llama/Meta-Llama-3-8B-Instruct",
            ),
            "snowflake": (
                "Snowflake/snowflake-arctic-instruct",
                "Snowflake/snowflake-arctic-instruct",
            ),
            "tenyxchat": ("tenyx/Llama3-TenyxChat-70B", "tenyx/Llama3-TenyxChat-70B"),
            "dolphin": (
                "cognitivecomputations/dolphin-2.9-llama3-8b",
                "cognitivecomputations/dolphin-2.9-llama3-8b",
            ),
        }
        model, tokenizer = self.model_configs.get(model_name, self.model_configs["microsoft"])
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.text_generation = pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            trust_remote_code=True,
        )
        self.summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_feed(self) -> Optional[str]:
        """
        Loads the content of the provided feed URL with file caching.

        Checks if a cached version of the feed exists and is still valid (within
        `cache_timeout`). If not, it fetches the feed from the URL, caches it,
        and returns the content.

        Returns:
            str | None: The content of the feed if successful, None otherwise.
        """
        cache_dir = "./cache/"
        cache_file = os.path.join(
            cache_dir, f"{hashlib.md5(self.feed_url.encode()).hexdigest()}.txt"
        )

        if os.path.exists(cache_file):
            file_modified_time = os.path.getmtime(cache_file)
            if time.time() - file_modified_time < self.cache_timeout:
                self.logger.info("Using cached feed for %s", self.feed_url)
                with open(cache_file, "r") as f:
                    return f.read()

        try:
            response = requests.get(self.feed_url)
            if response.status_code == 200:
                feed_content = response.text
                with open(cache_file, "w") as f:
                    f.write(feed_content)
                self.logger.info("Successfully loaded and cached feed from %s", self.feed_url)
                return feed_content
            else:
                self.logger.error(
                    "Failed to load feed from %s: Status Code %s",
                    self.feed_url,
                    response.status_code,
                )
                return None
        except Exception as e:
            self.logger.error("Error loading feed from %s: %s", self.feed_url, e)
            return None

    def get_items(self, feed_content: str) -> list[tuple[str, str, str]]:
        """
        Parses the feed content and retrieves its items.

        Extracts the title, description, and link for each entry in the RSS feed.

        Args:
            feed_content (str): The content of the feed (XML/RSS format).

        Returns:
            list[tuple[str, str, str]]: A list of tuples, where each tuple represents
                                        an item with (title, description, link).
        """
        parsed_feed = feedparser.parse(feed_content)
        self.logger.info("Parsed %d items from the feed.", len(parsed_feed.entries))
        return [
            (item.get("title", ""), item.get("description", ""), item.get("link", ""))
            for item in parsed_feed.entries
        ]

    def generate_text(self, prompt: str) -> str:
        """
        Generates text based on the provided prompt using the configured AI model
        and caches the result.

        If a cached result for the given prompt and model exists, it's returned.
        Otherwise, text is generated using `self.text_generation` pipeline, cached,
        and then returned.

        Args:
            prompt (str): The prompt string to feed to the text generation model.

        Returns:
            str: The generated text, with the original prompt removed and whitespace
                 trimmed.
        """

        cache_dir = "./cache/"
        cache_key = self.model + " " + prompt
        cache_file = os.path.join(cache_dir, f"{hashlib.md5(cache_key.encode()).hexdigest()}.txt")
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                self.logger.info(
                    "Using cached generated text for prompt hash %s",
                    hashlib.md5(cache_key.encode()).hexdigest(),
                )
                return f.read()

        self.logger.info("Generating text for prompt: %s", prompt[:100] + "...")
        generated_text = self.text_generation(prompt, max_new_tokens=2046, do_sample=True)[0][
            "generated_text"
        ]

        # Remove the original prompt from the generated text
        generated_text = generated_text.replace(prompt, "")

        # Trim whitespace characters from both ends of the generated text
        generated_text = generated_text.strip()

        with open(cache_file, "w") as f:
            f.write(generated_text)
        self.logger.info(
            "Generated text and cached for prompt hash %s",
            hashlib.md5(cache_key.encode()).hexdigest(),
        )

        return generated_text

    def load_template(self, template_path: str) -> str:
        """
        Loads template content from a specified file.

        Args:
            template_path (str): The absolute or relative path to the template file.

        Returns:
            str: The content of the template file.

        Raises:
            FileNotFoundError: If the template file does not exist.
            Exception: For other errors during file reading.
        """
        try:
            with open(template_path, "r") as file:
                template_content = file.read()
            self.logger.info("Loaded template from %s", template_path)
            return template_content
        except FileNotFoundError:
            self.logger.error("Template file not found: %s", template_path)
            raise
        except Exception as e:
            self.logger.error("Error loading template %s: %s", template_path, e)
            raise

    def generate_prompt(
        self,
        title: str,
        topic: Optional[str],
        row_titles: list[str],
        section: str,
        max_tokens: int = 768,
    ) -> str:
        """
        Generates a prompt for the AI model to create introduction, story
        introductions, or closing sections.

        This method interpolates provided data (newsletter title, topic, and
        article titles) into a template specific to the requested section
        (e.g., 'introduction', 'closing'). It also handles token limits for
        `row_titles`.

        Args:
            title (str): The main title of the newsletter.
            topic (str | None): The specific topic of the newsletter. If None,
                                `title` is used.
            row_titles (list[str]): A list of titles of each news item to be
                                    included in the prompt.
            section (str): The section of the newsletter for which to generate the
                           prompt (e.g., 'introduction', 'closing').
            max_tokens (int): Maximum number of tokens allowed for the `row_titles`
                              in the prompt. Defaults to 768.

        Returns:
            str: A generated prompt string suitable for text generation.

        Raises:
            ValueError: If an invalid `section` is provided.
        """
        if section not in ["introduction", "closing"]:
            self.logger.error(
                "Invalid section for prompt generation: %s. Must be "
                "'introduction' or 'closing'.",
                section,
            )
            raise ValueError("Section must be 'introduction' or 'closing'.")

        # Load template content
        template_path = os.path.join("prompts", f"{section}.md")
        template_content = self.load_template(template_path)

        # Interpolate variables into the template
        topic = topic or title
        prompt = template_content.replace("{{ title }}", title)
        prompt = template_content.replace("{{ topic }}", topic)

        rowTitles = ""
        current_token_count = 0
        for r_title in row_titles:
            tokens = self.tokenizer.encode(r_title, add_special_tokens=True)
            if current_token_count + len(tokens) > max_tokens:
                self.logger.warning(
                    "Truncating row titles for prompt due to token "
                    "limit. Current tokens: %d, Max tokens: %d",
                    current_token_count,
                    max_tokens,
                )
                break
            rowTitles += r_title + "\n"
            current_token_count += len(tokens)

        prompt = template_content.replace("{{ row_titles }}", rowTitles)
        self.logger.info("Generated prompt for section: %s", section)
        return prompt

    def generate_prompt_for_item(
        self,
        item: tuple[str, str, str],
        topic: Optional[str],
        estimated_tokens: int = 768,
    ) -> str:
        """
        Generates a prompt for the AI model to write a story introduction based on
        a single news item.

        This method cleans the item description, summarizes it if too long, and then
        interpolates the item's details into a specific template for individual stories.

        Args:
            item (tuple[str, str, str]): A news item containing (title, description, URL).
            topic (str | None): The overall topic of the newsletter.
            estimated_tokens (int): An estimated maximum number of tokens for the item
                                    description. If the description exceeds this, it will
                                    be summarized. Defaults to 768.

        Returns:
            str: The generated prompt string suitable for text generation for a single item.
        """
        title, description, url = item

        soup = BeautifulSoup(description, "html.parser")
        cleaned_description = str(soup.get_text())

        tokens = self.tokenizer.encode(cleaned_description)

        if len(tokens) > estimated_tokens:
            self.logger.info("Description too long (%d tokens), summarizing...", len(tokens))
            summary = self.summarizer(
                cleaned_description, max_length=1024, min_length=800, do_sample=False
            )
            summary_description = summary[0]["summary_text"] if summary else cleaned_description
            self.logger.info("Summary generated for item: %s", title)
        else:
            summary_description = cleaned_description
            self.logger.info("Using full description for item: %s", title)

        # Load item template
        template_path = os.path.join("prompts", "item.md")
        template_content = self.load_template(template_path)

        # Interpolate variables into the template
        prompt = template_content.replace("{{ item_title }}", title)
        prompt = template_content.replace("{{ item_description }}", summary_description)
        prompt = template_content.replace("{{ topic }}", topic if topic is not None else "")
        self.logger.info("Generated prompt for item: %s", title)
        return prompt

    def create_newsletter(
        self, title: str, topic: Optional[str], items: list[tuple[str, str, str]]
    ) -> str:
        """
        Creates a complete newsletter by generating text for introduction, each item,
        and closing.

        This method orchestrates the generation process, displaying progress using
        `rich.progress`, and combines all generated sections into a single
        newsletter text.

        Args:
            title (str): The main title of the newsletter.
            topic (str | None): The overall topic of the newsletter.
            items (list[tuple[str, str, str]]): A list of news items, each as a tuple
                                                (title, description, URL).

        Returns:
            str: The complete newsletter text.
        """
        self.logger.info("Starting newsletter creation for title: %s, topic: %s", title, topic)
        # Calculate the total number of tasks including the introduction, each item,
        # and closing
        total_tasks = 4 + len(items)

        with Progress() as progress:
            # Add a task for generating the newsletter with the total number of tasks
            task1 = progress.add_task("[cyan]Generating newsletter...", total=total_tasks)

            newsletter_output = []
            row_titles = [item[0] for item in items]

            progress.update(task1, advance=1, description="[cyan]Generating introduction...")
            intro_prompt = self.generate_prompt(title, topic, row_titles, "introduction")
            introduction = self.generate_text(intro_prompt)
            newsletter_output.append(introduction)
            self.logger.info("Generated introduction.")

            progress.update(task1, advance=1, description="[cyan]Generating item stories...")
            # Loop through each item, updating progress to show which story is being
            # generated
            for index, item in enumerate(items, start=1):
                progress.update(
                    task1,
                    advance=1,
                    description=f"[cyan]Generating story {index}/{len(items)}...",
                )
                self.logger.info("Generating story for item %d/%d: %s", index, len(items), item[0])
                story_prompt = self.generate_prompt_for_item(item, topic)
                story = self.generate_text(story_prompt)
                newsletter_output.append(story)
                self.logger.info("Generated story for item %d/%d.", index, len(items))

            progress.update(task1, advance=1, description="[cyan]Generating closing...")
            closing_prompt = self.generate_prompt(title, topic, row_titles, "closing")
            closing = self.generate_text(closing_prompt)
            newsletter_output.append(closing)
            self.logger.info("Generated closing.")

            progress.update(task1, advance=1, description="[cyan]Finalizing newsletter...")
            self.logger.info("Newsletter creation complete.")
            return "\n\n".join(newsletter_output)


def _version_callback(value: bool) -> None:
    """
    Callback function for the --version Typer option.

    Args:
        value (bool): True if the --version flag is present, False otherwise.

    Raises:
        typer.Exit: Exits the application after printing the version.
    """
    if value:
        try:
            with open(".version", "r") as file:
                logging.info("Version: %s", file.read().strip())
        except FileNotFoundError:
            logging.warning("Version file not found.")
        raise typer.Exit()


@app.command()
def main(
    feed_url: Annotated[str, typer.Option(help="URL of the feed")],
    title: Annotated[str, typer.Option(help="Title of the newsletter")],
    topic: Annotated[
        Optional[str], typer.Option(help="Topic of the newsletter " "(optional)")
    ] = None,
    max_items: Annotated[
        Optional[int],
        typer.Option("--max", help="Maximum number " "of items to process " "(optional)"),
    ] = None,
    model_name: Annotated[
        str,
        typer.Option(
            "-m",
            "--model-name",
            help="Model to "
            "use for text generation (microsoft, "
            "meta-llama, snowflake, dolphin)",
        ),
    ] = "microsoft",
    output_filename: Annotated[
        Optional[str],
        typer.Option("-o", "--output-filename", help="Output filename " "(optional)"),
    ] = None,
    version: Annotated[
        bool,
        typer.Option(
            "-v",
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Display the version number",
        ),
    ] = False,
):
    """
    Main function to handle command line arguments and initiate newsletter generation.

    This function serves as the entry point for the CLI application. It parses
    command-line arguments, initializes the NewsletterGenerator, and orchestrates
    the newsletter creation process.
    It also tracks and logs the total runtime
    of the process.

    Args:
        feed_url (str): The URL of the RSS feed to process.
        title (str): The title of the newsletter.
        topic (str | None): The topic of the newsletter (optional). If not provided,
                            the title is used.
        max_items (int | None): Maximum number of items to process from the feed
                                (optional).
        model_name (str): The name of the AI model to use for text generation.
                          Defaults to 'microsoft'.
        output_filename (str | None): The name of the file to write the newsletter to
                                      (optional). If not provided, the newsletter is
                                      printed to stdout.
        version (bool): If True, displays the version number and exits.
    """

    start_time = time.time()
    logging.info("Starting newsletter generation process.")

    # Create a cache directory if it's not already created
    cache_dir = "./cache/"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        logging.info("Created cache directory: %s", cache_dir)

    generator = NewsletterGenerator(feed_url, model_name=model_name)
    feed_content = generator.load_feed()
    if feed_content:
        items = generator.get_items(feed_content)

        if max_items:
            items = items[:max_items]
            logging.info("Processing %d items (max_items set to %d).", len(items), max_items)

        newsletter_text = generator.create_newsletter(title, topic, items)

        if output_filename:
            with open(output_filename, "w") as file:
                file.write(newsletter_text)
            logging.info("Newsletter written to %s", output_filename)
        else:
            logging.info("\n%s", newsletter_text)  # Print to stdout if no output file
    else:
        logging.error("Failed to generate newsletter.")

    # Print runtime
    logging.info("Total runtime: %.2f seconds", time.time() - start_time)


if __name__ == "__main__":
    app()
