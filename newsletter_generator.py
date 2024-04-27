#!/usr/bin/env python3

import argparse
import requests
import feedparser
import hashlib
import os
import time
import torch
from transformers import GPT2Tokenizer, GPTNeoForCausalLM, pipeline
from bs4 import BeautifulSoup
from rich.progress import Progress


class NewsletterGenerator:
    def __init__(self, feed_url, cache_timeout=3600):
        """
        Initializes the NewsletterGenerator.

        Args:
            feed_url (str): URL of the feed to parse.
            cache_timeout (int): Timeout for the cache in seconds.
        """
        self.feed_url = feed_url
        self.cache = {}
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
        self.model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
        self.cache_timeout = cache_timeout
        self.summarizer = pipeline(
            "summarization", model="sshleifer/distilbart-cnn-12-6"
        )

    def load_feed(self):
        """
        Loads the content of the provided feed URL with file caching.

        Returns:
            str: The content of the feed if successful, None otherwise.
        """
        cache_dir = "./cache/"
        cache_file = os.path.join(
            cache_dir, f"{hashlib.md5(self.feed_url.encode()).hexdigest()}.txt"
        )

        if os.path.exists(cache_file):
            file_modified_time = os.path.getmtime(cache_file)
            if time.time() - file_modified_time < self.cache_timeout:
                print("Using cached feed")
                with open(cache_file, "r") as f:
                    return f.read()

        try:
            response = requests.get(self.feed_url)
            if response.status_code == 200:
                feed_content = response.text
                with open(cache_file, "w") as f:
                    f.write(feed_content)
                return feed_content
            else:
                print("Failed to load feed:", response.status_code)
                return None
        except Exception as e:
            print("Error loading feed:", str(e))
            return None

    def get_items(self, feed_content):
        """
        Parses the feed content and retrieves its items.

        Args:
            feed_content (str): The content of the feed.

        Returns:
            list: A list of items parsed from the feed.
        """
        parsed_feed = feedparser.parse(feed_content)
        return [
            (item.get("title", ""), item.get("description", ""), item.get("link", ""))
            for item in parsed_feed.entries
        ]

    def generate_text(self, prompt):
        """
        Generates text based on the provided prompt using GPT-Neo 1.3B and caches the result.

        Args:
            prompt (str): Prompt to feed to the GPT-Neo 1.3B model.

        Returns:
            str: The generated text.
        """
        cache_dir = "./cache/"
        cache_file = os.path.join(
            cache_dir, f"{hashlib.md5(prompt.encode()).hexdigest()}.txt"
        )
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                return f.read()

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")

        attention_mask = torch.ones(
            input_ids.shape, dtype=torch.long
        )  # Ensure all tokens are attended to
        pad_token_id = (
            self.tokenizer.eos_token_id
        )  # EOS token is used as pad token in GPT-Neo 1.3B
        output = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            pad_token_id=pad_token_id,
            max_length=1024,
            do_sample=True,
        )
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        with open(cache_file, "w") as f:
            f.write(generated_text)

        return generated_text

    def generate_prompt(self, title, topic, row_titles, section, max_tokens=768):
        """
        Generates a prompt for the GPT-Neo 1.3B model to create the introduction, story introductions, or closing of the newsletter.

        Args:
            title (str): The title of the newsletter.
            topic (str): The topic of the newsletter.
            row_titles (list of str): A list of titles of each news item.
            section (str): The section of the newsletter for which to generate the prompt (e.g., 'introduction', 'closing').
            max_tokens (int): Maximum number of tokens allowed for the row titles in the prompt.

        Returns:
            str: A generated prompt suitable for text generation.
        """
        if section not in ["introduction", "closing"]:
            raise ValueError("Section must be 'introduction' or 'closing'.")

        # Initialize the row titles string and token count
        rowTitles = ""
        current_token_count = 0

        # Loop through each title and append it if it doesn't exceed the token limit
        for title in row_titles:
            tokens = self.tokenizer.encode(title, add_special_tokens=True)
            if current_token_count + len(tokens) > max_tokens:
                break
            rowTitles += title + "\n"
            current_token_count += len(tokens)

        prompt = f"{section.capitalize()} for the newsletter titled '{title}' on the topic '{topic}'.\n"
        prompt += "Featured articles include:\n" + rowTitles + "\n"
        if section == "introduction":
            prompt += "Please write an engaging introduction that sets the stage for the following articles."
        else:
            prompt += (
                "Please summarize the key points and conclude the newsletter elegantly."
            )

        return prompt

    def generate_prompt_for_item(self, item, topic, estimated_tokens=768):
        """
        Generates a prompt for GPT-Neo 1.3B to write a story introduction based on a single news item.

        Args:
            item (tuple): A news item containing title, description, and URL.
            topic (str): Topic of the newsletter.

        Returns:
            str: The generated prompt.
        """
        title, description, url = item

        # Remove all tags from the description
        soup = BeautifulSoup(description, "html.parser")
        cleaned_description = str(soup.get_text())

        # Tokenize the cleaned description to check its length
        tokens = self.tokenizer.encode(cleaned_description)

        # Check if description is longer than estimated_tokens tokens, if so, summarize it
        if len(tokens) > estimated_tokens:
            # Attempt to summarize to about estimated_tokens tokens (exact control of token length is hard in summarization)
            summary = self.summarizer(
                cleaned_description, max_length=1024, min_length=800, do_sample=False
            )
            summary_description = (
                summary[0]["summary_text"] if summary else cleaned_description
            )
        else:
            summary_description = cleaned_description

        prompt = f"Create a compelling story introduction for an article titled '{title}' which is part of a newsletter on the topic '{topic}'.\n"
        prompt += f"Description: {summary_description}\nURL: {url}\n\n"
        prompt += "Please write an engaging introduction that captures the essence of the article."
        return prompt

    def create_newsletter(self, title, topic, items):
        """
        Creates a newsletter with generated text for introduction, each item, and closing,
        displaying progress with visual feedback.

        Args:
            title (str): Title of the newsletter.
            topic (str): Topic of the newsletter.
            items (list): List of news items (title, description, URL).

        Returns:
            str: The complete newsletter text.
        """
        # Calculate the total number of tasks including the introduction, each item, and closing
        total_tasks = 4 + len(items)

        with Progress() as progress:
            # Add a task for generating the newsletter with the total number of tasks
            task1 = progress.add_task("[cyan]Generating newsletter...", total=total_tasks)

            newsletter_output = []
            row_titles = [item[0] for item in items]

            progress.update(
                task1, advance=1, description="[cyan]Generating introduction..."
            )
            intro_prompt = self.generate_prompt(
                title, topic, row_titles, "introduction"
            )
            introduction = self.generate_text(intro_prompt)
            newsletter_output.append(introduction)

            progress.update(
                task1, advance=1, description="[cyan]Generating item stories..."
            )
            # Loop through each item, updating progress to show which story is being generated
            for index, item in enumerate(items, start=1):
                progress.update(
                    task1,
                    advance=1,
                    description=f"[cyan]Generating story {index}/{len(items)}..."
                )
                story_prompt = self.generate_prompt_for_item(item, topic)
                story = self.generate_text(story_prompt)
                newsletter_output.append(story)

            progress.update(task1, advance=1, description="[cyan]Generating closing...")
            closing_prompt = self.generate_prompt(title, topic, row_titles, "closing")
            closing = self.generate_text(closing_prompt)
            newsletter_output.append(closing)

            progress.update(
                task1, advance=1, description="[cyan]Finalizing newsletter..."
            )
            return "\n\n".join(newsletter_output)


def main():
    """
    Main function to handle command line arguments and initiate newsletter generation.
    Tracks the total runtime of the newsletter generation process.
    """
    parser = argparse.ArgumentParser(description="Generate text-only newsletter from a feed")
    parser.add_argument("--feed", type=str, help="URL of the feed")
    parser.add_argument("--title", type=str, help="Title of the newsletter")
    parser.add_argument("--topic", type=str, help="Topic of the newsletter (optional)")
    args = parser.parse_args()

    if not args.feed or not args.title:
        parser.error("Please provide both --feed and --title arguments")


    # Start timing the newsletter generation process
    start_time = time.time()

    # Create a cache directory if it doesn't exist
    cache_dir = "./cache/"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    generator = NewsletterGenerator(args.feed)
    feed_content = generator.load_feed()
    if feed_content:
        items = generator.get_items(feed_content)
        newsletter_text = generator.create_newsletter(args.title, args.topic, items)
        print(newsletter_text)
    else:
        print("Failed to generate newsletter")

    # End timing and calculate the elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\n\nTotal runtime: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()
