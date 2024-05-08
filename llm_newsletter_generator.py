#!/usr/bin/env python3

"""
llm-newsletter-generator is an experimental Python script designed to generate text-only newsletters from RSS feeds using AI via 
PyTorch and Transformers. AI is used to create "compelling" newsletter content based on the provided feed, title, and optional topic. 
llm-newsletter-generator currently processes templated prompts using configurable LLMs and summarizes with 
sshleifer/distilbart-cnn-12-6.

Copyright (c) 2024-PRESENT Sam Estrin
This script is licensed under the MIT License (see LICENSE for details)
GitHub: https://github.com/samestrin/newsletter-generator
"""

import argparse
import requests
import feedparser
import hashlib
import sys
import os
import time
import torch
from transformers import pipeline, AutoTokenizer
from bs4 import BeautifulSoup
from rich.progress import Progress

class CustomHelpFormatter(argparse.HelpFormatter):
    """
    Custom help formatter class for argparse to output arguments like npm yargs.
    """
    def _format_action_invocation(self, action):
        if not action.option_strings:
            metavar, = self._metavar_formatter(action, action.dest)(1)
            return metavar
        else:
            parts = []
            # Display all option strings and show defaults if present
            parts.extend(action.option_strings)
            show_default = ' [default: %(default)s]' if 'default' in action.__dict__ else ''
            return '%s %s%s' % (', '.join(parts), self._format_args(action, action.dest), show_default)

    def _split_lines(self, text, width):
        # This method overrides the default line splitter to change how help strings are displayed.
        return text.splitlines()

class NewsletterGenerator:
    """
    NewsletterGenerator class designed to generate text-only newsletters from RSS feeds using AI.
    """ 
    def __init__(self, feed_url, cache_timeout=3600, model_name='default'):
        self.feed_url = feed_url
        self.cache_timeout = cache_timeout
        self.model_configs = {            
            'microsoft': ("microsoft/Phi-3-mini-128k-instruct", "microsoft/Phi-3-mini-128k-instruct"),
            'mistral': ("mistralai/Mistral-7B-Instruct-v0.2", "mistralai/Mistral-7B-Instruct-v0.2"),
            'meta-llama': ("meta-llama/Meta-Llama-3-8B-Instruct", "meta-llama/Meta-Llama-3-8B-Instruct"),
            'snowflake': ("Snowflake/snowflake-arctic-instruct", "Snowflake/snowflake-arctic-instruct"),
            'tenyxchat': ("tenyx/Llama3-TenyxChat-70B", "tenyx/Llama3-TenyxChat-70B"),
            'dolphin': ("cognitivecomputations/dolphin-2.9-llama3-8b", "cognitivecomputations/dolphin-2.9-llama3-8b")
        }
        model, tokenizer = self.model_configs.get(model_name, self.model_configs['microsoft'])
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.text_generation = pipeline(
            "text-generation", 
            model=model, 
            tokenizer=self.tokenizer,
            trust_remote_code=True
        )
        self.summarizer = pipeline(
            "summarization", 
            model="sshleifer/distilbart-cnn-12-6"
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
        Generates text based on the provided prompt using tiiuae/falcon-7b-instruct and caches the result.

        Args:
            prompt (str): Prompt to feed to the tiiuae/falcon-7b-instruct model.

        Returns:
            str: The generated text.
        """
        
        cache_dir = "./cache/"
        cache_key = self.model + " " + prompt
        cache_file = os.path.join(
            cache_dir, f"{hashlib.md5(cache_key.encode()).hexdigest()}.txt"
        )
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                return f.read()
        
        generated_text = self.text_generation(prompt, max_new_tokens=2046, do_sample=True)[0]["generated_text"]

        # Remove the original prompt from the generated text
        generated_text = generated_text.replace(prompt, "")       

        # Trim whitespace characters from both ends of the generated text
        generated_text = generated_text.strip()         
        
        with open(cache_file, "w") as f:
            f.write(generated_text)

        return generated_text

    def load_template(self, template_path):
        """
        Load template content from a file and return it.

        Args:
            template_path (str): Path to the template file.

        Returns:
            str: Content of the template file.
        """
        with open(template_path, "r") as file:
            template_content = file.read()
                    
        return template_content

    def generate_prompt(self, title, topic, row_titles, section, max_tokens=768):
        """
        Generates a prompt for the tiiuae/falcon-7b-instruct model to create the introduction, story introductions, or closing of the newsletter.

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

        # Load template content
        template_path = os.path.join("prompts", f"{section}.md")
        template_content = self.load_template(template_path)

        # Interpolate variables into the template
        prompt = template_content.replace("{{ title }}", title)
        prompt = prompt.replace("{{ topic }}", topic)

        rowTitles = ""
        current_token_count = 0
        for title in row_titles:
            tokens = self.tokenizer.encode(title, add_special_tokens=True)
            if current_token_count + len(tokens) > max_tokens:
                break
            rowTitles += title + "\n"
            current_token_count += len(tokens)

        prompt = prompt.replace("{{ row_titles }}", rowTitles)

        return prompt

    def generate_prompt_for_item(self, item, topic, estimated_tokens=768):
        """
        Generates a prompt for tiiuae/falcon-7b-instruct to write a story introduction based on a single news item.

        Args:
            item (tuple): A news item containing title, description, and URL.
            topic (str): Topic of the newsletter.

        Returns:
            str: The generated prompt.
        """
        title, description, url = item

        soup = BeautifulSoup(description, "html.parser")
        cleaned_description = str(soup.get_text())

        tokens = self.tokenizer.encode(cleaned_description)

        if len(tokens) > estimated_tokens:
            summary = self.summarizer(
                cleaned_description, max_length=1024, min_length=800, do_sample=False
            )
            summary_description = (
                summary[0]["summary_text"] if summary else cleaned_description
            )
        else:
            summary_description = cleaned_description

        # Load item template
        template_path = os.path.join("prompts", "item.md")
        template_content = self.load_template(template_path)

        # Interpolate variables into the template
        prompt = template_content.replace("{{ item_title }}", title)
        prompt = prompt.replace("{{ item_description }}", summary_description)
        prompt = prompt.replace("{{ topic }}", topic)

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
                topic = topic or title
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

    start_time = time.time() 
    
    # Early check for the version argument
    if '-v' in sys.argv or '--version' in sys.argv:
        try:
            with open(".version", "r") as file:
                print(file.read().strip())
        except FileNotFoundError:
            print("Version file not found.")
        sys.exit()

    parser = argparse.ArgumentParser(description="Generate text-only newsletter from a feed", formatter_class=CustomHelpFormatter)
    parser.add_argument("-f", "--feed-url", type=str, required=True, help="URL of the feed")
    parser.add_argument("-t", "--title", type=str, required=True, help="Title of the newsletter")
    parser.add_argument("-to", "--topic", type=str, help="Topic of the newsletter (optional)")
    parser.add_argument("--max", type=int, help="Maximum number of items to process (optional)")
    parser.add_argument("-m", "--model-name", type=str, default='microsoft', help="Model to use for text generation (microsoft, meta-llama, snowflake, dolphin)")
    parser.add_argument("-o", "--output-filename", type=str, help="Output filename (optional)")
    parser.add_argument("-v", "--version", action='store_true', help="Display the version number")

    args = parser.parse_args()

    # Create a cache directory if it doesn't exist
    cache_dir = "./cache/"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    

    generator = NewsletterGenerator(args.feed_url, model_name=args.model_name)
    feed_content = generator.load_feed()
    if feed_content:
        items = generator.get_items(feed_content)
        
        if args.max:
            items = items[:args.max]

        newsletter_text = generator.create_newsletter(args.title, args.topic, items)
        
        if args.output:
            with open(args.output, "w") as file:
                file.write(newsletter_text)
            print(f"Newsletter written to {args.output_filename}")
        else:
            print(newsletter_text)
    else:
        print("Failed to generate newsletter")

    # Print runtime
    print(f"\n\nTotal runtime: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
