#!/usr/bin/env python3

import argparse
import requests
import feedparser
import hashlib
import os
import time
from transformers import BigBirdTokenizer, BigBirdForCausalLM

import tensorflow as tf

class NewsletterGenerator:
    def __init__(self, feed_url, cache_timeout=3600):
        self.feed_url = feed_url
        self.cache = {}
        self.tokenizer = BigBirdTokenizer.from_pretrained("google/bigbird-roberta-base")
        self.model = BigBirdForCausalLM.from_pretrained("google/bigbird-roberta-base", is_decoder=True)
        self.cache_timeout = cache_timeout

    def load_feed(self):
        """Loads the content of the provided feed URL with file caching.

        Returns:
            str: The content of the feed if successful, None otherwise.
        """
        cache_dir = "./cache/"
        cache_file = os.path.join(cache_dir, f"{hashlib.md5(self.feed_url.encode()).hexdigest()}.txt")

        # Check if cache directory exists, create it if not
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
        # Check if feed is cached and not expired
        if os.path.exists(cache_file):
            file_modified_time = os.path.getmtime(cache_file)
            if time.time() - file_modified_time < self.cache_timeout:
                print("Using cached feed")
                with open(cache_file, 'r') as f:
                    return f.read()

        try:
            response = requests.get(self.feed_url)
            if response.status_code == 200:
                feed_content = response.text
                # Cache the feed content to filesystem
                if not os.path.exists(cache_dir):
                    os.makedirs(cache_dir)
                with open(cache_file, 'w') as f:
                    f.write(feed_content)
                return feed_content
            else:
                print("Failed to load feed:", response.status_code)
                return None
        except Exception as e:
            print("Error loading feed:", str(e))
            return None

    def get_items(self, feed_content, max_size=3000):
        """Parses the feed content and retrieves its items.

        Args:
            feed_content (str): The content of the feed.
            max_size (int): The maximum size of the CSV data.

        Returns:
            list: A list of items parsed from the feed.
        """
        parsed_feed = feedparser.parse(feed_content)
        items = parsed_feed.entries

        # Truncate items to fit within the max_size limit
        csv_items = []
        current_size = 0
        for item in items:
            item_title = item.get('title', '')
            item_description = item.get('description', '')
            item_url = item.get('link', '')

            # Calculate the size of the item
            item_size = len(item_title) + len(item_description) + len(item_url)

            # If adding the current item exceeds the max_size, stop adding more items
            if current_size + item_size > max_size:
                break

            # Append the item to the CSV list
            csv_items.append((item_title, item_description, item_url))
            current_size += item_size

        return csv_items

    def extract_info(self, item):
        """Extracts title, description, and URL from the given feed item.

        Args:
            item (dict): The feed item.

        Returns:
            tuple: A tuple containing title, description, and URL of the feed item.
        """        
        title = item['title']
        description = item['description']
        url = item['link']
        return title, description, url

    def get_newsletter_text(self, items_csv, title, topic=None):
        """Generates text for the newsletter based on the provided items, title, and optional topic.

        Args:
            items_csv (str): A CSV formatted string containing item titles and links.
            title (str): The title of the newsletter.
            topic (str, optional): The topic of the newsletter.

        Returns:
            str: The generated text for the newsletter.
        """        
        prompt = f'I\'d like to create a newsletter with the title "{title}"\n'
        if topic:
            prompt += f'The topic is "{topic}"\n'
        prompt += '\nHere are the stories to base the newsletter on.\n'
        prompt += items_csv
        prompt += '\n\n'
        prompt += ('When writing, two factors are crucial, “perplexity” and “burstiness.” '
                   'Perplexity measures the complexity of the text. Separately, burstiness '
                   'compares the variations of sentences. Humans tend to write with greater '
                   'burstiness, for example, with some longer or more complex sentences alongside '
                   'shorter ones. AI sentences tend to be more uniform. Therefore, when writing '
                   'the following content I will ask you to create, I need it to have a good '
                   'amount of perplexity and burstiness.\n\n'
                   'Write in the business domain, intending to inform and describe and target a '
                   'general audience while maintaining a formal formality. Keep a clear narrative '
                   'flow, be natural-sounding narrative, and write at a college level.\n\n'
                   'Write a creative newsletter following these directions.\n')

        # Tokenize the input prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')

        # Generate text using the Longformer model
        output = self.model.generate(input_ids, max_length=4096, do_sample=True)

        # Decode the generated output
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text


def main():
    parser = argparse.ArgumentParser(description="Generate text-only newsletter from a feed")
    parser.add_argument("--feed", type=str, help="URL of the feed")
    parser.add_argument("--title", type=str, help="Title of the newsletter")
    parser.add_argument("--topic", type=str, help="Topic of the newsletter (optional)")
    args = parser.parse_args()

    if not args.feed or not args.title:
        parser.error("Please provide both --feed and --title arguments")

    generator = NewsletterGenerator(args.feed)
    feed_content = generator.load_feed()
    if feed_content:
        items = generator.get_items(feed_content)
        items_csv = '\n'.join([f"{item[0]}, {item[2]}" for item in items])
        newsletter_text = generator.get_newsletter_text(items_csv, args.title, args.topic)
        print(newsletter_text)
    else:
        print("Failed to generate newsletter")


if __name__ == "__main__":
    main()