# newsletter-generator

[![Star on GitHub](https://img.shields.io/github/stars/samestrin/newsletter-generator?style=social)](https://github.com/samestrin/newsletter-generator/stargazers) [![Fork on GitHub](https://img.shields.io/github/forks/samestrin/newsletter-generator?style=social)](https://github.com/samestrin/newsletter-generator/network/members) [![Watch on GitHub](https://img.shields.io/github/watchers/samestrin/newsletter-generator?style=social)](https://github.com/samestrin/newsletter-generator/watchers)

![Version 0.0.4](https://img.shields.io/badge/Version-0.0.4-blue) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Built with Python](https://img.shields.io/badge/Built%20with-Python-green)](https://www.python.org/)

newsletter-generator is an experimental Python script designed to generate text-only newsletters from RSS feeds using AI. The Transformers library is used to create "compelling" newsletter content based on the provided feed, title, and optional topic. newsletter-generator currently processes prompts using configurable LLMs and summarizes with [sshleifer/distilbart-cnn-12-6](https://huggingface.co/sshleifer/distilbart-cnn-12-6).

## Features

- **RSS Feed Integration**: The generator seamlessly retrieves content from RSS feeds, ensuring up-to-date newsletter content.
- **Customizable Output**: Users can specify the title and topic of the newsletter to tailor content according to their preferences.
- **Transformer-based Text Generation**: Utilizing the powerful capabilities of the Transformers library, the generator produces engaging and diverse newsletter content using multiple models.

### Large Language Models

- **Dolphin 2.9 Llama 3 8b**: [cognitivecomputations/dolphin-2.9-llama3-8b](https://huggingface.co/cognitivecomputations/dolphin-2.9-llama3-8b)
- **Meta Llama 3**: [meta-llama/Meta-Llama-3-8B]()
- **Microsoft Phi-3-Mini-128K-Instruct**: [microsoft/Phi-3-mini-128k-instruct](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct) (Default)
- **Snowflake Artic**: [Snowflake/snowflake-arctic-instruct](https://huggingface.co/Snowflake/snowflake-arctic-instruct)

## Dependencies

The Newsletter Generator relies on the following dependencies:

- **argparse**: For parsing command-line arguments.
- **requests**: To make HTTP requests to retrieve the RSS feeds.
- **feedparser**: To parse the RSS feeds.
- **hashlib**: For generating hash values, used in caching mechanisms.
- **os**: To interact with the operating system, such as file path handling and checking file existence.
- **time**: To handle time-based functions, like checking cache timeouts.
- **torch**: PyTorch, used by the `transformers` library for managing deep learning models.
- **transformers**: From Hugging Face, used to load pre-trained models and pipelines for natural language processing tasks.
- **bs4 (BeautifulSoup)**: For parsing HTML content in descriptions to clean it.
- **rich**: Used for creating progress bars and rich text formatting in the console.

### Installation

To install and use newsletter-generator, follow these steps:

Clone the Repository: Begin by cloning the repository containing the newsletter-generator to your local machine.

```bash
git clone https://github.com/samestrin/newsletter-generator/
```

Navigate to the project directory:

```bash
cd newsletter-generator
```

Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

## Usage

To run the script, you need to provide two mandatory arguments: the feed (`--feed`) and the title of the newsletter (`--title`).

```bash
python newsletter_generator.py --feed <feed_url> --title <newsletter_title> [--topic <newsletter_topic>] [--max <max_items>] [--output <output_filename>] [--model <valid_model>]
```

Replace <feed_url> with the URL of the RSS feed you want to generate the newsletter from. Specify <newsletter_title> as the desired title for the newsletter. Optionally, you can include <newsletter_topic> to define a specific topic for the newsletter, <max_items> to limit the number of items included in your newsletter, <output> to specify an output filename, or <valid_model>. Valid models are: `dolphin`, `meta-llama`, `microsoft`, and `snowflake`.

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes or improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
