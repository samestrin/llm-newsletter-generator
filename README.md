# llm-newsletter-generator

[![Star on GitHub](https://img.shields.io/github/stars/samestrin/llm-newsletter-generator?style=social)](https://github.com/samestrin/llm-newsletter-generator/stargazers) [![Fork on GitHub](https://img.shields.io/github/forks/samestrin/llm-newsletter-generator?style=social)](https://github.com/samestrin/llm-newsletter-generator/network/members) [![Watch on GitHub](https://img.shields.io/github/watchers/samestrin/llm-newsletter-generator?style=social)](https://github.com/samestrin/llm-newsletter-generator/watchers)

![Version 0.0.4](https://img.shields.io/badge/Version-0.0.4-blue) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Built with Python](https://img.shields.io/badge/Built%20with-Python-green)](https://www.python.org/)

**llm-newsletter-generator** is an _experimental_ Python script designed to generate text-only newsletters from RSS feeds using AI via PyTorch and Transformers. AI is used to create "compelling" newsletter content based on the provided feed, title, and optional topic. llm-newsletter-generator currently processes templated prompts using configurable LLMs and summarizes with [sshleifer/distilbart-cnn-12-6](https://huggingface.co/sshleifer/distilbart-cnn-12-6).

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

llm-newsletter-generator relies on the following dependencies:

- **Python**: The script runs in a Python3 environment.
- **argparse**: For parsing command-line arguments.
- **bs4 (BeautifulSoup)**: For parsing HTML content in descriptions to clean it.
- **feedparser**: To parse the RSS feeds.
- **hashlib**: For generating hash values, used in caching mechanisms.
- **os**: To interact with the operating system, such as file path handling and checking file existence.
- **requests**: To make HTTP requests to retrieve the RSS feeds.
- **rich**: Used for creating progress bars and rich text formatting in the console.
- **time**: To handle time-based functions, like checking cache timeouts.
- **torch**: PyTorch, used by the `transformers` library for managing deep learning models.
- **transformers**: From Hugging Face, used to load pre-trained models and pipelines for natural language processing tasks.

### Installation

To install and use llm-newsletter-generator, follow these steps:

Clone the Repository: Begin by cloning the repository containing the llm-newsletter-generator to your local machine.

```bash
git clone https://github.com/samestrin/llm-newsletter-generator/
```

Navigate to the project directory:

```bash
cd llm-newsletter-generator
```

Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

## Usage

To run the script, you need to provide two mandatory arguments: the feed (`--feed-url`) and the title of the newsletter (`--title`).

```bash
python llm_newsletter_generator.py --feed-url <feed_url> --title <newsletter_title> [--topic <newsletter_topic>] [--max <max_items>] [--output-filename <output_filename>] [--model <valid_model>]
```

## Options

```
  -f, --feed-url          URL of the feed                 [string] [required]
  -t, --title             Title of the newsletter         [string] [required]
  -to, --topic            Topic of the newsletter         [string]
  --max                   Maximum number of items to      [number]
                          process
  -m, --model-name        Model to use for text           [string]
                          generation (microsoft,
                          meta-llama, snowflake,
                          dolphin)
  -o, --output-filename   Output filename                 [string]

  -v, --version           Display the version number      [boolean]
```

Replace <feed_url> with the URL of the RSS feed you want to generate the newsletter from. Specify <newsletter_title> as the desired title for the newsletter. Optionally, you can include <newsletter_topic> to define a specific topic for the newsletter, <max_items> to limit the number of items included in your newsletter, <output> to specify an output filename, or <valid_model>. Valid models are: `dolphin`, `meta-llama`, `microsoft`, and `snowflake`; `microsoft` is the default.

## Contribute

Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes or improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Share

[![Twitter](https://img.shields.io/badge/X-Tweet-blue)](https://twitter.com/intent/tweet?text=Check%20out%20this%20awesome%20project!&url=https://github.com/samestrin/llm-newsletter-generator) [![Facebook](https://img.shields.io/badge/Facebook-Share-blue)](https://www.facebook.com/sharer/sharer.php?u=https://github.com/samestrin/llm-newsletter-generator) [![LinkedIn](https://img.shields.io/badge/LinkedIn-Share-blue)](https://www.linkedin.com/sharing/share-offsite/?url=https://github.com/samestrin/llm-newsletter-generator)
