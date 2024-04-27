# newsletter-generator

[![Star on GitHub](https://img.shields.io/github/stars/samestrin/newsletter-generator?style=social)](https://github.com/samestrin/newsletter-generator/stargazers) [![Fork on GitHub](https://img.shields.io/github/forks/samestrin/newsletter-generator?style=social)](https://github.com/samestrin/newsletter-generator/network/members) [![Watch on GitHub](https://img.shields.io/github/watchers/samestrin/newsletter-generator?style=social)](https://github.com/samestrin/newsletter-generator/watchers)

![Version 0.0.2](https://img.shields.io/badge/Version-0.0.2-blue) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Built with Python](https://img.shields.io/badge/Built%20with-Python-green)](https://www.python.org/)

newsletter-generator is an experimental Python script designed to generate text-only newsletters from RSS feeds using AI. The Transformers library is used to create "compelling" newsletter content based on the provided feed, title, and optional topic. newsletter-generator currently processes prompts using [GPT-Neo 1.3B](https://huggingface.co/EleutherAI/gpt-neo-1.3B) and summarizes with [sshleifer/distilbart-cnn-12-6](https://huggingface.co/sshleifer/distilbart-cnn-12-6).

## Features

- **RSS Feed Integration**: The generator seamlessly retrieves content from RSS feeds, ensuring up-to-date newsletter content.
- **Customizable Output**: Users can specify the title and topic of the newsletter to tailor content according to their preferences.
- **Transformer-based Text Generation**: Utilizing the powerful capabilities of the Transformers library, the generator produces engaging and diverse newsletter content.

## Dependencies

The Newsletter Generator relies on the following dependencies:

- **argparse**: For parsing command-line arguments.
- **beautifulsoup4**: Used to strip HTML content from RSS feed items.
- **feedparser**: Utilized for parsing RSS feed content.
- **lxml**: HTML/XML parsing library for BeautifulSoup.
- **requests**: Handles HTTP requests to fetch RSS feed content.
- **rich**: Used to display progress during command-line operations.
- **transformers**: Employed for tokenization and model-based text generation.
- **tensorflow**: For machine learning operations (e.g., model loading and inference).

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
python newsletter_generator.py --feed <feed_url> --title <newsletter_title> [--topic <newsletter_topic>]
```

Replace <feed_url> with the URL of the RSS feed you want to generate the newsletter from. Specify <newsletter_title> as the desired title for the newsletter. Optionally, you can include <newsletter_topic> to define a specific topic for the newsletter.

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes or improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
