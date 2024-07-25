# `BookScraper` Class Documentation

## Overview

The `BookScraper` class is designed to scrape biographical data from a specified book hosted on the `runeberg.org` website. It fetches pages, extracts relevant information, and saves the text data to local files. The class supports resuming from a specified page and handles retries for network requests.

## Class Definition

```python
class BookScraper:
    def __init__(self, resume_book_id, resume_page=0):
        # Initialization code
```

### Initialization Parameters

- `resume_book_id` (str): The ID of the book from which to resume scraping.
- `resume_page` (int, optional): The page number from which to resume scraping. Defaults to 0.

## Methods

### `extract_bio_data(page_source)`

Extracts the biographical data from the HTML source of a page.

**Parameters:**

- `page_source` (str): The HTML content of the page.

**Returns:**

- `bio_data` (str or None): The extracted biographical data as a string, or `None` if the data could not be found.

### `fetch_page(url)`

Fetches the content of a web page, with retries on failure.

**Parameters:**

- `url` (str): The URL of the page to fetch.

**Returns:**

- `content` (bytes or None): The content of the page if successful, or `None` if all retries fail.

### `save_text(book_id, page_num, text)`

Saves the extracted text to a file.

**Parameters:**

- `book_id` (str): The ID of the book.
- `page_num` (int): The page number.
- `text` (str): The text content to save.

### `fetch_book_text(book_id, num_pages, start_page=1, end_page=952)`

Fetches and saves the text for the specified range of pages in a book.

**Parameters:**

- `book_id` (str): The ID of the book.
- `num_pages` (int): The total number of pages in the book.
- `start_page` (int, optional): The starting page number. Defaults to 1.
- `end_page` (int, optional): The ending page number. Defaults to 952.

### `run()`

Main method to run the scraping process. Reads the number of pages per book from a configuration file and resumes scraping from the specified book and page.

## Usage

### Running the Script

The script can be run from the command line and accepts two arguments: `resume_book_id` and `resume_page`.

```sh
python scraper.py <resume_book_id> --resume_page <resume_page>
```

### Example

To scrape the book with ID `2001` starting from page `0`:

```sh
python scraper.py 2001 --resume_page 0
```

### Integration with GitHub Actions

This class is integrated with GitHub Actions to automate the scraping process. The GitHub Actions workflow allows specifying the `resume_book_id` and `resume_page` as inputs, enabling the script to run automatically based on user input.

#### GitHub Actions Workflow Example

```yaml
name: Scrape Book Pages

on:
  workflow_dispatch:
    inputs:
      resume_book_id:
        description: 'Book ID to scrape'
        required: true
        default: '2001'
      resume_page:
        description: 'Page number to resume scraping from'
        required: false
        default: '0'

jobs:
  scrape:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout repository
      uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.x'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install requests beautifulsoup4

    - name: Run scraper
      run: |
        python scraper.py "${{ github.event.inputs.resume_book_id }}" --resume_page "${{ github.event.inputs.resume_page }}"
```

## Conclusion

The `BookScraper` class provides a robust solution for scraping biographical data from a specified book. It handles network retries, supports resuming from a specific page, and integrates seamlessly with GitHub Actions for automated workflows.
