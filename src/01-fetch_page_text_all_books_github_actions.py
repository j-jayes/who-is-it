import argparse
import requests
import time
from bs4 import BeautifulSoup
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
START_TAG = "<!-- mode=normal -->"
END_TAG = "<!-- NEWIMAGE2 -->"
BASE_URL = "http://runeberg.org/vemardet/{}/{:04d}.html"
DATA_DIR = Path("data/raw")
PAGES_CONFIG = Path("data/number_of_pages_per_book.json")
SLEEP_DURATION = 5
MAX_RETRIES = 3

class BookScraper:
    def __init__(self, book_id, start_page=1):
        self.book_id = book_id
        self.start_page = start_page

    def extract_bio_data(self, page_source):
        start_index = page_source.find(START_TAG)
        end_index = page_source.find(END_TAG)

        if (start_index == -1) or (end_index == -1):
            return None

        start_index += len(START_TAG)
        bio_data = page_source[start_index:end_index].strip()
        return bio_data

    def fetch_page(self, url):
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.get(url)
                response.raise_for_status()
                return response.content
            except requests.RequestException as e:
                logging.error(f"Attempt {attempt + 1} failed for URL {url}: {e}")
                time.sleep(SLEEP_DURATION)
        return None

    def save_text(self, book_id, page_num, text):
        file_path = DATA_DIR / f"{book_id}_page_text_{page_num}.txt"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as file:
            file.write(text)
        logging.info(f"Text saved to {file_path}")

    def fetch_book_text(self, num_pages):
        for i in range(self.start_page, num_pages + 1):
            file_path = DATA_DIR / f"{self.book_id}_page_text_{i}.txt"
            if file_path.exists():
                logging.info(f"Page text {file_path} already exists. Skipping...")
                continue

            url = BASE_URL.format(self.book_id, i)
            content = self.fetch_page(url)
            if content:
                soup = BeautifulSoup(content, "html.parser")
                bio_data = self.extract_bio_data(str(soup))
                if bio_data:
                    self.save_text(self.book_id, i, bio_data)
                else:
                    logging.warning(f"Biographical data not found for page {i}.")
                time.sleep(SLEEP_DURATION)
            else:
                logging.error(f"Failed to fetch page {i} after {MAX_RETRIES} attempts.")

    def run(self):
        with PAGES_CONFIG.open("r") as infile:
            number_of_pages_per_book = json.load(infile)

        if self.book_id in number_of_pages_per_book:
            num_pages = number_of_pages_per_book[self.book_id]
            logging.info(f"Fetching text for {self.book_id} starting from page {self.start_page}...")
            self.fetch_book_text(num_pages)
        else:
            logging.error(f"Book ID {self.book_id} not found in {PAGES_CONFIG}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Scrape a single book.')
    parser.add_argument('book_id', type=str, help='The book ID to scrape')
    parser.add_argument('--start_page', type=int, default=1, help='The page number to start scraping from')
    args = parser.parse_args()

    scraper = BookScraper(book_id=args.book_id, start_page=args.start_page)
    scraper.run()
