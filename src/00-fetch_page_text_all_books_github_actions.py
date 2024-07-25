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
    def __init__(self, resume_book_id, resume_page=0):
        self.resume_book_id = resume_book_id
        self.resume_page = resume_page

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

    def fetch_book_text(self, book_id, num_pages, start_page=1, end_page=3):
        for i in range(start_page, min(num_pages + 1, end_page + 1)):
            file_path = DATA_DIR / f"{book_id}_page_text_{i}.txt"
            if file_path.exists():
                logging.info(f"Page text {file_path} already exists. Skipping...")
                continue

            url = BASE_URL.format(book_id, i)
            content = self.fetch_page(url)
            if content:
                soup = BeautifulSoup(content, "html.parser")
                bio_data = self.extract_bio_data(str(soup))
                if bio_data:
                    self.save_text(book_id, i, bio_data)
                else:
                    logging.warning(f"Biographical data not found for page {i}.")
                time.sleep(SLEEP_DURATION)
            else:
                logging.error(f"Failed to fetch page {i} after {MAX_RETRIES} attempts.")

    def run(self):
        with PAGES_CONFIG.open("r") as infile:
            number_of_pages_per_book = json.load(infile)

        found_resume_book = False
        for book_id, num_pages in number_of_pages_per_book.items():
            if not found_resume_book:
                if book_id == self.resume_book_id:
                    found_resume_book = True
                    logging.info(f"Resuming text fetching for {book_id} from page {self.resume_page}...")
                    self.fetch_book_text(book_id, num_pages, self.resume_page)
                else:
                    logging.info(f"Skipping {book_id}...")
            else:
                logging.info(f"Fetching text for {book_id}...")
                self.fetch_book_text(book_id, num_pages)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Scrape book pages.')
    parser.add_argument('resume_book_id', type=str, help='The book ID to resume scraping from')
    parser.add_argument('--resume_page', type=int, default=0, help='The page number to resume scraping from')
    args = parser.parse_args()

    scraper = BookScraper(resume_book_id=args.resume_book_id, resume_page=args.resume_page)
    scraper.run()
