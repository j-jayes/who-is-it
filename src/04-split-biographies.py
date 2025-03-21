import os
import re
import yaml
import logging
import string

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def extract_year(book_name):
    """Extracts the year from the book name."""
    match = re.search(r'\d+$', book_name)
    if match:
        return int(match.group())
    else:
        raise ValueError(f"Could not extract year from book name: {book_name}")


def read_and_split_biographies(file_name, letter, year):
    """
    Reads a text file containing biographies and splits them based on the provided letter.
    
    Args:
    file_name (str): The path of the file to read.
    letter (str): The letter used to split the biographies.
    year (int): The year of the book, used to determine the case of the surnames.
    
    Returns:
    list: A list containing the split biographies.
    """
    try:
        with open(file_name, "r", encoding="utf-8") as file:
            text = file.read()
        
        # Replace multiple spaces with a single space
        text = re.sub(' +', ' ', text)
        
        pattern = fr'^[\s{re.escape(string.punctuation)}\d]*({letter}[a-zåäö{re.escape(string.punctuation)}]+),'

        # Special handling for "V" and "W"
        if letter == 'V':
            pattern = fr'^[\s{re.escape(string.punctuation)}\d]*([VW][a-zåäö{re.escape(string.punctuation)}]+),'

        # Split text based on the updated pattern
        split_text = re.split(pattern, text, flags=re.MULTILINE)
        
        # Correctly zip the captured surnames and biographies
        biographies = [f"{surname}, {bio}" for surname, bio in zip(split_text[1::2], split_text[2::2])]
        
        return biographies
    except Exception as e:
        logging.error(f"Error occurred while reading and splitting biographies from {file_name}: {e}")
        return []


def clean_biography(biography):
    """Cleans a biography by removing line breaks and trimming whitespace."""
    return ' '.join(biography.split()).strip()

def save_biography(biography, file_path):
    """Saves a cleaned biography to a specified file path."""
    try:
        with open(file_path, 'w', encoding='utf-8') as output_file:
            output_file.write(biography)
    except Exception as e:
        logging.error(f"Error occurred while saving biography to {file_path}: {e}")


def process_books():
    books = ["1918", "1925", "1933", "1939", "1943", "1945", "1953", "1955", "1957", "1963", "1967", "1969", "1977", "1981", "1985", "1993", "1995", "1997", "2001"]
    
    for book_name in books:
        try:
            year = extract_year(book_name)
        except ValueError as e:
            logging.error(e)
            continue
        
        input_directory = os.path.join('data', 'joined', book_name)
        output_directory = os.path.join('data', 'biographies', book_name)
        os.makedirs(output_directory, exist_ok=True)
        
        for letter in string.ascii_uppercase:
            input_file = os.path.join(input_directory, f'{book_name}_{letter}-names.txt')
            if os.path.exists(input_file):
                biographies = read_and_split_biographies(input_file, letter, year)
                for i, bio in enumerate(biographies):
                    cleaned_bio = clean_biography(bio)
                    output_file = os.path.join(output_directory, f'{book_name}_{letter}_{i+1}.txt')
                    save_biography(cleaned_bio, output_file)
            else:
                logging.info(f"File not found: {input_file}")


if __name__ == '__main__':
    process_books()