import os
import yaml

def join_text_files(book_name, start_page, end_page, letter, output_directory):
    joined_text = ''

    for page_number in range(start_page, end_page):
        file_name = f'{book_name}_page_text_{page_number}.txt'
        file_path = os.path.join('data', 'raw', file_name)

        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                joined_text += file.read()
        else:
            print(f"File not found: {file_path}")

    output_file_name = f'{book_name}_{letter}-names.txt'
    output_file_path = os.path.join(output_directory, output_file_name)

    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        output_file.write(joined_text)

def main():
    books = ["1918", "1925", "1933", "1939", "1943", "1945", "1953", "1955", "1957", "1963", "1967", "1969", "1977", "1981", "1985", "1993", "1995", "1997", "2001"]
    book_end_pages = {
        "1918": 444,
        "1925": 867,
        "1933": 964,
        "1939": 951,
        "1943": 948,
        "1945": 1244,
        "1953": 1210,
        "1955": 1093,
        "1957": 1135,
        "1963": 1245,
        "1967": 1065,
        "1969": 1104,
        "1977": 1184,
        "1981": 1213,
        "1985": 1254,
        "1993": 1250,
        "1995": 1252,
        "1997": 1266,
        "2001": 1279
    }

    for book_name in books:
        toc_path = os.path.join('data', f'{book_name}_toc.yaml')
        if not os.path.exists(toc_path):
            print(f"TOC not found for book: {book_name}. Skipping...")
            continue

        with open(toc_path, 'r', encoding='utf-8') as yaml_file:
            toc = yaml.safe_load(yaml_file)

        output_directory = os.path.join('data', 'joined', book_name)
        os.makedirs(output_directory, exist_ok=True)

        page_numbers = list(toc.values())
        letters = list(toc.keys())

        for i in range(len(letters)):
            start_page = page_numbers[i]
            end_page = page_numbers[i + 1] if i + 1 < len(page_numbers) else book_end_pages[book_name]
            letter = letters[i]

            join_text_files(book_name, start_page, end_page, letter, output_directory)

if __name__ == '__main__':
    main()
