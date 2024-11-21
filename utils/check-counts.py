import os
import re
from collections import defaultdict

def count_files_per_year(directory):
    file_counts = defaultdict(int)
    pattern = re.compile(r'^(\d{4})')

    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            year = match.group(1)
            file_counts[year] += 1

    return file_counts

def main():
    directory = 'data/raw'
    file_counts = count_files_per_year(directory)

    print("Year\tCount")
    for year, count in sorted(file_counts.items()):
        print(f"{year}\t{count}")

if __name__ == "__main__":
    main()