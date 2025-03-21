import os

def remove_br_tags(directory_path):
    """
    Removes all <br/> tags from the files in the specified directory.

    Args:
        directory_path (str): The path to the directory containing the files.

    Returns:
        None
    """
    # List all files in the directory
    files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
    
    for file_name in files:
        with open(os.path.join(directory_path, file_name), 'r', encoding='utf-8') as file:
            content = file.read()
            # Replace <br/> tags with empty string
            modified_content = content.replace('<br/>', '')

        # Write the modified content back to the file
        with open(os.path.join(directory_path, file_name), 'w', encoding='utf-8') as file:
            file.write(modified_content)

# Call the function
remove_br_tags("data/raw")


# Now we want to create a dictionary of the names on each page.


def get_char_counts_from_top_lines(directory_path, num_lines=5):
    char_counts_per_file = {}  # Dictionary to store character counts from each file's top lines

    # List all files in the directory
    files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

    for file_name in files:
        with open(os.path.join(directory_path, file_name), 'r', encoding='utf-8') as file:
            lines = [file.readline().strip() for _ in range(num_lines)]  # Read the first num_lines lines
            char_counts = [len(line) for line in lines]  # Count characters for each line
            char_counts_per_file[file_name] = char_counts

    return char_counts_per_file

# Call the function
char_counts = get_char_counts_from_top_lines("data/raw")
# print(char_counts)

# Calculate average for each line across all files
averages = [sum([counts[idx] for counts in char_counts.values()]) / len(char_counts) for idx in range(5)]
print(averages)

def classify_top_lines(directory_path, averages):
    classified_results = {}  # Dictionary to store classification for each file

    # Calculate the threshold based on the average of the 4th and 5th lines
    threshold = 0.7 * ((averages[3] + averages[4]) / 2)

    # List all files in the directory
    files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

    for file_name in files:
        with open(os.path.join(directory_path, file_name), 'r', encoding='utf-8') as file:
            lines = [file.readline().strip() for _ in range(5)]  # Read the first five lines
            char_counts = [len(line) for line in lines[:3]]  # Count characters for the first three lines

            # Check each of the first three lines against the threshold
            line_classifications = ['names' if count < threshold else 'text' for count in char_counts]
            classified_results[file_name] = line_classifications

    return classified_results


# Call the function
classification = classify_top_lines("data/raw", averages)
print(classification)

import re

def extract_names_and_page_numbers(directory_path, classifications):
    name_data = {}  # Dictionary to store names associated with their page numbers

    # Compile a regular expression to extract the page number from the filename
    page_number_regex = re.compile(r"\d{4}_page_text_(\d+)\.txt")

    # List all files in the directory
    files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

    for file_name in files:
        with open(os.path.join(directory_path, file_name), 'r', encoding='utf-8') as file:
            lines = [file.readline().strip() for _ in range(5)]  # Read the first five lines

            # Extract the page number from the file name
            page_number_match = page_number_regex.search(file_name)
            if page_number_match:
                page_number = int(page_number_match.group(1))

                # Check classification and add to the dictionary if it's "names"
                extracted_names = [lines[i] for i, classification in enumerate(classifications[file_name][:3]) if classification == "names"]
                if extracted_names:
                    name_data[page_number] = extracted_names

    return name_data

# Call the function
names_and_page_numbers = extract_names_and_page_numbers("data/raw", classification)
print(names_and_page_numbers)
