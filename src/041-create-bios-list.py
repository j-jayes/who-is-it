import os
import json

def get_files_in_directory(directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            relative_path = os.path.relpath(os.path.join(root, file), directory)
            # add "data/biographies" to the beginning of the relative path
            relative_path = os.path.join("data/biographies", relative_path)
            file_list.append({"filename": file, "relative_path": relative_path})
    return file_list

def main():
    directory = 'data/biographies'
    file_list = get_files_in_directory(directory)
    # arrange the list of dictionaries by file name
    file_list = sorted(file_list, key=lambda x: x['filename'])

    # create the output directory if it doesn't exist
    output_directory = 'data/biography_list'
    os.makedirs(output_directory, exist_ok=True)
    
    with open('data/biography_list/biography_list.json', 'w') as json_file:
        json.dump(file_list, json_file, indent=4)

if __name__ == "__main__":
    main()