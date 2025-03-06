import os
import re
import argparse

file_pattern = re.compile(r"^(\d+)\s+(SL|SP|RL|RP)\s+([1-5])\s+(H|P)\s*(UV)?\.jpg$")

def validate_filenames(folder_path):
    invalid_files = []
    
    for dirpath, _, filenames in os.walk(folder_path):
        for filename in filenames:
            if not file_pattern.match(filename):
                invalid_files.append(os.path.join(dirpath, filename))
    
    if invalid_files:
        print("Invalid file names:")
        for file in invalid_files:
            print(file)
    else:
        print("All file names are valid.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate file names in a folder.")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing files")
    args = parser.parse_args()
    
    validate_filenames(args.folder_path)
