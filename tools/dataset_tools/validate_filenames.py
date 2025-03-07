import os
import re
import argparse

file_pattern = re.compile(r"^(\d+)\s+(SL|SP|RL|RP)\s+([1-5])\s+(H|P)\s*(UV)?\.jpg$")

"""
File pattern: [PatientID] [LimbCode] [Digit] [Position][UV].jpg
   Components (space-separated):
   - PatientID: digits
   - LimbCode: SL/SP/RL/RP -> Left foot / Right foot / Left hand / Right hand
   - Digit: 1-5 -> Toe/Finger number
   - Position: H/P -> Horizontal/Parallel orientation
   - Optional: UV -> UV image
   Example: "123 SL 2 H UV.jpg"
"""


def is_img(filename):
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
    _, ext = os.path.splitext(filename.lower())
    return ext in image_extensions


def validate_filenames(folder_path):
    invalid_files = []

    for dirpath, _, filenames in os.walk(folder_path):
        for filename in filenames:
            if is_img(filename) and not file_pattern.match(filename):
                invalid_files.append(os.path.join(dirpath, filename))

    if invalid_files:
        print("Invalid file names:")
        for file in invalid_files:
            print(file)
    else:
        print("✓ All image file names are valid.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate file names in a folder.")
    parser.add_argument(
        "folder_path", type=str, help="Path to the folder containing files"
    )
    args = parser.parse_args()

    validate_filenames(args.folder_path)
