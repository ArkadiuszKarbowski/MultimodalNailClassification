import os
import argparse
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.get_metadata import get_img_metadata


def is_img(filename):
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
    _, ext = os.path.splitext(filename.lower())
    return ext in image_extensions


def check_filenames(dataset_dir):
    invalid_files = []

    for dirpath, _, filenames in os.walk(dataset_dir):
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            if is_img(filename) and not get_img_metadata(full_path):
                invalid_files.append(full_path)

    if invalid_files:
        print("Invalid file names:")
        for file in invalid_files:
            print(file)
    else:
        print("✓ All image file names are valid.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate file names in a folder.")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        help="Path to the folder containing files",
        default="datasets/dataset",
    )
    args = parser.parse_args()

    check_filenames(args.dataset_dir)
