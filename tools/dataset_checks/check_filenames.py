import os
from utils import get_metadata
import argparse


def is_img(filename):
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
    _, ext = os.path.splitext(filename.lower())
    return ext in image_extensions


def check_filenames(dataset_dir):
    invalid_files = []

    for dirpath, _, filenames in os.walk(dataset_dir):
        for filename in filenames:
            if is_img(filename) and not get_metadata(filename):
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
        "--dataset_dir",
        type=str,
        help="Path to the folder containing files",
        default="datasets/dataset",
    )
    args = parser.parse_args()

    check_filenames(args.dataset_dir)
