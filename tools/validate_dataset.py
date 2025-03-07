import argparse

from dataset_tools.validate_filenames import validate_filenames
from dataset_tools.check_duplicates import check_duplicates


def validate_dataset(root_dir):
    print("")
    validate_filenames(root_dir)
    check_duplicates(root_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all dataset validation checks.")
    parser.add_argument(
        "root_dir", type=str, help="Path to the root folder of the dataset"
    )
    args = parser.parse_args()

    validate_dataset(args.root_dir)
