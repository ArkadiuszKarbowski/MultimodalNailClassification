import argparse

from tools.dataset_checks.check_filenames import check_filenames
from dataset_checks.check_duplicates import check_duplicates


def validate_dataset(dataset_dir):
    print("")
    check_filenames(dataset_dir)
    check_duplicates(dataset_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all dataset validation checks.")
    parser.add_argument(
        "dataset_dir", type=str, help="Path to the root folder of the dataset"
    )
    args = parser.parse_args()

    validate_dataset(args.dataset_dir)
