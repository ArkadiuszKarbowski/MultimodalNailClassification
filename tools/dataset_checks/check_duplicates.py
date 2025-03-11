import os
import argparse
from collections import defaultdict


def get_patient_ids(dataset_dir):
    patient_ids = defaultdict(set)
    for class_name in os.listdir(dataset_dir):
            class_dir = os.path.join(dataset_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for patient_id in os.listdir(class_dir):
                patient_id.split()
                patient_ids[patient_id.split()[0]].add(class_name)
    return patient_ids


def check_duplicates(dataset_dir):
    patient_ids = get_patient_ids(dataset_dir)
    duplicate_patients = []
    for patient_id, classes in patient_ids.items():
        if len(classes) > 1:
            duplicate_patients.append(patient_id)
    if duplicate_patients:
        print("Duplicate patients found:")
        for patient_id in duplicate_patients:
            print(patient_id)
    print("✓ No duplicate patients found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check for duplicate patients across classes_dir categories."
    )
    parser.add_argument("--dataset_dir", type=str, help="Path to the dataset folder", default="datasets/dataset")
    args = parser.parse_args()

    check_duplicates(args.dataset_dir)