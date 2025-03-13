import os
import numpy as np
import pandas as pd
from collections import defaultdict


def get_nail_id(filename):
    parts = filename.split()
    # PatientID + LimbCode + Digit uniquely identifies a nail
    return f"{parts[0]}_{parts[1]}_{parts[2]}"


def get_image_type(filename):
    filename_no_ext = os.path.splitext(filename)[0]
    parts = filename_no_ext.split()
    position = parts[3]
    is_uv = False
    if len(parts) > 4 and "UV" in parts[4]:
        is_uv = True
    return position, is_uv


def split_dataset(
    dataset_dir,
    val_ratio=0.1,
    test_ratio=0.1,
    modality=1,
    random_state=2137,
    verbose=True,
):
    """
    Splits the dataset into train, validation, and test sets at the nail level with stratified class distribution.

    Parameters
    ----------
    dataset_dir : str
        Path to the dataset root directory containing class subdirectories
    val_ratio : float, optional
        Proportion of nails to use for validation (0.0-1.0). Default: 0.1
    test_ratio : float, optional
        Proportion of nails to use for testing (0.0-1.0). Default: 0.1
    modality : {1, 2, 4}, optional
        Image grouping strategy:
        - 1: Individual images (default)
        - 2: Paired UV/non-UV images from the same nail
        - 4: All combinations (H/P positions x UV/non-UV)
    random_state : int, optional
        Random seed for reproducible splits. Default: 2137
    verbose : bool, optional
        Enable split statistics output. Default: True

    Returns
    -------
    tuple[list[dict], list[dict], list[dict]]
        Train, validation, and test splits containing dictionaries with:
        - modality=1: {'file_path': str}
        - modality=2: {'normal': str, 'uv': str}
        - modality=4: {'H_normal': str, 'H_uv': str, 'P_normal': str, 'P_uv': str}
        Paths are absolute paths to image files.

    Notes
    -----
    - Performs nail-level splitting to prevent data leakage
    - Maintains original class distribution through stratification
    - Total split proportions: train (1-val_ratio-test_ratio), val (val_ratio), test (test_ratio)
    - H/P positions refer to different nail positions (horizontal/vertical)
    """

    all_files = []
    all_nail_ids = []
    all_classes = []
    all_positions = []
    all_is_uv = []

    for class_name in os.listdir(dataset_dir):
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        for patient_id in os.listdir(class_dir):
            patient_dir = os.path.join(class_dir, patient_id)
            if not os.path.isdir(patient_dir):
                continue

            for img_file in os.listdir(patient_dir):
                if img_file.endswith(".jpg"):
                    file_path = os.path.join(patient_dir, img_file)
                    nail_id = get_nail_id(img_file)
                    position, is_uv = get_image_type(img_file)

                    all_files.append(file_path)
                    all_nail_ids.append(nail_id)
                    all_classes.append(class_name)
                    all_positions.append(position)
                    all_is_uv.append(is_uv)

    df = pd.DataFrame(
        {
            "file_path": all_files,
            "nail_id": all_nail_ids,
            "class": all_classes,
            "position": all_positions,
            "is_uv": all_is_uv,
        }
    )

    # Create a nail-level DataFrame (one row per nail)
    nail_df = df.drop_duplicates(subset=["nail_id"])[["nail_id", "class"]]

    train_nail_ids = []
    val_nail_ids = []
    test_nail_ids = []

    # Stratify by class
    for class_name in nail_df["class"].unique():
        # Get all nail IDs for this class
        class_nail_ids = nail_df[nail_df["class"] == class_name]["nail_id"].values
        np.random.seed(random_state)
        np.random.shuffle(class_nail_ids)

        n_total = len(class_nail_ids)
        n_test = int(n_total * test_ratio)
        n_val = int(n_total * val_ratio)
        n_train = n_total - n_test - n_val

        train_nail_ids.extend(class_nail_ids[:n_train])
        val_nail_ids.extend(class_nail_ids[n_train : n_train + n_val])
        test_nail_ids.extend(class_nail_ids[n_train + n_val :])

    # Group files by nail_id
    train = create_modality_groups(df[df["nail_id"].isin(train_nail_ids)], modality)
    val = create_modality_groups(df[df["nail_id"].isin(val_nail_ids)], modality)
    test = create_modality_groups(df[df["nail_id"].isin(test_nail_ids)], modality)

    if verbose:
        print(f"{'=' * 50}")
        print(f"DATASET SPLIT SUMMARY")
        print(f"{'=' * 50}")
        print(f"Train samples: {len(train)}")
        print(f"Validation samples: {len(val)}")
        print(f"Test samples: {len(test)}")
        print(f"Total samples: {len(train) + len(val) + len(test)}")

        # Calculate class distributions with percentages
        print(f"\n{'=' * 50}")
        print(f"CLASS DISTRIBUTION")
        print(f"{'=' * 50}")

        # Train distribution
        train_dist = df[df["nail_id"].isin(train_nail_ids)]["class"].value_counts()
        train_total = train_dist.sum()
        print("\nTRAIN SET:")
        for cls, count in train_dist.items():
            percentage = (count / train_total) * 100
            print(f"  {cls}: {count} samples ({percentage:.1f}%)")

        # Validation distribution
        val_dist = df[df["nail_id"].isin(val_nail_ids)]["class"].value_counts()
        val_total = val_dist.sum()
        print("\nVALIDATION SET:")
        for cls, count in val_dist.items():
            percentage = (count / val_total) * 100
            print(f"  {cls}: {count} samples ({percentage:.1f}%)")

        # Test distribution
        test_dist = df[df["nail_id"].isin(test_nail_ids)]["class"].value_counts()
        test_total = test_dist.sum()
        print("\nTEST SET:")
        for cls, count in test_dist.items():
            percentage = (count / test_total) * 100
            print(f"  {cls}: {count} samples ({percentage:.1f}%)")

        print(f"\n{'=' * 50}")

    return train, val, test


def create_modality_groups(df, modality):
    if modality == 1:
        # Return each file as a dict with file_path key
        return [{"file_path": path} for path in df["file_path"].tolist()]

    if modality == 2:
        # Group by nail_id and position (UV pairs)
        grouped = defaultdict(dict)

        for _, row in df.iterrows():
            nail_id = row["nail_id"]
            position = row["position"]
            is_uv = row["is_uv"]

            key = f"{nail_id}_{position}"
            if is_uv:
                grouped[key]["uv"] = row["file_path"]
            else:
                grouped[key]["normal"] = row["file_path"]

        # Convert to list of dicts, ensure all have both keys (even if None)
        result = []
        for key, value in grouped.items():
            complete_dict = {
                "normal": value.get("normal", None),
                "uv": value.get("uv", None),
            }
            result.append(complete_dict)
        return result

    elif modality == 4:
        # Group by nail_id (all 4 combinations)
        grouped = defaultdict(dict)
        possible_keys = ["H_normal", "H_uv", "P_normal", "P_uv"]

        for _, row in df.iterrows():
            nail_id = row["nail_id"]
            position = row["position"]
            is_uv = row["is_uv"]

            key = f"{position}_{'uv' if is_uv else 'normal'}"
            grouped[nail_id][key] = row["file_path"]

        # Convert to list of dicts, ensure all have all 4 keys (even if None)
        result = []
        for nail_id, paths in grouped.items():
            complete_dict = {key: paths.get(key, None) for key in possible_keys}
            result.append(complete_dict)
        return result


if __name__ == "__main__":
    train_files, val_files, test_files = split_dataset(
        "datasets/dataset", verbose=True, modality=4
    )
    print("\nExample samples:")
    for file in test_files[:5]:
        print(file)

    print("\n")
