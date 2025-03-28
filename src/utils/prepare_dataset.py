import os
from src.utils.prepare_dataset_utils.split_dataset import split_dataset
from src.utils.prepare_dataset_utils.dataset_stats import get_or_calc_stats
from src.utils.get_metadata import get_img_metadata


def prepare_dataset(
    dataset_dir,
    modality=1,
    resize_shape=(512, 512),
    verbose=True,
    force_recalculate_stats=False,
    filter_incomplete=True,
    filter_perspective=False,
    seed=2137,
):
    """
    Prepare dataset with proper statistics calculation

    Parameters:
    -----------
    dataset_dir : str
        Path to dataset directory
    modality : int
        Dataset modality (1, 2, or 4)
    resize_shape : tuple
        Shape to resize images to before calculating statistics
    force_recalculate_stats : bool
        Whether to force recalculation of statistics
    filter_incomplete : bool
        Whether to filter incomplete items
    """
    # Define modality requirements
    MODALITY_REQUIREMENTS = {
        1: {"file_path"},
        2: {"normal", "uv"},
        4: {"H_normal", "H_uv", "P_normal", "P_uv"},
    }

    # Validate modality
    if modality not in MODALITY_REQUIREMENTS:
        raise ValueError(f"Invalid modality {modality}. Choose 1, 2, or 4.")

    required_keys = MODALITY_REQUIREMENTS[modality]

    if modality == 4 and filter_perspective:
        raise ValueError("Perspective filtering does not apply to modality 4.")

    # Split dataset
    train_paths, val_paths, test_paths = split_dataset(
        dataset_dir, verbose=verbose, modality=modality, seed=seed
    )

    # Filter incomplete items
    def _filter_incomplete(items):
        new_items = []
        for item in items:
            if all(item.get(key) is not None for key in required_keys):
                new_items.append(item)
            elif verbose:
                print(f"Filtered incomplete item: {item}")
        return new_items

    def _filter_perspective(items):
        new_items = []
        for item in items:
            for path in item.values():
                if get_img_metadata(path)["position"] == "H":
                    new_items.append(item)
                    break
        return new_items

    if filter_incomplete:
        train_paths = _filter_incomplete(train_paths)
        val_paths = _filter_incomplete(val_paths)
        test_paths = _filter_incomplete(test_paths)

        if verbose:
            print("Filtered dataset sizes:")
            print(f"Train: {len(train_paths)}")
            print(f"Val: {len(val_paths)}")
            print(f"Test: {len(test_paths)}")

    if filter_perspective:
        train_paths = _filter_perspective(train_paths)
        val_paths = _filter_perspective(val_paths)
        test_paths = _filter_perspective(test_paths)

        if verbose:
            print("Filtered dataset sizes (perspective):")
            print(f"Train: {len(train_paths)}")
            print(f"Val: {len(val_paths)}")
            print(f"Test: {len(test_paths)}")

    # Collect paths for statistics
    unpacked_paths = []
    for item in train_paths + val_paths:
        for key in required_keys:
            path = item.get(key)

            if filter_incomplete:
                unpacked_paths.append(path)
            elif isinstance(path, str) and path is not None:
                unpacked_paths.append(path)

    # Get/create stats
    stats_dir = os.path.join(dataset_dir, ".stats")
    os.makedirs(stats_dir, exist_ok=True)

    stats_json_path = os.path.join(
        stats_dir,
        f"modality{modality}_stats_resized_{resize_shape[0]}x{resize_shape[1]}.json",
    )

    dataset_stats = get_or_calc_stats(
        unpacked_paths,
        stats_json_path,
        resize_shape=resize_shape,
        force_recalculate=force_recalculate_stats,
    )

    return train_paths, val_paths, test_paths, dataset_stats


if __name__ == "__main__":
    train_dataset, val_dataset, test_dataset, stats, train_dist = prepare_dataset(
        "datasets/dataset",
        modality=1,
        resize_shape=(512, 512),
        force_recalculate_stats=False,
    )

    print(f"Dataset statistics: {stats}")
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")

    print(f"Training set example: {train_dataset[0:5]}")

    print(f"Training distribution: {train_dist}")
