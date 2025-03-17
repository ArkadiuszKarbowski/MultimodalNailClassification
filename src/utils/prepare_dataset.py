import os
from .prepare_dataset_utils.split_dataset import split_dataset
from .prepare_dataset_utils.dataset_stats import get_or_calc_stats


def prepare_dataset(
    dataset_dir,
    modality=1,
    resize_shape=(512, 512),
    verbose=True,
    force_recalculate_stats=False,
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
    """
    # Create stats directory if it doesn't exist
    stats_dir = os.path.join(dataset_dir, ".stats")
    os.makedirs(stats_dir, exist_ok=True)

    # Define stats JSON path
    stats_json_path = os.path.join(
        stats_dir,
        f"modality{modality}_stats_resized_{resize_shape[0]}x{resize_shape[1]}.json",
    )

    # Split dataset
    train_paths, val_paths, test_paths = split_dataset(
        dataset_dir, verbose=verbose, modality=modality
    )

    # Get all image paths for training + validation
    unpacked_paths = []
    for item in train_paths + val_paths:
        for key, path in item.items():
            if isinstance(path, str) and path is not None:
                unpacked_paths.append(path)

    # Get or calculate dataset statistics
    dataset_stats = get_or_calc_stats(
        unpacked_paths,
        stats_json_path,
        resize_shape=resize_shape,
        force_recalculate=force_recalculate_stats,
    )

    return train_paths, val_paths, test_paths, dataset_stats


if __name__ == "__main__":
    train_dataset, val_dataset, test_dataset, stats = prepare_dataset(
        "datasets/dataset",
        modality=2,
        resize_shape=(512, 512),
        force_recalculate_stats=False,
    )

    print(f"Dataset statistics: {stats}")
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")

    print(f"Training set example: {train_dataset[0:5]}")
