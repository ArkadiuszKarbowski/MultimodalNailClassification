import os
import json
import torch
import torchvision.transforms as T
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class ImageDataset(Dataset):
    def __init__(self, image_paths, resize_shape=(512, 512), is_uv=False):
        self.image_paths = image_paths
        self.resize_shape = resize_shape
        self.is_uv = is_uv
        self.transform = T.Compose(
            [
                T.Resize(resize_shape),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img = read_image(img_path)
            if img is None or img.shape[0] == 0:
                return torch.zeros((3, 1, 1)), False

            # Resize the image
            img = self.transform(img)

            # Convert to float32 and normalize
            img = img.float() / 255.0
            return img, True
        except Exception as e:
            # Print error for debugging
            print(f"Error loading {img_path}: {str(e)}")
            return torch.zeros((3, 1, 1)), False


def calculate_dataset_statistics(
    image_paths,
    resize_shape=(512, 512),
    batch_size=48,
    use_gpu=True,
    output_json=None,
    verbose=True,
):
    """
    Calculate dataset statistics after resizing images.

    Parameters:
    -----------
    image_paths : list of str
        List of image paths
    resize_shape : tuple
        Target shape for resizing (height, width)
    batch_size : int
        Batch size for processing
    use_gpu : bool
        Whether to use GPU for processing
    output_json : str
        Path to save statistics as JSON (if None, no saving)

    Returns:
    --------
    dict
        Dictionary containing mean and std for UV and normal images
    """
    device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"

    uv_paths = []
    normal_paths = []
    for path in image_paths:
        if isinstance(path, str) and path is not None:
            if "UV" in os.path.basename(path).upper():
                uv_paths.append(path)
            else:
                normal_paths.append(path)
    if verbose:
        print(f"Using device: {device}")
        print(f"Found {len(uv_paths)} UV images and {len(normal_paths)} normal images")

    # Create datasets and dataloaders
    uv_dataset = ImageDataset(uv_paths, resize_shape=resize_shape, is_uv=True)
    normal_dataset = ImageDataset(normal_paths, resize_shape=resize_shape, is_uv=False)

    # Reduce num_workers to avoid memory issues
    uv_loader = DataLoader(
        uv_dataset,
        batch_size=batch_size,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=True,
    )
    normal_loader = DataLoader(
        normal_dataset,
        batch_size=batch_size,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=True,
    )

    def process_dataloader(loader, desc):
        means = []
        stds = []

        for batch, valid in tqdm(loader, desc=desc):
            # Process in smaller chunks if needed
            if valid.any():
                valid_batch = batch[valid]

                chunk_size = 32
                for i in range(0, len(valid_batch), chunk_size):
                    chunk = valid_batch[i : i + chunk_size].to(device)

                    # Calculate statistics
                    with torch.no_grad():
                        chunk_mean = torch.mean(chunk, dim=[2, 3])
                        chunk_std = torch.std(chunk, dim=[2, 3])

                    # Move results back to CPU immediately
                    means.append(chunk_mean.cpu())
                    stds.append(chunk_std.cpu())

                    # Clear GPU memory
                    del chunk
                    if device == "cuda":
                        torch.cuda.empty_cache()

        # Combine all statistics
        all_means = torch.cat(means) if means else torch.zeros((0, 3))
        all_stds = torch.cat(stds) if stds else torch.zeros((0, 3))

        # Calculate mean statistics
        mean = (
            torch.mean(all_means, dim=0).numpy()
            if len(all_means) > 0
            else torch.zeros(3).numpy()
        )
        std = (
            torch.mean(all_stds, dim=0).numpy()
            if len(all_stds) > 0
            else torch.ones(3).numpy()
        )

        return mean, std

    mean_uv, std_uv = process_dataloader(uv_loader, "Processing UV images")
    mean_normal, std_normal = process_dataloader(
        normal_loader, "Processing normal images"
    )

    stats = {
        "resize_shape": resize_shape,
        "mean_uv": mean_uv.tolist(),
        "std_uv": std_uv.tolist(),
        "mean_normal": mean_normal.tolist(),
        "std_normal": std_normal.tolist(),
        "num_uv_images": len(uv_paths),
        "num_normal_images": len(normal_paths),
    }

    if output_json:
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(stats, f, indent=4)
        print(f"Statistics saved to {output_json}")

    return stats


def get_or_calc_stats(
    image_paths,
    stats_json_path,
    resize_shape=(512, 512),
    verbose=True,
    force_recalculate=False,
):
    """Get statistics from file or calculate them if needed"""
    if os.path.exists(stats_json_path) and not force_recalculate:
        print(f"Loading existing statistics from {stats_json_path}")
        with open(stats_json_path, "r") as f:
            return json.load(f)
    else:
        print("Calculating new dataset statistics")
        return calculate_dataset_statistics(
            image_paths,
            resize_shape=resize_shape,
            output_json=stats_json_path,
            verbose=verbose,
        )
