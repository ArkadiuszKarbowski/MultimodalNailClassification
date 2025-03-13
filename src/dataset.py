import cv2
from torch.utils.data import Dataset
from utils.get_metadata import get_metadata_from_image_path


class NailDataset(Dataset):
    def __init__(self, dataset_items, transform=None):
        """
        Initialize the NailDataset.

        Parameters:
        -----------
        dataset_items : list of dict
            - modality=1: [{'file_path': str}, ...]
            - modality=2: [{'normal': str, 'uv': str}, ...]
            - modality=4: [{'H_normal': str, 'H_uv': str, 'P_normal': str, 'P_uv': str}, ...]
        transform : callable, optional
            Optional transform to be applied to images
        """
        self.dataset_items = dataset_items
        self.transform = transform

        # Determine modality based on keys in the first non-None item
        self.modality = self._determine_modality()

        # Process items to include metadata
        self.processed_items = []
        for item in self.dataset_items:
            processed_item = {}

            # Process each path in the item according to modality
            for key, path in item.items():
                if path is not None:
                    metadata = get_metadata_from_image_path(path)
                    metadata["image_path"] = path
                    processed_item[key] = metadata
                else:
                    processed_item[key] = None

            self.processed_items.append(processed_item)

    def _determine_modality(self):
        """Determine modality based on first non-None item"""
        for item in self.dataset_items:
            keys = list(item.keys())
            if "file_path" in keys:
                return 1
            elif set(keys) == {"normal", "uv"}:
                return 2
            elif set(keys) == {"H_normal", "H_uv", "P_normal", "P_uv"}:
                return 4

        # Default fallback
        return 1

    def __len__(self):
        return len(self.dataset_items)

    def __getitem__(self, idx):
        item = self.processed_items[idx]
        result = {}

        for key, metadata in item.items():
            if metadata is not None:
                image = self._load_and_transform_image(metadata["image_path"])
                result[key] = {
                    "image": image,
                }
            else:
                result[key] = None

        return result

    def _load_and_transform_image(self, image_path):
        """Load and transform an image from the given path"""
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)["image"]

        return image


if __name__ == "__main__":
    from utils.split_dataset import split_dataset

    # Test with modality 1
    train_files, val_files, test_files = split_dataset(
        "datasets/dataset", verbose=False, modality=1
    )
    dataset = NailDataset(train_files[:5])
    print(f"Modality: {dataset.modality}")
    print(f"Dataset length: {len(dataset)}")
    print(f"Sample item keys: {dataset[0].keys()}")

    # Test with modality 2
    train_files, val_files, test_files = split_dataset(
        "datasets/dataset", verbose=False, modality=2
    )
    dataset = NailDataset(train_files[:5])
    print(f"\nModality: {dataset.modality}")
    print(f"Dataset length: {len(dataset)}")
    print(f"Sample item keys: {dataset[0].keys()}")
