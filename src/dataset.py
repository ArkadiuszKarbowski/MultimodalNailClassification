import cv2
from torch.utils.data import Dataset
from src.utils.get_metadata import get_img_metadata
import torch
from sklearn.preprocessing import LabelEncoder


class NailDataset(Dataset):
    def __init__(self, dataset_items, transforms=None):
        """
        Generalized multimodal nail dataset class
        Parameters:
        -----------
        dataset_items : list of dict
            - Each dict contains modality keys and image paths
            - Metadata should be preloaded in get_img_metadata()
        transforms : dict, optional
            Transforms for each image type ('normal'/'uv')
        """
        self.dataset_items = dataset_items
        self.transforms = transforms or {}
        self.label_encoder = LabelEncoder()

        self.processed_items = []
        self._process_items()
        self._encode_labels()

        self.modality_type = self._determine_modality()
        self._setup_modality_strategy()

    def _determine_modality(self):
        """Determine modality type from keys"""
        first_item = next(iter(self.dataset_items), {})
        keys = set(first_item.keys())

        if {"H_normal", "H_uv", "P_normal", "P_uv"}.issubset(keys):
            return 4
        if {"normal", "uv"}.issubset(keys):
            return 2
        if "file_path" in keys:
            return 1
        raise ValueError("Unknown modality structure")

    def _setup_modality_strategy(self):
        """Set up the appropriate item retrieval strategy based on modality"""
        if self.modality_type == 1:
            self._get_item_strategy = self._get_item_mod1
        elif self.modality_type == 2:
            self._get_item_strategy = self._get_item_mod2
        elif self.modality_type == 4:
            self._get_item_strategy = self._get_item_mod4
        else:
            raise ValueError(f"Unsupported modality type: {self.modality_type}")

    def _validate_metadata(self, metadata, path):
        """Validate required fields in metadata"""
        if "class" not in metadata:
            raise ValueError(f"Missing class in metadata for {path}")
        if "is_uv" not in metadata:
            raise ValueError(f"Missing is_uv in metadata for {path}")

    def _verify_label_consistency(self, labels, raw_item):
        """Verify that all images in an item have the same label"""
        if len(labels) != 1:
            raise ValueError(f"Multiple classes in item: {labels}")
        return labels.pop()

    def _process_items(self):
        """Process and validate all dataset items"""
        for raw_item in self.dataset_items:
            processed = {"images": {}, "label": None}
            labels = set()

            # Process each modality in deterministic order
            for key in sorted(raw_item.keys()):
                path = raw_item[key]
                if path is None:
                    continue

                metadata = get_img_metadata(path)
                metadata["image_path"] = path

                self._validate_metadata(metadata, path)

                # Store image data and collect labels
                processed["images"][key] = {
                    "path": path,
                    "is_uv": metadata["is_uv"],
                    "metadata": metadata,
                }
                labels.add(metadata["class"])

            # Verify label consistency
            processed["label"] = self._verify_label_consistency(labels, raw_item)
            self.processed_items.append(processed)

    def _encode_labels(self):
        """Encode all labels using the label encoder"""
        # Initialize label encoder
        all_labels = [item["label"] for item in self.processed_items]
        self.label_encoder.fit(all_labels)

        # Pre-encode all labels
        for item in self.processed_items:
            item["encoded_label"] = torch.tensor(
                self.label_encoder.transform([item["label"]])[0], dtype=torch.long
            )

    def _load_image(self, path):
        """Helper method to load and convert an image"""
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    def _get_item_mod1(self, item):
        """Strategy for modality 1 (single image)"""
        # TODO
        pass

    def _get_item_mod2(self, item):
        """Strategy for modality 2 (normal + UV)"""
        # Load images
        normal_img = self._load_image(item["images"]["normal"]["path"])
        uv_img = self._load_image(item["images"]["uv"]["path"])

        # First apply shared transform to both images (resize & geometric transforms)
        transformed = self.transforms["shared_transform"](image=normal_img, uv=uv_img)
        normal_transformed = transformed["image"]
        uv_transformed = transformed["uv"]

        # Then apply modality-specific transforms (color adjustments & normalization)
        normal_final = self.transforms["normal_transform"](image=normal_transformed)[
            "image"
        ]
        uv_final = self.transforms["uv_transform"](image=uv_transformed)["image"]

        return normal_final, uv_final, item["encoded_label"]

    def _get_item_mod4(self, item):
        """Strategy for modality 4 (H_normal, H_uv, P_normal, P_uv)"""
        # TODO
        pass

    def __len__(self):
        return len(self.processed_items)

    def __getitem__(self, idx):
        """Get an item using the predetermined strategy for current modality"""
        item = self.processed_items[idx]
        return self._get_item_strategy(item)

    def get_metadata(self, idx):
        """Helper to access raw metadata"""
        return self.processed_items[idx]["images"]

    def label_to_class(self, label):
        """Convert label to class"""
        return self.label_encoder.inverse_transform([label])[0]

    def get_class_distribution(self):
        """Get class distribution of the dataset"""
        classes = [item["encoded_label"] for item in self.processed_items]
        return torch.bincount(torch.tensor(classes))


if __name__ == "__main__":
    from utils.prepare_dataset import prepare_dataset
    from augmentation import get_transforms

    modality = 2
    train_files, val_files, test_files, stats = prepare_dataset(
        "datasets/dataset",
        verbose=False,
        modality=modality,
    )

    dataset = NailDataset(
        train_files[:5], transforms=get_transforms(stats, modality, training=True)
    )
    print(f"Modality: {dataset.modality_type}")
    print(f"Dataset length: {len(dataset)}")
    print(f"Dataset item: {dataset[0]}")
    print(f"Images shape: {dataset[0][0].shape}, {dataset[0][1].shape}")
