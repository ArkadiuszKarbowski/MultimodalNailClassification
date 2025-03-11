import cv2
from pathlib import Path
from torch.utils.data import Dataset

class NailDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def parse_filename(self, filename):
        parts = filename.split()
        if len(parts) < 4:
            raise ValueError(f"Filename '{filename}' does not match expected format.")
        info = {
            "patient_id": int(parts[0]),
            "limb": parts[1],
            "digit": int(parts[2]),
            "position": parts[3].split(".")[0],
            "is_uv": "UV" in filename,
        }
        limb_mapping = {
            "SL": "left_foot",
            "SP": "right_foot",
            "RL": "left_hand",
            "RP": "right_hand",
        }
        info["limb_full"] = limb_mapping.get(info["limb"], info["limb"])
        return info

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Unable to read image at path: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)["image"]
        class_name = Path(img_path).parent.parent.name
        filename = Path(img_path).name
        file_info = self.parse_filename(filename)
        return {
            "image": image,
            "class_name": class_name,
            "patient_id": file_info["patient_id"],
            "limb": file_info["limb"],
            "limb_full": file_info["limb_full"],
            "digit": file_info["digit"],
            "position": file_info["position"],
            "is_uv": file_info["is_uv"],
        }