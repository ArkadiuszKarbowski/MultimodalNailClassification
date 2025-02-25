import os
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json

class NailAugmentation:
    def __init__(self, mean, std, p=0.5):
        self.transform = A.Compose([
            A.OneOf([
                A.RandomBrightnessContrast(p=1),
                A.ColorJitter(p=1),
                A.HueSaturationValue(p=1),
            ], p=0.5),
            A.OneOf([
                A.GaussianBlur(p=1),
                A.MotionBlur(p=1),
                A.MedianBlur(p=1),
            ], p=0.3),
            A.OneOf([
                A.ElasticTransform(p=1),
                A.GridDistortion(p=1),
                A.OpticalDistortion(p=1),
            ], p=0.2),
            A.OneOf([
                A.RandomRotate90(p=1),
                A.HorizontalFlip(p=1),
                A.ShiftScaleRotate(p=1),
            ], p=0.5),
            A.Resize(224, 224),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ], p=p)

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
            'patient_id': int(parts[0]),
            'limb': parts[1],
            'digit': int(parts[2]),
            'position': parts[3].split('.')[0],
            'is_uv': 'UV' in filename
        }
        limb_mapping = {
            'SL': 'left_foot',
            'SP': 'right_foot',
            'RL': 'left_hand',
            'RP': 'right_hand'
        }
        info['limb_full'] = limb_mapping.get(info['limb'], info['limb'])
        return info

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Unable to read image at path: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)['image']
        class_name = Path(img_path).parent.parent.name
        filename = Path(img_path).name
        file_info = self.parse_filename(filename)
        return {
            'image': image,
            'class_name': class_name,
            'patient_id': file_info['patient_id'],
            'limb': file_info['limb'],
            'limb_full': file_info['limb_full'],
            'digit': file_info['digit'],
            'position': file_info['position'],
            'is_uv': file_info['is_uv']
        }

def get_dataset_statistics(root_dir, filename='dataset_stats.json'):
    stats_path = os.path.join(root_dir, filename)
    if not os.path.exists(stats_path):
        stats = get_dataset_stats(root_dir)
        with open(stats_path, 'w') as f:
            json.dump(stats, f)
    else:
        with open(stats_path, 'r') as f:
            stats = json.load(f)
    return (
        stats['mean_uv'],
        stats['std_uv'],
        stats['mean_normal'],
        stats['std_normal']
    )

def prepare_dataset(root_dir, img_size=224, batch_size=32):
    mean_uv, std_uv, mean_normal, std_normal = get_dataset_statistics(root_dir)

    transform_uv = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=mean_uv, std=std_uv),
        ToTensorV2()
    ])

    transform_normal = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=mean_normal, std=std_normal),
        ToTensorV2()
    ])

    all_image_paths = []
    for main_category in os.listdir(root_dir):
        main_category_path = os.path.join(root_dir, main_category)
        if not os.path.isdir(main_category_path):
            continue
        for patient_dir in os.listdir(main_category_path):
            patient_path = os.path.join(main_category_path, patient_dir)
            if not os.path.isdir(patient_path):
                continue
            for img_name in os.listdir(patient_path):
                if img_name.endswith(('.jpg')):
                    img_path = os.path.join(patient_path, img_name)
                    all_image_paths.append(img_path)

    nail_groups = {}
    for img_path in all_image_paths:
        filename = Path(img_path).name
        file_info = NailDataset([], None).parse_filename(filename)
        nail_id = f"{file_info['patient_id']}_{file_info['limb']}_{file_info['digit']}"
        if nail_id not in nail_groups:
            nail_groups[nail_id] = {
                'paths': [],
                'class_name': Path(img_path).parent.parent.name,  
                'has_uv': False,
                'has_normal': False
            }
        nail_groups[nail_id]['paths'].append(img_path)
        if file_info['is_uv']:
            nail_groups[nail_id]['has_uv'] = True
        else:
            nail_groups[nail_id]['has_normal'] = True

    nail_ids = list(nail_groups.keys())
    class_labels = [nail_groups[nail_id]['class_name'] for nail_id in nail_ids]  

    train_nail_ids, test_nail_ids = train_test_split(
        nail_ids,
        test_size=0.2,
        random_state=42,
        stratify=class_labels 
    )

    train_paths = [path for nail_id in train_nail_ids for path in nail_groups[nail_id]['paths']]
    test_paths = [path for nail_id in test_nail_ids for path in nail_groups[nail_id]['paths']]

    train_dataset = NailDataset(train_paths, transform=transform_uv if 'UV' in train_paths[0] else transform_normal)
    test_dataset = NailDataset(test_paths, transform=transform_uv if 'UV' in test_paths[0] else transform_normal)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def get_dataset_stats(root_dir):
    stats = {
        'total_images': 0,
        'categories': {},
        'patients_per_category': {},
        'limb_stats': {},
        'uv_stats': {'UV': 0, 'normal': 0}
    }

    dataset = NailDataset([])
    for main_category in os.listdir(root_dir):
        main_category_path = os.path.join(root_dir, main_category)
        if not os.path.isdir(main_category_path):
            continue
        stats['categories'][main_category] = {'total': 0, 'UV': 0, 'normal': 0}
        patient_set = set()
        for patient_dir in os.listdir(main_category_path):
            patient_path = os.path.join(main_category_path, patient_dir)
            if not os.path.isdir(patient_path):
                continue
            for img_name in os.listdir(patient_path):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    file_info = dataset.parse_filename(img_name)
                    patient_set.add(file_info['patient_id'])
                    stats['total_images'] += 1
                    stats['categories'][main_category]['total'] += 1
                    if file_info['is_uv']:
                        stats['categories'][main_category]['UV'] += 1
                        stats['uv_stats']['UV'] += 1
                    else:
                        stats['categories'][main_category]['normal'] += 1
                        stats['uv_stats']['normal'] += 1
                    limb = file_info['limb_full']
                    if limb not in stats['limb_stats']:
                        stats['limb_stats'][limb] = 0
                    stats['limb_stats'][limb] += 1
        stats['patients_per_category'][main_category] = len(patient_set)

    return stats

if __name__ == '__main__':
    root_dir = 'datasets/dataset'
    mean_uv, std_uv, mean_normal, std_normal = get_dataset_statistics(root_dir)
    print(mean_uv, std_uv, mean_normal, std_normal)