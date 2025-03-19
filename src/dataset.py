import cv2
from torch.utils.data import Dataset
from utils.get_metadata import get_img_metadata
import torch
from torch.utils.data import Dataset
import cv2
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
        self.modality_type = self._determine_modality()
        self.label_encoder = LabelEncoder()
        
        # Process items and validate consistency
        self.processed_items = []
        self._process_items()
        
        # Initialize label encoder
        all_labels = [item['label'] for item in self.processed_items]
        self.label_encoder.fit(all_labels)

    def _determine_modality(self):
        """Determine modality type from keys"""
        first_item = next(iter(self.dataset_items), {})
        keys = set(first_item.keys())
        
        if {'H_normal', 'H_uv', 'P_normal', 'P_uv'}.issubset(keys):
            return 4
        if {'normal', 'uv'}.issubset(keys):
            return 2
        if 'file_path' in keys:
            return 1
        raise ValueError("Unknown modality structure")

    def _process_items(self):
        """Validate and preprocess all items"""
        for raw_item in self.dataset_items:
            processed = {'images': {}, 'label': None}
            labels = set()
            
            # Process each modality in deterministic order
            for key in sorted(raw_item.keys()):
                path = raw_item[key]
                if path is None:
                    continue
                    
                metadata = get_img_metadata(path)
                metadata['image_path'] = path
                
                # Validate required fields
                if 'class' not in metadata:
                    raise ValueError(f"Missing class in metadata for {path}")
                if 'is_uv' not in metadata:
                    raise ValueError(f"Missing is_uv in metadata for {path}")
                
                # Store image data and collect labels
                processed['images'][key] = {
                    'path': path,
                    'is_uv': metadata['is_uv'],
                    'metadata': metadata
                }
                labels.add(metadata['class'])
            
            # Verify label consistency
            if len(labels) != 1:
                raise ValueError(f"Multiple classes in item: {labels}")
            
            processed['label'] = labels.pop()
            self.processed_items.append(processed)

    def __len__(self):
        return len(self.processed_items)

    def __getitem__(self, idx):
        item = self.processed_items[idx]
        
        images = {}
        transformed = {}
        for key in sorted(item['images'].keys()):
            path = item['images'][key]['path']
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            images[key] = img
    
        if self.modality_type == 2:
            # Extract the 'image' from the returned dict
            transformed['image'] = self.transforms['resize_transform'](image=images['normal'])['image']
            transformed['uv'] = self.transforms['resize_transform'](image=images['uv'])['image']
            transformed = self.transforms['shared_transform'](image=transformed['image'], uv=transformed['uv'])
            transformed['image'] = self.transforms['normal_transform'](image=transformed['image'])['image']
            transformed['uv'] = self.transforms['uv_transform'](image=transformed['uv'])['image']
            label = torch.tensor(self.label_encoder.transform([item['label']])[0], dtype=torch.long)
            return transformed['image'], transformed['uv'], label



    def get_metadata(self, idx):
        """Helper to access raw metadata"""
        return self.processed_items[idx]['images']
    

if __name__ == "__main__":
    from utils.prepare_dataset import prepare_dataset
    from augmentation import get_transforms
    modality = 2 
    train_files, val_files, test_files, stats = prepare_dataset(
        "datasets/dataset",
        verbose=False,
        modality=modality,
    )

    dataset = NailDataset(train_files[:5], transforms=get_transforms(stats, modality, trainning=True))
    print(f"Modality: {dataset.modality_type}")
    print(f"Dataset length: {len(dataset)}")
    print(f"Dataset item: {dataset[0]}")
    print(f"Images shape: {dataset[0][0].shape}, {dataset[0][1].shape}")

