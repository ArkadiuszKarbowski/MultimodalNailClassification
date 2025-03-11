import os
import numpy as np
import pandas as pd

def get_nail_id(filename):
    parts = filename.split()
    # PatientID + LimbCode + Digit uniquely identifies a nail
    return f"{parts[0]}_{parts[1]}_{parts[2]}"

def split_dataset(dataset_dir, val_ratio=0.1, test_ratio=0.1, random_state=2137, verbose=True):
    '''
    Splits the dataset into train, validation, and test sets at the nail level stratified by class.
    
    Parameters:
        dataset_dir (str): Path to the dataset directory.
        val_ratio (float): Ratio of validation nails.
        test_ratio (float): Ratio of test nails.
        random_state (int): Seed for random number generator.
        verbose (bool): Whether to print statistics.

    Output:
        train_files (list): List of file paths for the training set.
        val_files (list): List of file paths for the validation set.
        test_files (list): List of file paths for the test set.
    '''
    
    all_files = []
    all_nail_ids = []
    all_classes = []
    
    for class_name in os.listdir(dataset_dir):
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        for patient_id in os.listdir(class_dir):
            patient_dir = os.path.join(class_dir, patient_id)
            if not os.path.isdir(patient_dir):
                continue
                
            for img_file in os.listdir(patient_dir):
                if img_file.endswith('.jpg'):
                    file_path = os.path.join(patient_dir, img_file)
                    nail_id = get_nail_id(img_file)
                    
                    all_files.append(file_path)
                    all_nail_ids.append(nail_id)
                    all_classes.append(class_name)
    
    df = pd.DataFrame({
        'file_path': all_files,
        'nail_id': all_nail_ids,
        'class': all_classes
    })
    
    # Create a nail-level DataFrame (one row per nail)
    nail_df = df.drop_duplicates(subset=['nail_id'])[['nail_id', 'class']]
    
    train_nail_ids = []
    val_nail_ids = []
    test_nail_ids = []
    
    # Stratify by class
    for class_name in nail_df['class'].unique():
        # Get all nail IDs for this class
        class_nail_ids = nail_df[nail_df['class'] == class_name]['nail_id'].values
        np.random.seed(random_state)
        np.random.shuffle(class_nail_ids)
        
        n_total = len(class_nail_ids)
        n_test = int(n_total * test_ratio)
        n_val = int(n_total * val_ratio)
        n_train = n_total - n_test - n_val
        
        train_nail_ids.extend(class_nail_ids[:n_train])
        val_nail_ids.extend(class_nail_ids[n_train:n_train+n_val])
        test_nail_ids.extend(class_nail_ids[n_train+n_val:])
    
    train_files = df[df['nail_id'].isin(train_nail_ids)]['file_path'].tolist()
    val_files = df[df['nail_id'].isin(val_nail_ids)]['file_path'].tolist()
    test_files = df[df['nail_id'].isin(test_nail_ids)]['file_path'].tolist()
    
    if verbose:
        # Print statistics
        print(f"Total nails: {len(nail_df)}")
        print(f"Train nails: {len(train_nail_ids)} ({len(train_files)} images)")
        print(f"Val nails: {len(val_nail_ids)} ({len(val_files)} images)")
        print(f"Test nails: {len(test_nail_ids)} ({len(test_files)} images)")
        
        # Check class distribution in each split
        for split_name, nail_ids in [('Train', train_nail_ids), ('Val', val_nail_ids), ('Test', test_nail_ids)]:
            class_counts = df[df['nail_id'].isin(nail_ids)]['class'].value_counts(normalize=True)
            print(f"\n{split_name} class distribution:")
            print(class_counts)
    
    return train_files, val_files, test_files

if __name__ == "__main__":
    train_files, val_files, test_files = split_dataset('datasets/dataset')
