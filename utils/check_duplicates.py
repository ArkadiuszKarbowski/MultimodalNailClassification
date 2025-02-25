import os
import argparse
from collections import defaultdict

def get_patient_ids(folder_path):
    patient_ids = defaultdict(set)
    
    for disease in os.listdir(folder_path):
        disease_path = os.path.join(folder_path, disease)
        if os.path.isdir(disease_path):
            for patient_folder in os.listdir(disease_path):
                patient_id = patient_folder.split(" ")[0]
                patient_ids[patient_id].add(disease)
    
    return patient_ids

def find_duplicate_patients(folder_path):
    patient_ids = get_patient_ids(folder_path)
    duplicates = {pid: diseases for pid, diseases in patient_ids.items() if len(diseases) > 1}
    
    if duplicates:
        print("Patients present in multiple disease categories:")
        for pid, diseases in duplicates.items():
            print(f"Patient {pid} found in: {', '.join(diseases)}")
    else:
        print("No duplicate patients found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check for duplicate patients across disease categories.")
    parser.add_argument("folder_path", type=str, help="Path to the dataset folder")
    args = parser.parse_args()
    
    find_duplicate_patients(args.folder_path)
