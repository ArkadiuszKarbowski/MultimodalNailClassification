import os

def split_dataset(dataset_dir, output_dir):
    image_paths = get_img_paths(dataset_dir)
    

def get_img_paths(dataset_dir):
    all_image_paths = []
    for class_dir in os.listdir(dataset_dir):
        class_path = os.path.join(dataset_dir, class_dir)
        if not os.path.isdir(class_dir):
            continue
        for patient_dir in os.listdir(class_path):
            patient_path = os.path.join(class_path, patient_dir)
            if not os.path.isdir(patient_path):
                continue
            for img_name in os.listdir(patient_path):
                if img_name.endswith((".jpg")):
                    img_path = os.path.join(patient_path, img_name)
                    all_image_paths.append(img_path)
    return all_image_paths

if __name__ == "__main__":
    pass