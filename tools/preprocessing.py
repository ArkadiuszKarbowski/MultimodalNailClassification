import argparse
import os
import albumentations as A
import cv2
from tqdm import tqdm


def convert_image(image_path, resize):
    img = cv2.imread(image_path)
    resized_img = resize(image=img)["image"]

    return resized_img


def convert_images(dataset_dir, output_dir, target_size):
    target_height, target_width = target_size
    os.makedirs(output_dir, exist_ok=True)

    resize = A.Compose(
        [
            A.LongestMaxSize(max_size=max(target_height, target_width)),
            A.PadIfNeeded(
                min_height=target_height,
                min_width=target_width,
                border_mode=cv2.BORDER_CONSTANT,
            ),
            A.CenterCrop(height=target_height, width=target_width),
        ]
    )

    file_paths = []

    for class_name in os.listdir(dataset_dir):
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        if class_name == ".stats":
            continue
        output_class_dir = os.path.join(output_dir, class_name)
        for patient_id in os.listdir(class_dir):
            patient_dir = os.path.join(class_dir, patient_id)
            if not os.path.isdir(patient_dir):
                continue
            output_patient_dir = os.path.join(output_class_dir, patient_id)
            os.makedirs(output_patient_dir, exist_ok=True)

            for img_file in os.listdir(patient_dir):
                if img_file.endswith(".jpg"):
                    file_path = os.path.join(patient_dir, img_file)
                    output_path = os.path.join(output_patient_dir, img_file)
                    file_paths.append((file_path, output_path))

    for file_path, output_path in tqdm(file_paths):
        img = convert_image(file_path, resize)
        cv2.imwrite(output_path, img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess dataset")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        help="Path to the root folder of the dataset",
        default="datasets/dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to the output folder",
        default="datasets/preprocessed_dataset",
    )
    parser.add_argument(
        "--target_size",
        type=int,
        nargs=2,
        help="Target image size (height, width)",
        default=[512, 512],
    )

    args = parser.parse_args()

    convert_images(args.dataset_dir, args.output_dir, args.target_size)
