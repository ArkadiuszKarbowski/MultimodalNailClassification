import os
import cv2
import numpy as np
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def process_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Could not read image at {img_path}. Skipping.")
        return img_path, None, None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    mean = img.mean(axis=(0, 1))
    std = img.std(axis=(0, 1))
    return img_path, mean, std


def process_image_batch(image_paths):
    results = []
    for img_path in image_paths:
        results.append(process_image(img_path))
    return results


def calculate_dataset_statistics(root_dir, batch_size=32):
    image_paths = []
    for main_category in os.listdir(root_dir):
        main_category_path = os.path.join(root_dir, main_category)
        if not os.path.isdir(main_category_path):
            continue
        for patient_dir in os.listdir(main_category_path):
            patient_path = os.path.join(main_category_path, patient_dir)
            if not os.path.isdir(patient_path):
                continue
            for img_name in os.listdir(patient_path):
                if img_name.endswith((".jpg", ".jpeg", ".png")):
                    img_path = os.path.join(patient_path, img_name)
                    image_paths.append(img_path)

    means_uv = []
    stds_uv = []
    means_normal = []
    stds_normal = []

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i : i + batch_size]
            futures.append(executor.submit(process_image_batch, batch))

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing images"
        ):
            batch_results = future.result()
            for img_path, mean, std in batch_results:
                if mean is None or std is None:
                    continue
                if "UV" in img_path:
                    means_uv.append(mean)
                    stds_uv.append(std)
                else:
                    means_normal.append(mean)
                    stds_normal.append(std)

    mean_uv = np.mean(means_uv, axis=0)
    std_uv = np.mean(stds_uv, axis=0)
    mean_normal = np.mean(means_normal, axis=0)
    std_normal = np.mean(stds_normal, axis=0)

    return {
        "mean_uv": mean_uv.tolist(),
        "std_uv": std_uv.tolist(),
        "mean_normal": mean_normal.tolist(),
        "std_normal": std_normal.tolist(),
    }


def save_to_json(data, output_path):
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate dataset statistics.")
    parser.add_argument("root_dir", type=str, help="Root directory of the dataset")
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/dataset/dataset_stats.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for image loading"
    )

    args = parser.parse_args()

    statistics = calculate_dataset_statistics(args.root_dir, args.batch_size)
    save_to_json(statistics, args.output)
    print(f"Statistics saved to {args.output}")
