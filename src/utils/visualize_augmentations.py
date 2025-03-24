from src.dataset import NailDataset
from src.utils.prepare_dataset import prepare_dataset
from src.augmentation import get_transforms
import cv2
import random
import numpy as np
import os

os.environ["QT_QPA_PLATFORM"] = "xcb"


def denormalize(img, mean, std):
    img = img * std + mean
    img = img * 255
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def print_channel_info(img, name):
    """Print min, max, mean values for each channel to diagnose color issues"""
    print(f"--- {name} Image Channel Info ---")
    for i, channel in enumerate(["B", "G", "R"]):
        print(
            f"Channel {channel}: min={img[:, :, i].min():.2f}, max={img[:, :, i].max():.2f}, mean={img[:, :, i].mean():.2f}"
        )


if __name__ == "__main__":
    train, val, test, stats = prepare_dataset(
        "datasets/preprocessed_dataset",
        verbose=False,
        modality=2,
        filter_incomplete=True,
        resize_shape=(512, 512),
    )
    rand = random.randint(0, len(train))
    dataset = NailDataset(
        train[rand : rand + 1], transforms=get_transforms(stats, 2, training=True)
    )
    print(f"image path: '{dataset.dataset_items[0]['uv']}'")

    cv2.namedWindow("Normal", cv2.WINDOW_NORMAL)
    cv2.namedWindow("UV", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Normal", 512, 512)
    cv2.resizeWindow("UV", 512, 512)

    while True:
        normal, uv, label = dataset[0]
        normal = normal.numpy().transpose(1, 2, 0)
        uv = uv.numpy().transpose(1, 2, 0)

        normal = denormalize(normal, stats["mean_normal"], stats["std_normal"])
        uv = denormalize(uv, stats["mean_uv"], stats["std_uv"])

        normal = cv2.cvtColor(normal, cv2.COLOR_RGB2BGR)
        uv = cv2.cvtColor(uv, cv2.COLOR_RGB2BGR)

        cv2.imshow("Normal", normal)
        cv2.imshow("UV", uv)

        key = cv2.waitKey(0)
        if key == ord("q"):
            break
        elif key == ord("n"):
            continue
