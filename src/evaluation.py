import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
    Accuracy,
)

from src.model_archs.Resnet18_attention_v2 import MultimodalResNet
from src.utils.prepare_dataset import prepare_dataset
from src.dataset import NailDataset
from src.augmentation import get_transforms


def test_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load datasets
    train_files, val_files, test_files, stats = prepare_dataset(
        config["dataset_path"],
        verbose=False,
        modality=config["modality"],
        filter_incomplete=True,
        resize_shape=config["target_shape"],
        seed=config["seed"],
    )

    model_path = config["model_path"]

    model = MultimodalResNet(num_classes=config["num_classes"]).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"Loaded model from {model_path}")

    # Test dataset
    test_dataset = NailDataset(
        test_files, transforms=get_transforms(stats, config["modality"], training=False)
    )
    print(f"Test dataset size: {len(test_dataset)} samples")

    # Class distribution
    class_distribution = test_dataset.get_class_distribution()
    print("Test class distribution:", class_distribution)

    def collate_fn(batch):
        normals = torch.stack([item[0] for item in batch])
        uvs = torch.stack([item[1] for item in batch])
        labels = torch.stack([item[2] for item in batch])
        return normals, uvs, labels

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn,
    )

    metrics = {
        "acc": Accuracy(task="multiclass", num_classes=config["num_classes"]).to(
            device
        ),
        "per_class_acc": MulticlassAccuracy(
            num_classes=config["num_classes"], average=None
        ).to(device),
        "confusion_matrix": MulticlassConfusionMatrix(
            num_classes=config["num_classes"]
        ).to(device),
    }

    # Test loop
    print("Evaluating model...")
    with torch.no_grad():
        for normals, uvs, labels in test_loader:
            normals = normals.to(device)
            uvs = uvs.to(device)
            labels = labels.to(device)

            outputs = model(normals, uvs)
            _, predicted = torch.max(outputs, 1)

            # Update metrics
            for metric in metrics.values():
                metric(predicted, labels)

    # Calculate and print results
    accuracy = metrics["acc"].compute()
    per_class_accuracies = metrics["per_class_acc"].compute()
    conf_matrix = metrics["confusion_matrix"].compute()

    print("\n" + "=" * 50)
    print(f"Test Accuracy: {accuracy:.2f}")
    print("=" * 50)

    print("\nPer-class Accuracy:")
    for i, acc in enumerate(per_class_accuracies):
        class_name = test_dataset.label_to_class(i)
        print(f"  Class {class_name}: {acc.item():.4f}")

    print("\nConfusion Matrix:")
    print(conf_matrix)

    plt.figure(figsize=(10, 8))
    conf_matrix_np = conf_matrix.cpu().numpy()
    class_names = [test_dataset.label_to_class(i) for i in range(config["num_classes"])]

    sns.heatmap(
        conf_matrix_np,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    # Save results
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/confusion_matrix.png")
    print("Confusion matrix saved to results/confusion_matrix.png")

    return accuracy, per_class_accuracies, conf_matrix


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to config file")
    args = parser.parse_args()

    config = json.load(open(args.config))

    test_model(config)
