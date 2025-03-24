import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
from torch.amp import GradScaler, autocast
import torchmetrics
from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix
from tqdm import tqdm
import os
from datetime import datetime

from src.dataset import NailDataset
from src.model_archs.Resnet34_attention import MultimodalResNet
from src.utils.prepare_dataset import prepare_dataset
from src.augmentation import get_transforms


def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_files, val_files, test_files, stats = prepare_dataset(
        config["dataset_path"],
        verbose=False,
        modality=config["modality"],
        filter_incomplete=True,
        resize_shape=config["target_shape"],
    )

    train_dataset = NailDataset(
        train_files, transforms=get_transforms(stats, config["modality"], training=True)
    )

    # Data Loaders
    def collate_fn(batch):
        normals = torch.stack([item[0] for item in batch])
        uvs = torch.stack([item[1] for item in batch])
        labels = torch.stack([item[2] for item in batch])
        return normals, uvs, labels

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn,
    )

    val_dataset = NailDataset(
        val_files, transforms=get_transforms(stats, config["modality"], training=False)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=6,
        collate_fn=collate_fn,
    )

    # Model initialization
    model = MultimodalResNet(num_classes=config["num_classes"]).to(device)

    # Optimizer with weight decay
    optimizer = Adam(
        [
            {"params": model.normal_branch.parameters(), "lr": 1e-5},
            {"params": model.uv_branch.parameters(), "lr": 1e-5},
            {"params": model.attention.parameters()},
            {"params": model.classifier.parameters()},
        ],
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    # Scheduler and scaler
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"], eta_min=1e-6
    )
    scaler = GradScaler(device=device)

    # Class weights
    class_distribution = train_dataset.get_class_distribution()
    class_weights = class_distribution.sum() / class_distribution
    sqrt_weights = torch.sqrt(class_distribution.sum() / class_distribution)

    # alpha = class_weights / class_weights.sum() * 0.5

    # Loss and metrics
    # criterion = FocalLoss(alpha=alpha, gamma=1.5, reduction="mean").to(device)
    criterion = nn.CrossEntropyLoss(weight=sqrt_weights).to(device)
    train_metrics = {
        "acc": torchmetrics.Accuracy(
            task="multiclass", num_classes=config["num_classes"]
        ).to(device),
    }
    val_metrics = {
        "acc": torchmetrics.Accuracy(
            task="multiclass", num_classes=config["num_classes"]
        ).to(device),
        "per_class_acc": MulticlassAccuracy(
            num_classes=config["num_classes"], average=None
        ).to(device),
        "confusion_matrix": MulticlassConfusionMatrix(
            num_classes=config["num_classes"]
        ).to(device),
    }

    output_dir = "models"
    os.makedirs(output_dir, exist_ok=True)

    # Training loop
    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    best_val_acc = 0.0
    early_stop_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(config["epochs"]):
        # Training phase
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['epochs']} [Train]")

        for normals, uvs, labels in pbar:
            normals = normals.to(device, non_blocking=True)
            uvs = uvs.to(device, non_blocking=True)
            labels = labels.to(device)

            with autocast(device_type=device.type):
                outputs = model(normals, uvs)
                loss = criterion(outputs, labels)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            train_metrics["acc"](outputs, labels)

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                }
            )

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for normals, uvs, labels in val_loader:
                normals = normals.to(device)
                uvs = uvs.to(device)
                labels = labels.to(device)

                outputs = model(normals, uvs)
                val_loss += criterion(outputs, labels).item()
                val_metrics["acc"](outputs, labels)
                val_metrics["per_class_acc"](outputs, labels)
                val_metrics["confusion_matrix"](outputs, labels)
        # Metrics calculation
        train_loss = running_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        train_acc = train_metrics["acc"].compute().item()
        val_acc = val_metrics["acc"].compute().item()
        per_class_accuracies = val_metrics["per_class_acc"].compute()
        conf_matrix = val_metrics["confusion_matrix"].compute()

        # Update history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Scheduler step
        scheduler.step()

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(output_dir, f"model_{time}.pt"))
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        # Print metrics
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Train Acc: {train_acc:.2%}")
        print(f"Val Acc: {val_acc:.2%}")
        print(f"Best Val Acc: {best_val_acc:.2%}\n")
        print("Accuracy per class:")
        for i, acc in enumerate(per_class_accuracies):
            class_name = val_dataset.label_to_class(i)
            print(f"  Class {class_name}: {acc:.4f}")
        print("Confusion Matrix:")
        print(conf_matrix)

        # Reset metrics
        for metric in train_metrics.values():
            metric.reset()
        for metric in val_metrics.values():
            metric.reset()

        if early_stop_counter >= config["early_stop_patience"]:
            print("Early stopping triggered!")
            break

    return history, model


if __name__ == "__main__":
    # Configuration dictionary
    config = {
        "dataset_path": "datasets/preprocessed_dataset",
        "num_classes": 3,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "epochs": 50,
        "early_stop_patience": 10,
        "batch_size": 32,
        "modality": 2,
        "target_shape": (512, 512),
    }

    # Training
    history, model = train_model(config)
