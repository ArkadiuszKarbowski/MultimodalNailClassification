import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
import torchmetrics
from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix
from tqdm import tqdm
import os
from datetime import datetime
import json

# import sys
import signal
from torch import nn

from src.dataset import NailDataset
from src.utils.prepare_dataset import prepare_dataset
from src.augmentation import get_transforms

# from src.focal_loss import FocalLoss
from src.utils.get_model import get_model


def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join("outputs", time)
    os.makedirs(output_dir, exist_ok=True)

    config["output_dir"] = output_dir
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f)

    best_model_path = os.path.join(output_dir, "best_model_state.pth")
    last_model_path = os.path.join(output_dir, "last_model_state.pth")
    history_path = os.path.join(output_dir, "history.json")

    # Initialize a stop training flag
    stop_training = False

    def save_checkpoint(final=False):
        torch.save(model.state_dict(), last_model_path)

        with open(history_path, "w") as f:
            json.dump(history, f)

        if final:
            print(f"\nTraining complete! History and models saved to {output_dir}")
        else:
            print(f"\nCheckpoint saved to {output_dir}")

    # Handle Ctrl+C more gracefully
    def signal_handler(sig, frame):
        nonlocal stop_training
        stop_training = True

    signal.signal(signal.SIGINT, signal_handler)

    train_files, val_files, test_files, stats = prepare_dataset(
        config["dataset_path"],
        verbose=False,
        modality=config["modality"],
        filter_incomplete=True,
        resize_shape=config["target_shape"],
        seed=config["seed"],
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

    def sampler(dataset):
        class_counts = dataset.get_class_distribution()
        class_weights = torch.sqrt(class_counts.sum() / class_counts)

        sample_weights = []
        for idx in range(len(dataset)):
            _, _, label = dataset[idx]
            sample_weights.append(class_weights[label].item())

        return torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(dataset),
            replacement=True,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        num_workers=6,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn,
        sampler=sampler(train_dataset),
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
    model = get_model(
        config["model_arch_path"],
        model_args={
            "num_classes": config["num_classes"],
            "initial_freeze": config["initial_freeze"],
        },
    )
    model = model.to(device)

    # Assert types for static analysis
    assert isinstance(model.normal_branch, nn.Module)
    assert isinstance(model.uv_branch, nn.Module)
    assert isinstance(model.classifier, nn.Module)

    # Optimizer with weight decay
    optimizer = Adam(
        [
            {"params": model.normal_branch.parameters(), "lr": config["backbone_lr"]},
            {"params": model.uv_branch.parameters(), "lr": config["backbone_lr"]},
            {"params": model.classifier.parameters(), "lr": config["classifier_lr"]},
        ],
        weight_decay=config["weight_decay"],
    )

    # Calculate total number of training steps
    total_steps = len(train_loader) * config["epochs"]
    warmup_steps = int(0.2 * total_steps)  # 20% warmup like pct_start in OneCycleLR

    # Factor to start with lower learning rate (same as div_factor in OneCycleLR)
    start_factor = 1 / 25

    # Warm-up scheduler - linear ramp from small lr to max_lr
    warmup_scheduler = lr_scheduler.LinearLR(
        optimizer, start_factor=start_factor, end_factor=1.0, total_iters=warmup_steps
    )

    # Main scheduler - cosine annealing from max_lr down to very small lr
    main_scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps, eta_min=config["backbone_lr"] / 1e4
    )

    # Combine schedulers
    scheduler = lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_steps],
    )

    scaler = GradScaler()

    # Class weights
    class_distribution = train_dataset.get_class_distribution()
    class_weights = class_distribution.sum() / class_distribution
    print("Class distribution:", class_distribution)
    print("Class weights:", class_weights)
    sqrt_weights = torch.sqrt(class_distribution.sum() / class_distribution)
    alpha = class_weights / class_weights.sum() * 0.5

    # Loss and metrics
    # criterion = FocalLoss(alpha=None, gamma=1.5, reduction="mean").to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
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
    # Training loop
    best_val_acc = 0.0
    early_stop_counter = 0
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "conf_matrix": [],
    }

    for epoch in range(config["epochs"]):
        # Unfreeze last blocks after specified number of epochs
        if epoch == config["unfreeze_after_epochs"]:
            if hasattr(model, "unfreeze_last_block") and callable(
                getattr(model, "unfreeze_last_block")
            ):
                model.unfreeze_last_block()
                print("Unfrozen last blocks in both branches")
            else:
                print(
                    f"Warning: Model does not have a callable 'unfreeze_last_block' method. Skipping unfreeze at epoch {epoch + 1}."
                )

        # Training phase
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['epochs']} [Train]")

        try:
            for normals, uvs, labels in pbar:
                if stop_training:
                    print("\nKeyboard interrupt received.")
                    break
                normals = normals.to(device, non_blocking=True)
                uvs = uvs.to(device, non_blocking=True)
                labels = labels.to(device)

                with autocast(device_type=device.type):
                    outputs = model(normals, uvs)
                    loss = criterion(outputs, labels)

                if torch.isnan(loss):
                    print("NaN detected! Skipping batch")
                    optimizer.zero_grad()
                    continue

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scheduler.step()
                scaler.update()

                running_loss += loss.item()
                train_metrics["acc"](outputs, labels)

                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "lr": f"{optimizer.param_groups[-1]['lr']:.2e}",
                    }
                )

            if stop_training:
                break

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
            history["conf_matrix"].append(conf_matrix.tolist())

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), best_model_path)
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
        except Exception as e:
            print(f"Error during training: {e}")
            save_checkpoint()
            raise e

    # Save final model and clean up resources
    save_checkpoint(final=not stop_training)

    # Clean up DataLoader workers
    if hasattr(train_loader, "_iterator") and train_loader._iterator is not None:
        train_loader._iterator._shutdown_workers()
    if hasattr(val_loader, "_iterator") and val_loader._iterator is not None:
        val_loader._iterator._shutdown_workers()

    return history, model


if __name__ == "__main__":
    config = {
        "dataset_path": "datasets/preprocessed_dataset",
        "num_classes": 3,
        "backbone_lr": 5e-6,  # Lower learning rate for backbone
        "classifier_lr": 2e-4,  # Reduced learning rate for classifier
        "weight_decay": 3e-4,  # Increased weight decay
        "epochs": 70,
        "early_stop_patience": 12,
        "batch_size": 8,
        "modality": 2,
        "target_shape": (512, 512),
        "seed": 2137,
        "model_arch_path": "src/model_archs/convnexts/Convnext_small_TL.py",
        "initial_freeze": True,
        "unfreeze_after_epochs": 10,
    }

    # Training
    history, model = train_model(config)
