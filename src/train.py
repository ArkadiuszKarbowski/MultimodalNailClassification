import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
from torch.amp import GradScaler, autocast
import torchmetrics
from tqdm import tqdm
import os

from dataset import NailDataset
from model_archs.Resnet18_attention import MultimodalResNet
from utils.prepare_dataset import prepare_dataset
from augmentation import get_transforms


def train_model(config):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_files, val_files, test_files, stats = prepare_dataset(
        "datasets/dataset",
        verbose=False,
        modality=config['modality'],
        filter_incomplete=True
    )

    train_dataset = NailDataset(train_files, transforms=get_transforms(stats, config['modality'], training=True))
    # Data Loaders
    def collate_fn(batch):
        normals = torch.stack([item[0] for item in batch])
        uvs = torch.stack([item[1] for item in batch])
        labels = torch.stack([item[2] for item in batch])
        return normals, uvs, labels

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn
    )
    
    val_dataset = NailDataset(val_files, transforms=get_transforms(stats, config['modality'], training=False))
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    # Model initialization
    model = MultimodalResNet(num_classes=config['num_classes']).to(device)
    
    # Optimizer with weight decay
    optimizer = Adam(
        [
        {'params': model.normal_branch.parameters(), 'lr': 1e-5},  
        {'params': model.uv_branch.parameters(), 'lr': 1e-5},
        {'params': model.attention.parameters()},
        {'params': model.classifier.parameters()}
    ],
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    # Scheduler and scaler
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',
        patience=3,
        factor=0.5
    )
    scaler = GradScaler(device=device)
    
    # Loss and metrics
    criterion = nn.CrossEntropyLoss()
    train_metrics = {
        'acc': torchmetrics.Accuracy(task='multiclass', num_classes=config['num_classes']).to(device),
        'top3_acc': torchmetrics.Accuracy(task='multiclass', num_classes=config['num_classes']).to(device)
    }
    val_metrics = {
        'acc': torchmetrics.Accuracy(task='multiclass', num_classes=config['num_classes']).to(device),
        'top3_acc': torchmetrics.Accuracy(task='multiclass', num_classes=config['num_classes']).to(device)
    }

    output_dir = 'models'
    os.makedirs(output_dir, exist_ok=True)

    # Training loop
    best_val_acc = 0.0
    early_stop_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(config['epochs']):
        # Training phase
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Train]")
        
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
            train_metrics['acc'](outputs, labels)
            train_metrics['top3_acc'](outputs, labels)
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
            })

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
                val_metrics['acc'](outputs, labels)
                val_metrics['top3_acc'](outputs, labels)

        # Metrics calculation
        train_loss = running_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        train_acc = train_metrics['acc'].compute().item()
        val_acc = val_metrics['acc'].compute().item()

        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        # Scheduler step
        scheduler.step(val_acc)

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        # Print metrics
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Train Acc: {train_acc:.2%}")
        print(f"Val Acc: {val_acc:.2%}")
        print(f"Best Val Acc: {best_val_acc:.2%}\n")

        # Reset metrics
        for metric in train_metrics.values():
            metric.reset()
        for metric in val_metrics.values():
            metric.reset()

        if early_stop_counter >= config['early_stop_patience']:
            print("Early stopping triggered!")
            break

    return history, model

if __name__ == "__main__":
    # Configuration dictionary
    config = {
        'num_classes': 3, 
        'lr': 1e-4,
        'weight_decay': 1e-4,
        'epochs': 50,
        'early_stop_patience': 7,
        'batch_size': 24,
        'modality': 2,
    }

    # Training
    history, model = train_model(config)