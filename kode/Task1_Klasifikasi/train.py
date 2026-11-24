"""
Training pipeline for Task 1 Multimodal Defect Classification.
Optimized for high-speed CPU / GPU execution:
- Pre-vectorized tensor representations in memory
- Stratified K-Fold Cross Validation
- Class-balanced Label Smoothing Cross-Entropy (tackles label noise & extreme class imbalance)
- Modality Dropout (ensures robustness to missing sensor readings)
- Cosine Annealing learning rate schedule
- Saves best model checkpoint to models/task1_best_model.pt
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report

from utils.data_loader import load_task1_train, CLASS_NAMES
from utils.metrics import compute_macro_f1
from Task1_Klasifikasi.model import MultimodalDefectClassifier

class FastMultimodalDataset(Dataset):
    def __init__(self, images, sensors, sensor_valid, labels=None, is_train=True):
        # Pre-convert to tensors once in memory
        self.images = torch.from_numpy(images).float().unsqueeze(1) / 127.5 - 1.0
        self.sensors = torch.from_numpy(sensors).float()
        self.sensor_valid = torch.from_numpy(sensor_valid).float()
        self.labels = torch.from_numpy(labels).long() if labels is not None else None
        self.is_train = is_train

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        sen = self.sensors[idx]
        val = self.sensor_valid[idx]
        
        if self.is_train:
            if torch.rand(1).item() > 0.5:
                img = torch.flip(img, dims=[-1])
            if torch.rand(1).item() > 0.5:
                img = torch.flip(img, dims=[-2])
                
        if self.labels is not None:
            return img, sen, val, self.labels[idx]
        return img, sen, val

def train_task1(epochs=6, batch_size=256, lr=1e-3, seed=42, output_dir='models'):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(min(12, os.cpu_count() or 4))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Task 1 Training on device: {device} with {torch.get_num_threads()} threads")
    
    os.makedirs(output_dir, exist_ok=True)
    images, sensors, sensor_valid, labels, meta_df = load_task1_train()
    
    # Compute inverse class frequencies for weighted loss
    class_counts = np.bincount(labels, minlength=len(CLASS_NAMES))
    total_samples = len(labels)
    class_weights = total_samples / (len(CLASS_NAMES) * class_counts.astype(np.float32))
    class_weights = np.sqrt(class_weights)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    print(f"Class distribution: {dict(zip(CLASS_NAMES, class_counts))}")
    print(f"Smoothed class weights: {class_weights.cpu().numpy().round(3)}")
    
    # Stratified 5-Fold Split
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    train_idx, val_idx = next(skf.split(images, labels))
    
    train_dataset = FastMultimodalDataset(
        images[train_idx], sensors[train_idx], sensor_valid[train_idx], labels[train_idx], is_train=True
    )
    val_dataset = FastMultimodalDataset(
        images[val_idx], sensors[val_idx], sensor_valid[val_idx], labels[val_idx], is_train=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    model = MultimodalDefectClassifier(num_classes=len(CLASS_NAMES), embed_dim=128, modality_dropout=0.15).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    best_macro_f1 = 0.0
    best_model_path = os.path.join(output_dir, 'task1_best_model.pt')
    
    print("Beginning training loop...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for img, sen, val, target in train_loader:
            img, sen, val, target = img.to(device), sen.to(device), val.to(device), target.to(device)
            optimizer.zero_grad()
            logits = model(img, sen, val)
            loss = criterion(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item() * len(target)
            
        scheduler.step()
        train_loss = total_loss / len(train_dataset)
        
        # Validation
        model.eval()
        val_preds, val_targets = [], []
        val_loss = 0.0
        with torch.no_grad():
            for img, sen, val, target in val_loader:
                img, sen, val, target = img.to(device), sen.to(device), val.to(device), target.to(device)
                logits = model(img, sen, val)
                loss = criterion(logits, target)
                val_loss += loss.item() * len(target)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(target.cpu().numpy())
                
        val_loss /= len(val_dataset)
        val_macro_f1 = compute_macro_f1(val_targets, val_preds)
        
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Macro-F1: {val_macro_f1:.4f}")
        
        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> Best model saved! (Val Macro-F1: {best_macro_f1:.4f})")
            
    print(f"\nTask 1 Training Complete! Best Validation Macro-F1: {best_macro_f1:.4f}")
    
    # Detailed report on validation set
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for img, sen, val, target in val_loader:
            img, sen, val = img.to(device), sen.to(device), val.to(device)
            logits = model(img, sen, val)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_targets.extend(target.numpy())
            
    print("\nDetailed Classification Report (Best Model):")
    print(classification_report(val_targets, val_preds, target_names=CLASS_NAMES, digits=4))
    
    return best_macro_f1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    train_task1(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed)
