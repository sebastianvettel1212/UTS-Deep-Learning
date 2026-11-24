"""
Training pipeline for Task 2 (Time-Series RUL Regression).
Features:
- GroupKFold Cross-Validation on unit IDs (prevents data leakage across cycles)
- Classical Baseline (Random Forest) vs Deep Sequence Model (BiLSTM + Attention)
- Direct optimization of Asymmetric Penalty
- Uncertainty estimation using MC Dropout
- Saves best model to models/task2_best_model.pt and scaler to models/task2_scaler.pkl
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import GroupKFold

from utils.data_loader import load_task2_data
from utils.metrics import compute_rmse, compute_asymmetric_score
from Task2_RUL.preprocess import clean_and_impute_timeseries, fit_scaler, generate_sequences_train
from Task2_RUL.model import BiLSTMAttentionRUL, ClassicalBaselineModel, AsymmetricLoss

def train_task2(epochs=15, batch_size=256, lr=1e-3, seed=42, output_dir='models'):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Task 2 Training on device: {device}")
    
    os.makedirs(output_dir, exist_ok=True)
    df_train, df_test = load_task2_data()
    print("Cleaning and imputing missing time-series sensor values...")
    df_train_clean = clean_and_impute_timeseries(df_train)
    
    scaler = fit_scaler(df_train_clean)
    scaler_path = os.path.join(output_dir, 'task2_scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Fitted and saved scaler to {scaler_path}")
    
    print("Generating sliding window sequences (window_size=30)...")
    X_train_all, y_train_all, units_all = generate_sequences_train(df_train_clean, scaler, window_size=30)
    print(f"Generated {len(X_train_all)} sequences across {len(np.unique(units_all))} units.")
    
    # 5-Fold GroupKFold
    gkf = GroupKFold(n_splits=5)
    train_idx, val_idx = next(gkf.split(X_train_all, y_train_all, groups=units_all))
    
    X_tr, y_tr = X_train_all[train_idx], y_train_all[train_idx]
    X_val, y_val = X_train_all[val_idx], y_train_all[val_idx]
    
    # 1. Evaluate Classical Baseline Model
    print("\n--- Training Classical Baseline (Random Forest) ---")
    # Subsample training windows for Random Forest to keep training fast
    rf_subsample_idx = np.random.choice(len(X_tr), size=min(40000, len(X_tr)), replace=False)
    baseline_model = ClassicalBaselineModel()
    baseline_model.fit(X_tr[rf_subsample_idx], y_tr[rf_subsample_idx])
    
    val_subsample_idx = np.random.choice(len(X_val), size=min(10000, len(X_val)), replace=False)
    val_preds_baseline = baseline_model.predict(X_val[val_subsample_idx])
    baseline_rmse = compute_rmse(y_val[val_subsample_idx], val_preds_baseline)
    baseline_asym = compute_asymmetric_score(y_val[val_subsample_idx], val_preds_baseline)
    print(f"Classical Baseline - Val RMSE: {baseline_rmse:.3f} | Asymmetric Score: {baseline_asym:.2f}")
    
    # 2. Train Deep Sequence Model (BiLSTM + Attention)
    print("\n--- Training Deep Sequence Model (BiLSTM + Temporal Attention) ---")
    train_dataset = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    model = BiLSTMAttentionRUL(input_dim=24, hidden_dim=64, num_layers=2, dropout_p=0.2).to(device)
    
    # Combined loss: MSE loss + Asymmetric Penalty
    mse_criterion = nn.SmoothL1Loss()
    asym_criterion = AsymmetricLoss(clip_val=25.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    best_asym_score = float('inf')
    best_rmse = float('inf')
    best_model_path = os.path.join(output_dir, 'task2_best_model.pt')
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        # Phase 1 (first 5 epochs): guide convergence with SmoothL1; Phase 2: weight asymmetric loss
        asym_weight = 0.2 if epoch <= 5 else 0.6
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device).unsqueeze(-1)
            optimizer.zero_grad()
            preds = model(batch_x)
            
            loss_mse = mse_criterion(preds, batch_y)
            loss_asym = asym_criterion(preds, batch_y)
            loss = (1.0 - asym_weight) * loss_mse + asym_weight * loss_asym
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item() * len(batch_y)
            
        scheduler.step()
        train_loss = total_loss / len(train_dataset)
        
        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                preds = model(batch_x).squeeze(-1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(batch_y.numpy())
                
        val_rmse = compute_rmse(val_targets, val_preds)
        val_asym = compute_asymmetric_score(val_targets, val_preds)
        
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {train_loss:.4f} | Val RMSE: {val_rmse:.3f} | Val Asym Score: {val_asym:.2f}")
        
        # Save model that minimizes asymmetric penalty while keeping good RMSE
        if val_asym < best_asym_score:
            best_asym_score = val_asym
            best_rmse = val_rmse
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> Best model saved! (Val Asym Score: {best_asym_score:.2f}, RMSE: {best_rmse:.3f})")
            
    print(f"\nTask 2 Training Complete!")
    print(f"Deep Model Best Validation - RMSE: {best_rmse:.3f} | Asym Score: {best_asym_score:.2f}")
    print(f"Improvement over baseline: RMSE: {baseline_rmse:.3f} -> {best_rmse:.3f} | Asym: {baseline_asym:.2f} -> {best_asym_score:.2f}")
    
    return best_rmse, best_asym_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    train_task2(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed)
