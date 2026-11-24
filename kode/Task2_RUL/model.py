"""
Models for Task 2 (RUL Regression).
1. Classical Baseline: Window Feature Engineering + Ridge / Random Forest
2. Deep Sequence Model: BiLSTM with Temporal Attention and MC Dropout
3. Custom Asymmetric Loss Function
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss Function directly optimizing competition score:
    d = y_pred - y_true
    s = exp(-d/13) - 1 if d < 0
        exp(d/10) - 1 if d >= 0
    Includes smooth gradient clamping for numerical stability during training.
    """
    def __init__(self, clip_val=20.0):
        super().__init__()
        self.clip_val = clip_val

    def forward(self, y_pred, y_true):
        d = (y_pred - y_true).squeeze(-1)
        
        # Smooth Huber-like clamping for extreme errors
        d_clamped = torch.clamp(d, -self.clip_val, self.clip_val)
        
        loss_under = torch.exp(-d_clamped / 13.0) - 1.0
        loss_over = torch.exp(d_clamped / 10.0) - 1.0
        
        base_loss = torch.where(d < 0, loss_under, loss_over)
        
        # Add linear penalty for outliers exceeding clip_val
        excess = torch.relu(torch.abs(d) - self.clip_val)
        total_loss = base_loss + 2.0 * excess
        return torch.mean(total_loss)

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, rnn_out):
        # rnn_out: (B, seq_len, hidden_dim)
        scores = self.attn(rnn_out) # (B, seq_len, 1)
        weights = F.softmax(scores, dim=1) # (B, seq_len, 1)
        context = torch.sum(rnn_out * weights, dim=1) # (B, hidden_dim)
        return context, weights

class BiLSTMAttentionRUL(nn.Module):
    """
    Bidirectional LSTM with Temporal Attention and MC Dropout for RUL estimation.
    """
    def __init__(self, input_dim=24, hidden_dim=64, num_layers=2, dropout_p=0.2):
        super().__init__()
        self.dropout_p = dropout_p
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_p if num_layers > 1 else 0.0
        )
        self.attention = TemporalAttention(hidden_dim * 2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x: (B, seq_len, input_dim)
        h_proj = self.proj(x)
        lstm_out, _ = self.lstm(h_proj)
        context, _ = self.attention(lstm_out)
        out = self.fc(context) # (B, 1)
        # RUL is non-negative and capped at 150
        out = torch.clamp(out, 0.0, 150.0)
        return out

    def predict_with_uncertainty(self, x, num_mc_samples=30):
        """
        Monte Carlo Dropout for epistemic uncertainty estimation.
        Keeps dropout active at test time and computes predictive mean & std.
        """
        self.train() # Enable dropout
        preds = []
        with torch.no_grad():
            for _ in range(num_mc_samples):
                pred = self.forward(x).squeeze(-1)
                preds.append(pred)
        preds = torch.stack(preds, dim=0) # (num_mc_samples, B)
        mean_pred = torch.mean(preds, dim=0)
        std_pred = torch.std(preds, dim=0)
        return mean_pred, std_pred

class ClassicalBaselineModel:
    """
    Classical baseline: engineered window features (mean, std, min, max, delta)
    fit with Ridge Regression or Random Forest.
    """
    def __init__(self):
        from sklearn.ensemble import RandomForestRegressor
        self.model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)

    def extract_features(self, X_seq):
        # X_seq: (N, W, D)
        feat_mean = np.mean(X_seq, axis=1)
        feat_std = np.std(X_seq, axis=1)
        feat_min = np.min(X_seq, axis=1)
        feat_max = np.max(X_seq, axis=1)
        feat_last = X_seq[:, -1, :]
        feat_delta = X_seq[:, -1, :] - X_seq[:, 0, :]
        return np.hstack([feat_mean, feat_std, feat_min, feat_max, feat_last, feat_delta])

    def fit(self, X_seq, y):
        X_feats = self.extract_features(X_seq)
        self.model.fit(X_feats, y)

    def predict(self, X_seq):
        X_feats = self.extract_features(X_seq)
        preds = self.model.predict(X_feats)
        return np.clip(preds, 0.0, 150.0)
