"""
Multimodal Defect Classification Model for Task 1.
Combines 2D CNN (Image Surface) and 1D CNN + BiGRU (Press Sensors)
with Gated Multimodal Fusion and Modality Dropout for robustness against missing sensor data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.GELU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        res = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + res)
        return out

class ImageEncoder(nn.Module):
    """
    CNN Feature Extractor for 64x64 product surface images.
    """
    def __init__(self, embed_dim=128):
        super().__init__()
        self.init_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU()
        )
        self.layer1 = ResBlock2D(32, 64, stride=2)   # -> 32x32
        self.layer2 = ResBlock2D(64, 128, stride=2)  # -> 16x16
        self.layer3 = ResBlock2D(128, 256, stride=2) # -> 8x8
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

    def forward(self, x):
        # x: (B, 1, 64, 64)
        out = self.init_conv(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.pool(out).flatten(1)
        out = self.fc(out)
        return out

class SensorEncoder(nn.Module):
    """
    1D-CNN + BiGRU Feature Extractor for 3-channel time-series press sensor signals (3, 128).
    """
    def __init__(self, embed_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2), # -> (64, 64)
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(2)  # -> (128, 32)
        )
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        ) # hidden 2*64 = 128
        self.attn = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.fc = nn.Sequential(
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

    def forward(self, x):
        # x: (B, 3, 128)
        feat = self.conv(x) # (B, 128, 32)
        feat = feat.permute(0, 2, 1) # (B, 32, 128)
        gru_out, _ = self.gru(feat) # (B, 32, 128)
        
        # Attention pooling
        weights = F.softmax(self.attn(gru_out), dim=1) # (B, 32, 1)
        pooled = torch.sum(gru_out * weights, dim=1)  # (B, 128)
        out = self.fc(pooled)
        return out

class MultimodalDefectClassifier(nn.Module):
    """
    Full Multimodal Architecture with Gated Fusion, Modality Dropout,
    and Missing-Modality Tolerance.
    """
    def __init__(self, num_classes=7, embed_dim=128, modality_dropout=0.15):
        super().__init__()
        self.embed_dim = embed_dim
        self.modality_dropout = modality_dropout
        
        self.image_encoder = ImageEncoder(embed_dim=embed_dim)
        self.sensor_encoder = SensorEncoder(embed_dim=embed_dim)
        
        # Gated Multimodal Unit (GMU)
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )
        
        # Missing modality learned fallback representation
        self.sensor_null_token = nn.Parameter(torch.zeros(1, embed_dim))
        
        # Final classifier head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, image, sensor, sensor_valid=None):
        # image: (B, 1, 64, 64)
        # sensor: (B, 3, 128)
        # sensor_valid: (B, 3) or (B,)
        B = image.size(0)
        z_img = self.image_encoder(image) # (B, embed_dim)
        z_sen = self.sensor_encoder(sensor) # (B, embed_dim)
        
        # Handle missing sensor: if sensor is invalid or modality dropped
        if sensor_valid is not None:
            # sensor_valid check: if any channel is valid
            if sensor_valid.dim() > 1:
                is_valid = (sensor_valid.sum(dim=-1, keepdim=True) > 0).float()
            else:
                is_valid = sensor_valid.unsqueeze(1).float()
            null_tokens = self.sensor_null_token.expand(B, -1)
            z_sen = is_valid * z_sen + (1.0 - is_valid) * null_tokens
            
        # Modality dropout during training: randomly drop sensor to force image robustness
        if self.training and self.modality_dropout > 0:
            mask = (torch.rand(B, 1, device=image.device) > self.modality_dropout).float()
            null_tokens = self.sensor_null_token.expand(B, -1)
            z_sen = mask * z_sen + (1.0 - mask) * null_tokens
            
        # Gated fusion
        gate_weights = self.gate(torch.cat([z_img, z_sen], dim=-1))
        fused = gate_weights * z_img + (1.0 - gate_weights) * z_sen
        
        # Combined representation
        out_repr = torch.cat([fused, z_img], dim=-1)
        logits = self.classifier(out_repr)
        return logits
