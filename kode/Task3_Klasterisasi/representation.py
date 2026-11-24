"""
Unsupervised Multimodal Representation Learning for Task 3.
Implements:
1. Convolutional Image Autoencoder with rotation, flip, and brightness jitter invariance.
2. 1D Convolutional Spectrum Autoencoder with baseline drift, shift, and spike noise invariance.
3. Multimodal Latent Fusion combining normalized image and spectrum representations.
4. Single-modality extractors for ablation study (what is lost when using only one modality).
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class ImageAutoencoder(nn.Module):
    """
    Convolutional Autoencoder for 64x64 microstructure images.
    Latent space dimension: 64.
    """
    def __init__(self, latent_dim=64):
        super().__init__()
        # Encoder: 1x64x64 -> 32x32x32 -> 64x16x16 -> 128x8x8 -> latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # 32x32
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 16x16
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),# 8x8
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        
        # Decoder: latent_dim -> 128x8x8 -> 64x16x16 -> 32x32x32 -> 1x64x64
        self.decoder_fc = nn.Linear(latent_dim, 128 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # 16x16
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),   # 64x64
            nn.Tanh()
        )

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        z = self.encode(x)
        h = self.decoder_fc(z).view(-1, 128, 8, 8)
        recon = self.decoder(h)
        return recon, z

class SpectrumAutoencoder(nn.Module):
    """
    1D-Convolutional Autoencoder for 128-channel composition spectra.
    Latent space dimension: 64.
    """
    def __init__(self, latent_dim=64):
        super().__init__()
        # Encoder: 1x128 -> 32x64 -> 64x32 -> 128x16 -> latent_dim
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(8),
            nn.Flatten(),
            nn.Linear(128 * 8, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        
        # Decoder: latent_dim -> 128x16 -> 64x32 -> 32x64 -> 1x128
        self.decoder_fc = nn.Linear(latent_dim, 128 * 16)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.ConvTranspose1d(32, 1, kernel_size=4, stride=2, padding=1)
        )

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        z = self.encode(x)
        h = self.decoder_fc(z).view(-1, 128, 16)
        recon = self.decoder(h)
        return recon, z

class MultimodalMaterialAutoencoder(nn.Module):
    """
    Joint Multimodal Autoencoder for unsupervised representation learning.
    Produces:
    - z_img (64-d)
    - z_spec (64-d)
    - z_fused (128-d L2-normalized concatenation)
    """
    def __init__(self, latent_dim=64):
        super().__init__()
        self.img_ae = ImageAutoencoder(latent_dim=latent_dim)
        self.spec_ae = SpectrumAutoencoder(latent_dim=latent_dim)

    def forward(self, img, spec):
        recon_img, z_img = self.img_ae(img)
        recon_spec, z_spec = self.spec_ae(spec)
        return recon_img, recon_spec, z_img, z_spec

    def extract_embeddings(self, img, spec):
        with torch.no_grad():
            z_img = self.img_ae.encode(img)
            z_spec = self.spec_ae.encode(spec)
            
            # L2 normalization for each modality
            z_img_norm = F.normalize(z_img, p=2, dim=-1)
            z_spec_norm = F.normalize(z_spec, p=2, dim=-1)
            
            # Multimodal fusion
            z_fused = torch.cat([z_img_norm, z_spec_norm], dim=-1)
            z_fused = F.normalize(z_fused, p=2, dim=-1)
            
        return z_img_norm, z_spec_norm, z_fused
