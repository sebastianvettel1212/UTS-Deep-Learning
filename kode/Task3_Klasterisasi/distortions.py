"""
Exploration of Multimodal Distortions for Task 3.
Inspects and visualizes:
1. Image distortions:
   - Microscope differences (M1 vs M2 brightness and contrast distribution)
   - Rotation, scale, and noise
2. Spectrum distortions:
   - Baseline drift (polynomial baseline offsets)
   - Spectral peak shifts
   - High-frequency spikes and detector noise
Generates figures/distortions_exploration.png for the report.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from utils.data_loader import load_task3_data

def analyze_and_plot_distortions(output_dir='laporan/figures'):
    os.makedirs(output_dir, exist_ok=True)
    images, spectra, meta_df = load_task3_data()
    mics = meta_df['mikroskop'].values
    
    print("=== Analyzing Image Distortions ===")
    m1_mask = (mics == 'M1')
    m2_mask = (mics == 'M2')
    
    m1_means = images[m1_mask].mean(axis=(1, 2))
    m2_means = images[m2_mask].mean(axis=(1, 2))
    m1_stds = images[m1_mask].std(axis=(1, 2))
    m2_stds = images[m2_mask].std(axis=(1, 2))
    
    print(f"Microscope M1: Mean pixel intensity = {m1_means.mean():.2f} +/- {m1_means.std():.2f}")
    print(f"Microscope M2: Mean pixel intensity = {m2_means.mean():.2f} +/- {m2_means.std():.2f}")
    print(f"Microscope M1: Mean contrast (std) = {m1_stds.mean():.2f}")
    print(f"Microscope M2: Mean contrast (std) = {m2_stds.mean():.2f}")
    
    print("\n=== Analyzing Spectrum Distortions ===")
    spec_mins = spectra.min(axis=1)
    spec_maxs = spectra.max(axis=1)
    spec_baselines = spectra[:, :10].mean(axis=1) # Initial channels indicate baseline offset
    
    print(f"Spectrum baseline offset range: [{spec_baselines.min():.3f}, {spec_baselines.max():.3f}], mean: {spec_baselines.mean():.3f}")
    print(f"Spectrum min value range: [{spec_mins.min():.3f}, {spec_mins.max():.3f}]")
    print(f"Spectrum max value range: [{spec_maxs.min():.3f}, {spec_maxs.max():.3f}]")
    
    # Generate multi-panel diagnostic figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=150)
    plt.subplots_adjust(hspace=0.35, wspace=0.25)
    
    # 1. Image sample M1
    m1_idx = np.where(m1_mask)[0][0]
    axes[0, 0].imshow(images[m1_idx], cmap='gray')
    axes[0, 0].set_title(f"Microstructure (M1: {mics[m1_idx]})\nMean Brightness: {images[m1_idx].mean():.1f}", fontsize=11)
    axes[0, 0].axis('off')
    
    # 2. Image sample M2
    m2_idx = np.where(m2_mask)[0][0]
    axes[0, 1].imshow(images[m2_idx], cmap='gray')
    axes[0, 1].set_title(f"Microstructure (M2: {mics[m2_idx]})\nMean Brightness: {images[m2_idx].mean():.1f}", fontsize=11)
    axes[0, 1].axis('off')
    
    # 3. Brightness distribution M1 vs M2
    axes[0, 2].hist(m1_means, bins=40, alpha=0.6, label='M1 (Standard)', color='tab:blue', density=True)
    axes[0, 2].hist(m2_means, bins=40, alpha=0.6, label='M2 (High Gain)', color='tab:orange', density=True)
    axes[0, 2].set_title("Microscope Brightness Shift", fontsize=11)
    axes[0, 2].set_xlabel("Mean Image Intensity")
    axes[0, 2].set_ylabel("Density")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Spectra showing baseline drift
    x_spec = np.arange(128)
    for i in range(5):
        axes[1, 0].plot(x_spec, spectra[i], alpha=0.8, label=f"Sample {i}")
    axes[1, 0].set_title("Spectra: Baseline Drift & Offset", fontsize=11)
    axes[1, 0].set_xlabel("Channel Index (1-128)")
    axes[1, 0].set_ylabel("Intensity")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Baseline offset histogram
    axes[1, 1].hist(spec_baselines, bins=40, color='purple', alpha=0.7, edgecolor='black', linewidth=0.5)
    axes[1, 1].set_title("Spectrum Baseline Distribution", fontsize=11)
    axes[1, 1].set_xlabel("Estimated Baseline (First 10 Channels)")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Peak alignment / Shift inspection
    # Overlay 20 normalized spectra around the primary peak
    norm_spectra = spectra[:25] - spectra[:25].min(axis=1, keepdims=True)
    norm_spectra = norm_spectra / (norm_spectra.max(axis=1, keepdims=True) + 1e-6)
    for i in range(15):
        axes[1, 2].plot(x_spec, norm_spectra[i], alpha=0.5)
    axes[1, 2].set_title("Normalized Spectra (Peak Shifts & Spikes)", fontsize=11)
    axes[1, 2].set_xlabel("Channel Index")
    axes[1, 2].set_ylabel("Normalized Intensity")
    axes[1, 2].grid(True, alpha=0.3)
    
    out_path = os.path.join(output_dir, 'distortions_exploration.png')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Distortion exploration figure saved to {out_path}")

if __name__ == '__main__':
    analyze_and_plot_distortions()
