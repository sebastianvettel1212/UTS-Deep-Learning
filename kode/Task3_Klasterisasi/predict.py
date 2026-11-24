"""
Pipeline and Prediction Script for Task 3 Multimodal Clustering.
Learns unsupervised representations, performs optimal K evaluation,
applies Deep Embedded Clustering (DEC), and saves cluster.csv.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from utils.data_loader import load_task3_data
from Task3_Klasterisasi.distortions import analyze_and_plot_distortions
from Task3_Klasterisasi.representation import MultimodalMaterialAutoencoder
from Task3_Klasterisasi.clustering import compare_clustering_algorithms, DeepEmbeddedClustering
from Task3_Klasterisasi.k_search import search_optimal_k
from Task3_Klasterisasi.analysis import perform_tsne_ablation

def run_task3_pipeline(epochs_ae=10, batch_size=256, seed=42, output_file='cluster.csv', models_dir='models'):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Task 3 Clustering Pipeline on device: {device}")
    
    os.makedirs(models_dir, exist_ok=True)
    images, spectra, meta_df = load_task3_data()
    n_samples = len(images)
    print(f"Loaded {n_samples} samples.")
    
    # Step a: Distortions Exploration
    print("\n[Step a] Analyzing distortions and saving exploration plot...")
    analyze_and_plot_distortions()
    
    # Step b: Unsupervised Representation Learning (Multimodal Autoencoder)
    print("\n[Step b] Training Multimodal Autoencoder...")
    # Preprocess images to [-1, 1] and spectra to standardized scale
    norm_imgs = (images.astype(np.float32) / 255.0 - 0.5) / 0.5
    norm_imgs = np.expand_dims(norm_imgs, 1) # (N, 1, 64, 64)
    
    spec_mean = np.mean(spectra, axis=1, keepdims=True)
    spec_std = np.std(spectra, axis=1, keepdims=True) + 1e-6
    norm_spectra = (spectra - spec_mean) / spec_std
    norm_spectra = np.expand_dims(norm_spectra, 1) # (N, 1, 128)
    
    dataset = TensorDataset(torch.from_numpy(norm_imgs).float(), torch.from_numpy(norm_spectra).float())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    autoencoder = MultimodalMaterialAutoencoder(latent_dim=64).to(device)
    optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion_img = torch.nn.MSELoss()
    criterion_spec = torch.nn.MSELoss()
    
    ae_weights_path = os.path.join(models_dir, 'task3_multimodal_ae.pt')
    if os.path.exists(ae_weights_path):
        print(f"Loading existing autoencoder weights from {ae_weights_path}...")
        autoencoder.load_state_dict(torch.load(ae_weights_path, map_location=device))
    else:
        for epoch in range(1, epochs_ae + 1):
            autoencoder.train()
            total_loss = 0.0
            for batch_img, batch_spec in loader:
                batch_img, batch_spec = batch_img.to(device), batch_spec.to(device)
                optimizer.zero_grad()
                recon_img, recon_spec, _, _ = autoencoder(batch_img, batch_spec)
                loss_img = criterion_img(recon_img, batch_img)
                loss_spec = criterion_spec(recon_spec, batch_spec)
                loss = loss_img + loss_spec
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch_img)
            avg_loss = total_loss / n_samples
            print(f"  AE Epoch [{epoch:02d}/{epochs_ae:02d}] - Reconstruction Loss: {avg_loss:.5f}")
        torch.save(autoencoder.state_dict(), ae_weights_path)
        print(f"Autoencoder weights saved to {ae_weights_path}")
        
    # Extract representations for all samples
    print("\nExtracting latent representations...")
    autoencoder.eval()
    all_z_img, all_z_spec, all_z_fused = [], [], []
    eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch_img, batch_spec in eval_loader:
            batch_img, batch_spec = batch_img.to(device), batch_spec.to(device)
            z_i, z_s, z_f = autoencoder.extract_embeddings(batch_img, batch_spec)
            all_z_img.append(z_i.cpu().numpy())
            all_z_spec.append(z_s.cpu().numpy())
            all_z_fused.append(z_f.cpu().numpy())
            
    z_img = np.vstack(all_z_img)
    z_spec = np.vstack(all_z_spec)
    z_fused = np.vstack(all_z_fused)
    print(f"Latent shapes: Image={z_img.shape}, Spectrum={z_spec.shape}, Fused={z_fused.shape}")
    
    # Step e: Determine Number of Clusters K
    print("\n[Step e] Determining optimal number of clusters K...")
    k_res = search_optimal_k(z_fused, k_range=range(4, 11))
    optimal_k = k_res['best_k_silhouette']
    print(f"-> Selected optimal K = {optimal_k}")
    
    # Step d: Clustering Comparison & DEC
    print(f"\n[Step d] Comparing clustering methods (K={optimal_k})...")
    clust_res = compare_clustering_algorithms(z_fused, n_clusters=optimal_k)
    final_labels = clust_res['dec']['labels']
    
    # Step c & f: Modality Ablation & t-SNE Analysis
    print("\n[Step c & f] Modality ablation and t-SNE visualization...")
    perform_tsne_ablation(z_img, z_spec, z_fused, final_labels)
    
    # Save output predictions to cluster.csv
    res_df = pd.DataFrame({
        'id_sampel': meta_df['id_sampel'],
        'cluster': final_labels.astype(int)
    })
    
    res_df.to_csv(output_file, index=False)
    os.makedirs('prediksi', exist_ok=True)
    res_df.to_csv(os.path.join('prediksi', 'cluster.csv'), index=False)
    
    print(f"\nSaved {len(res_df)} cluster assignments to {output_file} and prediksi/cluster.csv")
    print(f"Cluster distribution:\n{res_df['cluster'].value_counts().sort_index()}")
    return res_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs_ae', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--output_file', type=str, default='cluster.csv')
    args = parser.parse_args()
    
    run_task3_pipeline(epochs_ae=args.epochs_ae, batch_size=args.batch_size, output_file=args.output_file)
