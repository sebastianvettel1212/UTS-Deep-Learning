"""
Clustering Algorithms for Task 3:
1. Deep Embedded Clustering (DEC) using Student-t distribution and KL-divergence target refinement.
2. Classical Clustering: K-Means++ and Gaussian Mixture Models (GMM).
3. Quantitative comparison across Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Index.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from utils.metrics import compute_clustering_criteria

class ClusteringLayer(nn.Module):
    """
    Clustering layer converting latent representations to soft cluster assignments Q
    using Student-t distribution kernel.
    """
    def __init__(self, n_clusters, latent_dim, alpha=1.0):
        super().__init__()
        self.n_clusters = n_clusters
        self.latent_dim = latent_dim
        self.alpha = alpha
        # Centroids initialized by K-Means
        self.centroids = nn.Parameter(torch.Tensor(n_clusters, latent_dim))

    def forward(self, z):
        # z: (B, latent_dim)
        # Compute squared Euclidean distance between each z and each centroid
        # dist_sq[i, j] = ||z_i - mu_j||^2
        dist_sq = torch.sum((z.unsqueeze(1) - self.centroids.unsqueeze(0)) ** 2, dim=-1)
        
        # Student-t distribution soft assignment
        q = 1.0 / (1.0 + dist_sq / self.alpha)
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = q / torch.sum(q, dim=1, keepdim=True)
        return q

def target_distribution(q):
    """
    Computes auxiliary target distribution P from soft assignments Q:
    p_{ij} = (q_{ij}^2 / f_j) / sum_{j'} (q_{ij'}^2 / f_j')
    where f_j = sum_i q_{ij} is the cluster frequency.
    """
    weight = q ** 2 / torch.sum(q, dim=0)
    p = weight / torch.sum(weight, dim=1, keepdim=True)
    return p

class DeepEmbeddedClustering:
    """
    DEC model that initializes centroids with K-Means and fine-tunes
    cluster assignments by minimizing KL(P || Q).
    """
    def __init__(self, n_clusters, latent_dim, alpha=1.0, lr=1e-3, epochs=20, batch_size=256):
        self.n_clusters = n_clusters
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.cluster_layer = ClusteringLayer(n_clusters, latent_dim, alpha=alpha)

    def fit(self, embeddings, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        print(f"\n[DEC] Initializing {self.n_clusters} cluster centroids with K-Means++...")
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42)
        initial_labels = kmeans.fit_predict(embeddings)
        self.cluster_layer.centroids.data.copy_(torch.from_numpy(kmeans.cluster_centers_))
        
        self.cluster_layer.to(device)
        optimizer = torch.optim.Adam(self.cluster_layer.parameters(), lr=self.lr)
        
        tensor_embeddings = torch.from_numpy(embeddings).float().to(device)
        dataset = torch.utils.data.TensorDataset(tensor_embeddings)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        print("[DEC] Training Deep Embedded Clustering via KL divergence minimization...")
        for epoch in range(1, self.epochs + 1):
            # Compute full target distribution P every epoch
            with torch.no_grad():
                q_all = self.cluster_layer(tensor_embeddings)
                p_all = target_distribution(q_all)
                
            total_loss = 0.0
            for (batch_z,) in loader:
                optimizer.zero_grad()
                q_batch = self.cluster_layer(batch_z)
                # Compute KL divergence: sum(P * log(P / Q))
                with torch.no_grad():
                    p_batch = target_distribution(q_batch)
                loss = F.kl_div(q_batch.log(), p_batch, reduction='batchmean')
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            if epoch % 5 == 0 or epoch == self.epochs:
                current_preds = torch.argmax(self.cluster_layer(tensor_embeddings), dim=1).cpu().numpy()
                criteria = compute_clustering_criteria(embeddings, current_preds)
                print(f"DEC Epoch [{epoch:02d}/{self.epochs:02d}] - Loss: {total_loss/len(loader):.5f} | Silhouette: {criteria['silhouette']:.4f} | Davies-Bouldin: {criteria['davies_bouldin']:.4f}")

    def predict(self, embeddings, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cluster_layer.to(device)
        tensor_embeddings = torch.from_numpy(embeddings).float().to(device)
        with torch.no_grad():
            q = self.cluster_layer(tensor_embeddings)
            preds = torch.argmax(q, dim=1).cpu().numpy()
        return preds

def compare_clustering_algorithms(embeddings, n_clusters=7):
    """
    Compares K-Means++, GMM, and DEC on the learned embeddings.
    """
    print(f"\n==========================================")
    print(f"Comparing Clustering Methods on Embeddings (K={n_clusters})")
    print(f"==========================================")
    
    # 1. K-Means++
    print("Running K-Means++...")
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels_km = km.fit_predict(embeddings)
    crit_km = compute_clustering_criteria(embeddings, labels_km)
    print(f"K-Means++: Silhouette={crit_km['silhouette']:.4f}, Davies-Bouldin={crit_km['davies_bouldin']:.4f}, Calinski-Harabasz={crit_km['calinski_harabasz']:.1f}")
    
    # 2. Gaussian Mixture Model (GMM)
    print("Running Gaussian Mixture Model (GMM)...")
    gmm = GaussianMixture(n_components=n_clusters, random_state=42, covariance_type='diag', max_iter=100)
    labels_gmm = gmm.fit_predict(embeddings)
    crit_gmm = compute_clustering_criteria(embeddings, labels_gmm)
    print(f"GMM:       Silhouette={crit_gmm['silhouette']:.4f}, Davies-Bouldin={crit_gmm['davies_bouldin']:.4f}, Calinski-Harabasz={crit_gmm['calinski_harabasz']:.1f}")
    
    # 3. Deep Embedded Clustering (DEC)
    dec = DeepEmbeddedClustering(n_clusters=n_clusters, latent_dim=embeddings.shape[1], epochs=15)
    dec.fit(embeddings)
    labels_dec = dec.predict(embeddings)
    crit_dec = compute_clustering_criteria(embeddings, labels_dec)
    print(f"DEC:       Silhouette={crit_dec['silhouette']:.4f}, Davies-Bouldin={crit_dec['davies_bouldin']:.4f}, Calinski-Harabasz={crit_dec['calinski_harabasz']:.1f}")
    
    return {
        'kmeans': {'labels': labels_km, 'criteria': crit_km},
        'gmm': {'labels': labels_gmm, 'criteria': crit_gmm},
        'dec': {'labels': labels_dec, 'criteria': crit_dec, 'model': dec}
    }
