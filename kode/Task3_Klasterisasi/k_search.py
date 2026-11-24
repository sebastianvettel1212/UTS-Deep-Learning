"""
Evaluation of the optimal number of clusters K for Task 3.
Evaluates K in range [4, 12] across multiple criteria:
1. Silhouette Score (higher is better)
2. Davies-Bouldin Index (lower is better)
3. Inertia / Elbow Method (change of slope)
4. Calinski-Harabasz Index (higher is better)
Saves figures/k_search_criteria.png.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from utils.metrics import compute_clustering_criteria

def search_optimal_k(embeddings, k_range=range(4, 13), output_dir='laporan/figures'):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nEvaluating optimal K across range: {list(k_range)}...")
    
    inertias = []
    silhouettes = []
    davies_bouldins = []
    calinskis = []
    
    # Subsample 10000 samples for fast criteria computation if needed
    n_samples = len(embeddings)
    if n_samples > 10000:
        sub_idx = np.random.choice(n_samples, size=10000, replace=False)
        sub_emb = embeddings[sub_idx]
    else:
        sub_emb = embeddings
        
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit(embeddings)
        inertias.append(km.inertia_)
        
        preds_sub = km.predict(sub_emb)
        crit = compute_clustering_criteria(sub_emb, preds_sub)
        silhouettes.append(crit['silhouette'])
        davies_bouldins.append(crit['davies_bouldin'])
        calinskis.append(crit['calinski_harabasz'])
        
        print(f"  K={k:02d} | Inertia={km.inertia_:.1f} | Silhouette={crit['silhouette']:.4f} | Davies-Bouldin={crit['davies_bouldin']:.4f} | Calinski-Harabasz={crit['calinski_harabasz']:.1f}")
        
    # Plotting criteria curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    plt.subplots_adjust(hspace=0.3, wspace=0.25)
    
    ks = list(k_range)
    # 1. Inertia (Elbow)
    axes[0, 0].plot(ks, inertias, 'o-', color='tab:blue', linewidth=2, markersize=7)
    axes[0, 0].set_title("Elbow Method (Inertia / WCSS)", fontsize=12)
    axes[0, 0].set_xlabel("Number of Clusters (K)")
    axes[0, 0].set_ylabel("Inertia")
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Silhouette Score
    axes[0, 1].plot(ks, silhouettes, 's-', color='tab:green', linewidth=2, markersize=7)
    axes[0, 1].set_title("Silhouette Score (Higher is Better)", fontsize=12)
    axes[0, 1].set_xlabel("Number of Clusters (K)")
    axes[0, 1].set_ylabel("Silhouette Score")
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Davies-Bouldin Index
    axes[1, 0].plot(ks, davies_bouldins, '^-', color='tab:red', linewidth=2, markersize=7)
    axes[1, 0].set_title("Davies-Bouldin Index (Lower is Better)", fontsize=12)
    axes[1, 0].set_xlabel("Number of Clusters (K)")
    axes[1, 0].set_ylabel("Davies-Bouldin Index")
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Calinski-Harabasz Index
    axes[1, 1].plot(ks, calinskis, 'd-', color='tab:purple', linewidth=2, markersize=7)
    axes[1, 1].set_title("Calinski-Harabasz Index (Higher is Better)", fontsize=12)
    axes[1, 1].set_xlabel("Number of Clusters (K)")
    axes[1, 1].set_ylabel("Variance Ratio Criterion")
    axes[1, 1].grid(True, alpha=0.3)
    
    out_path = os.path.join(output_dir, 'k_search_criteria.png')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Optimal K criteria curves saved to {out_path}")
    
    # Optimal K from Silhouette
    best_k_sil = ks[np.argmax(silhouettes)]
    best_k_db = ks[np.argmin(davies_bouldins)]
    print(f"\nBest K by Silhouette Score: {best_k_sil} (Score: {max(silhouettes):.4f})")
    print(f"Best K by Davies-Bouldin:   {best_k_db} (Index: {min(davies_bouldins):.4f})")
    
    return {
        'best_k_silhouette': best_k_sil,
        'best_k_davies_bouldin': best_k_db,
        'ks': ks,
        'silhouettes': silhouettes,
        'davies_bouldins': davies_bouldins,
        'inertias': inertias
    }

if __name__ == '__main__':
    # Test stub
    pass
