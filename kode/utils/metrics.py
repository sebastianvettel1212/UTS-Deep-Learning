"""
Evaluation metrics for UTS Deep Learning tasks.
Task 1: Macro-F1
Task 2: RMSE and Asymmetric Penalty Score
Task 3: Adjusted Rand Index (ARI), Silhouette, Davies-Bouldin
"""

import numpy as np

def compute_macro_f1(y_true, y_pred, num_classes=None):
    """
    Computes Macro-F1 score across all classes.
    """
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, average='macro'))

def compute_rmse(y_true, y_pred):
    """
    Root Mean Squared Error: sqrt(mean((y_pred - y_true)^2))
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

def compute_asymmetric_score(y_true, y_pred):
    """
    Asymmetric scoring function for RUL:
    d = y_pred - y_true
    s = sum_{i} (exp(-d/13) - 1) if d < 0 (early / conservative prediction)
        (exp(d/10) - 1) if d >= 0 (late / over-optimistic prediction, penalized more heavily!)
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    d = y_pred - y_true
    
    score = 0.0
    for diff in d:
        if diff < 0:
            score += np.exp(-diff / 13.0) - 1.0
        else:
            score += np.exp(diff / 10.0) - 1.0
    return float(score)

def compute_ari(y_true, y_pred):
    """
    Adjusted Rand Index for clustering comparison against ground truth.
    """
    from sklearn.metrics import adjusted_rand_score
    return float(adjusted_rand_score(y_true, y_pred))

def compute_clustering_criteria(X, labels):
    """
    Computes Silhouette Score and Davies-Bouldin Index.
    """
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    n_samples = len(X)
    # Use subset for silhouette if dataset is very large to speed up evaluation
    if n_samples > 10000:
        sample_indices = np.random.choice(n_samples, size=10000, replace=False)
        sil = float(silhouette_score(X[sample_indices], labels[sample_indices]))
    else:
        sil = float(silhouette_score(X, labels))
    
    db = float(davies_bouldin_score(X, labels))
    ch = float(calinski_harabasz_score(X, labels))
    return {
        'silhouette': sil,
        'davies_bouldin': db,
        'calinski_harabasz': ch
    }
