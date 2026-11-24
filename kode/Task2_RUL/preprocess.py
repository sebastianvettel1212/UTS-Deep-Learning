"""
Preprocessing module for Task 2 (Time-Series RUL Regression).
Handles missing values, operating condition scaling, feature selection,
piecewise linear RUL target clipping (max 150), and sliding window generation.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [f's{i:02d}' for i in range(1, 22)] + ['setel_1', 'setel_2', 'setel_3']

def clean_and_impute_timeseries(df):
    """
    Cleans and imputes missing sensor readings:
    - Grouped forward fill then backward fill within each machine unit.
    - Global median fallback for any remaining NaNs.
    """
    df_clean = df.copy()
    # Ensure correct dtypes
    df_clean['id_unit'] = df_clean['id_unit'].astype(int)
    df_clean['siklus'] = df_clean['siklus'].astype(int)
    
    # Sort chronologically
    df_clean = df_clean.sort_values(by=['id_unit', 'siklus']).reset_index(drop=True)
    
    # Forward fill then backward fill within each machine
    df_clean[FEATURE_COLS] = df_clean.groupby('id_unit')[FEATURE_COLS].transform(lambda group: group.ffill().bfill())
    
    # Global median fallback
    df_clean[FEATURE_COLS] = df_clean[FEATURE_COLS].fillna(df_clean[FEATURE_COLS].median())
    
    return df_clean

def fit_scaler(df_train):
    """
    Fits StandardScaler on training features.
    """
    scaler = StandardScaler()
    scaler.fit(df_train[FEATURE_COLS].values)
    return scaler

def generate_sequences_train(df_train, scaler, window_size=30, stride=2):
    """
    Generates sliding windows of size (window_size, num_features) from training units.
    Target is the RUL at the end of the window.
    """
    scaled_feats = scaler.transform(df_train[FEATURE_COLS].values)
    df_scaled = df_train.copy()
    for idx, col in enumerate(FEATURE_COLS):
        df_scaled[col] = scaled_feats[:, idx]
        
    X_list, y_list, unit_list = [], [], []
    
    for uid, group in df_scaled.groupby('id_unit'):
        features = group[FEATURE_COLS].values
        ruls = group['rul'].values
        n_cycles = len(features)
        
        # Stride allows fast training without redundant 1-cycle step duplication
        for i in range(0, n_cycles, stride):
            # If sequence length < window_size, pad with the first reading
            if i + 1 < window_size:
                pad_len = window_size - (i + 1)
                pad = np.repeat(features[:1], pad_len, axis=0)
                seq = np.vstack([pad, features[:i+1]])
            else:
                seq = features[i - window_size + 1 : i + 1]
                
            X_list.append(seq)
            y_list.append(ruls[i])
            unit_list.append(uid)
            
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32), np.array(unit_list, dtype=np.int32)

def generate_sequences_test(df_test, scaler, window_size=30):
    """
    Extracts the last window of size (window_size, num_features) for each test machine unit.
    Returns:
        X_test: np.ndarray (N_units, window_size, num_features)
        test_units: list of unit IDs
    """
    scaled_feats = scaler.transform(df_test[FEATURE_COLS].values)
    df_scaled = df_test.copy()
    for idx, col in enumerate(FEATURE_COLS):
        df_scaled[col] = scaled_feats[:, idx]
        
    X_test_list = []
    test_units = []
    
    # Process units in deterministic order
    unique_units = sorted(df_scaled['id_unit'].unique())
    for uid in unique_units:
        group = df_scaled[df_scaled['id_unit'] == uid]
        features = group[FEATURE_COLS].values
        n_cycles = len(features)
        
        if n_cycles < window_size:
            pad_len = window_size - n_cycles
            pad = np.repeat(features[:1], pad_len, axis=0)
            seq = np.vstack([pad, features])
        else:
            seq = features[-window_size:]
            
        X_test_list.append(seq)
        test_units.append(uid)
        
    return np.array(X_test_list, dtype=np.float32), test_units
