"""
Data loaders and preprocessing utilities for UTS Deep Learning.
Handles Task 1 (Multimodal Classification), Task 2 (Time-series RUL), and Task 3 (Multimodal Clustering).
"""

import os
import gzip
import numpy as np
import pandas as pd

CLASS_NAMES = ['normal', 'gores', 'noda', 'retak', 'gelembung', 'las_dingin', 'kontaminasi']
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}

def load_task1_train(data_dir='data'):
    """
    Loads Task 1 training dataset.
    Returns:
        images: np.ndarray (N, 64, 64), uint8
        sensors: np.ndarray (N, 3, 128), float32 with NaNs handled
        sensor_valid: np.ndarray (N, 3), float32 mask (1 if modality present, 0 if missing)
        labels: np.ndarray (N,), int64
        meta_df: pd.DataFrame
    """
    npz_path = os.path.join(data_dir, 'klasifikasi_train.npz')
    meta_path = os.path.join(data_dir, 'klasifikasi_train_meta.csv')
    
    meta_df = pd.read_csv(meta_path)
    data = np.load(npz_path)
    images = data['gambar']
    sensors = data['sensor'].astype(np.float32)
    
    # Process sensor missingness
    # Missing sensor mask per channel: 1 if not all NaN, 0 if completely missing
    sensor_valid = (~np.isnan(sensors).all(axis=-1)).astype(np.float32) # (N, 3)
    
    # Replace NaN with 0 for neural network consumption
    sensors = np.nan_to_num(sensors, nan=0.0)
    
    # Extract labels
    labels = np.array([CLASS_TO_IDX[c] for c in meta_df['kelas']], dtype=np.int64)
    
    return images, sensors, sensor_valid, labels, meta_df

def load_task1_test(data_dir='data'):
    """
    Loads Task 1 test dataset.
    Returns:
        images: np.ndarray (N, 64, 64), uint8
        sensors: np.ndarray (N, 3, 128), float32 with NaNs handled
        sensor_valid: np.ndarray (N, 3), float32 mask
        meta_df: pd.DataFrame
    """
    npz_path = os.path.join(data_dir, 'klasifikasi_test.npz')
    meta_path = os.path.join(data_dir, 'klasifikasi_test_meta.csv')
    
    meta_df = pd.read_csv(meta_path)
    data = np.load(npz_path)
    images = data['gambar']
    sensors = data['sensor'].astype(np.float32)
    
    sensor_valid = (~np.isnan(sensors).all(axis=-1)).astype(np.float32)
    sensors = np.nan_to_num(sensors, nan=0.0)
    
    return images, sensors, sensor_valid, meta_df

def load_task2_data(data_dir='data'):
    """
    Loads Task 2 train and test time-series datasets.
    Computes true RUL targets capped at 150 for train data.
    """
    train_path = os.path.join(data_dir, 'rul_train.csv.gz')
    test_path = os.path.join(data_dir, 'rul_test.csv.gz')
    
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    # Compute max cycles per unit in train to calculate RUL
    max_cycles = df_train.groupby('id_unit')['siklus'].max().rename('max_siklus')
    df_train = df_train.merge(max_cycles, on='id_unit', how='left')
    df_train['rul_true'] = df_train['max_siklus'] - df_train['siklus']
    df_train['rul'] = np.clip(df_train['rul_true'], 0, 150)
    df_train.drop(columns=['max_siklus', 'rul_true'], inplace=True)
    
    return df_train, df_test

def load_task3_data(data_dir='data'):
    """
    Loads Task 3 multimodal clustering dataset.
    Returns:
        images: np.ndarray (36000, 64, 64), uint8
        spectra: np.ndarray (36000, 128), float32
        meta_df: pd.DataFrame with id_sampel and mikroskop
    """
    npz_path = os.path.join(data_dir, 'clustering.npz')
    meta_path = os.path.join(data_dir, 'clustering_meta.csv')
    
    meta_df = pd.read_csv(meta_path)
    data = np.load(npz_path)
    images = data['gambar']
    spectra = data['spektrum'].astype(np.float32)
    
    return images, spectra, meta_df
