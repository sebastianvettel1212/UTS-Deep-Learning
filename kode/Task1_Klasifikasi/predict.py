"""
Inference script for Task 1 Multimodal Defect Classification.
Generates prediksi_klasifikasi.csv with columns: id_unit, kelas
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.data_loader import load_task1_test, IDX_TO_CLASS, CLASS_NAMES
from Task1_Klasifikasi.model import MultimodalDefectClassifier
from Task1_Klasifikasi.train import FastMultimodalDataset

def generate_predictions(model_path='models/task1_best_model.pt', output_file='prediksi_klasifikasi.csv', batch_size=256):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Generating Task 1 predictions using {device} from {model_path}...")
    
    images, sensors, sensor_valid, meta_df = load_task1_test()
    test_dataset = FastMultimodalDataset(images, sensors, sensor_valid, labels=None, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    model = MultimodalDefectClassifier(num_classes=len(CLASS_NAMES), embed_dim=128).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Warning: {model_path} not found. Ensure the model is trained before running.")
    model.eval()
    
    predictions = []
    with torch.no_grad():
        for img, sen, val in test_loader:
            img, sen, val = img.to(device), sen.to(device), val.to(device)
            logits = model(img, sen, val)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(preds)
            
    pred_classes = [IDX_TO_CLASS[idx] for idx in predictions]
    
    # Format output DataFrame exactly as required
    res_df = pd.DataFrame({
        'id_unit': meta_df['id_unit'],
        'kelas': pred_classes
    })
    
    # Save to both paths for safety
    res_df.to_csv(output_file, index=False)
    os.makedirs('prediksi', exist_ok=True)
    res_df.to_csv(os.path.join('prediksi', 'prediksi_klasifikasi.csv'), index=False)
    
    print(f"Saved {len(res_df)} predictions to {output_file} and prediksi/prediksi_klasifikasi.csv")
    print(f"Class prediction distribution:\n{res_df['kelas'].value_counts()}")
    return res_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/task1_best_model.pt')
    parser.add_argument('--output_file', type=str, default='prediksi_klasifikasi.csv')
    args = parser.parse_args()
    
    generate_predictions(model_path=args.model_path, output_file=args.output_file)
