"""
Inference script for Task 2 Time-Series Remaining Useful Life (RUL) Regression.
Computes RUL predictions at the final recorded cycle of each test machine unit.
Applies MC Dropout uncertainty calibration to penalize over-optimistic predictions.
Generates prediksi_rul.csv with columns: id_unit, rul
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle
import argparse
import pandas as pd
import numpy as np
import torch

from utils.data_loader import load_task2_data
from Task2_RUL.preprocess import clean_and_impute_timeseries, generate_sequences_test
from Task2_RUL.model import BiLSTMAttentionRUL

def generate_predictions(model_path='models/task2_best_model.pt', 
                         scaler_path='models/task2_scaler.pkl',
                         output_file='prediksi_rul.csv',
                         use_mc_dropout=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Generating Task 2 RUL predictions using {device}...")
    
    _, df_test = load_task2_data()
    print("Cleaning test time-series data...")
    df_test_clean = clean_and_impute_timeseries(df_test)
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
        
    X_test, test_units = generate_sequences_test(df_test_clean, scaler, window_size=30)
    print(f"Extracted {len(X_test)} test sequences for {len(test_units)} unique units.")
    
    model = BiLSTMAttentionRUL(input_dim=24, hidden_dim=64, num_layers=2, dropout_p=0.2).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Warning: {model_path} not found. Ensure model is trained.")
        
    X_test_tensor = torch.from_numpy(X_test).to(device)
    
    if use_mc_dropout:
        print("Running Monte Carlo Dropout (30 forward passes) for uncertainty estimation...")
        mean_preds, std_preds = model.predict_with_uncertainty(X_test_tensor, num_mc_samples=30)
        mean_preds = mean_preds.cpu().numpy()
        std_preds = std_preds.cpu().numpy()
        
        # Asymmetric bias correction:
        # Since over-prediction (d >= 0) is penalized with exp(d/10) vs exp(-d/13),
        # shifting the prediction downward slightly proportional to uncertainty reduces expected loss!
        calibrated_preds = mean_preds - 0.20 * std_preds
        final_preds = np.clip(calibrated_preds, 0.0, 150.0)
        print(f"Uncertainty stats: Mean std={np.mean(std_preds):.2f}, Max std={np.max(std_preds):.2f}")
    else:
        model.eval()
        with torch.no_grad():
            preds = model(X_test_tensor).squeeze(-1).cpu().numpy()
            final_preds = np.clip(preds, 0.0, 150.0)
            
    res_df = pd.DataFrame({
        'id_unit': test_units,
        'rul': np.round(final_preds, 2)
    })
    
    # Save to both paths
    res_df.to_csv(output_file, index=False)
    os.makedirs('prediksi', exist_ok=True)
    res_df.to_csv(os.path.join('prediksi', 'prediksi_rul.csv'), index=False)
    
    print(f"Saved {len(res_df)} RUL predictions to {output_file} and prediksi/prediksi_rul.csv")
    print(f"RUL Summary Statistics:\n{res_df['rul'].describe()}")
    return res_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/task2_best_model.pt')
    parser.add_argument('--scaler_path', type=str, default='models/task2_scaler.pkl')
    parser.add_argument('--output_file', type=str, default='prediksi_rul.csv')
    args = parser.parse_args()
    
    generate_predictions(model_path=args.model_path, scaler_path=args.scaler_path, output_file=args.output_file)
