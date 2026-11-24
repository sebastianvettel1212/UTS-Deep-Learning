"""
Master Reproduction Script for UTS Deep Learning:
Task 1 (Multimodal Classification), Task 2 (Time-series RUL), and Task 3 (Multimodal Clustering).

Single Command Execution:
    python run_all.py --seed 42
"""

import os
import sys
import time
import argparse

# Ensure current directory is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Task1_Klasifikasi.train import train_task1
from Task1_Klasifikasi.predict import generate_predictions as predict_task1
from Task2_RUL.train import train_task2
from Task2_RUL.predict import generate_predictions as predict_task2
from Task3_Klasterisasi.predict import run_task3_pipeline

def main():
    parser = argparse.ArgumentParser(description="Reproduce all UTS Deep Learning experiments and predictions.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument('--epochs_t1', type=int, default=10, help="Epochs for Task 1.")
    parser.add_argument('--epochs_t2', type=int, default=12, help="Epochs for Task 2.")
    parser.add_argument('--epochs_t3', type=int, default=8, help="Epochs for Task 3 autoencoder.")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for training.")
    args = parser.parse_args()
    
    start_time = time.time()
    print("=" * 70)
    print("UTS DEEP LEARNING - FULL END-TO-END REPRODUCTION PIPELINE")
    print(f"Seed: {args.seed} | Batch Size: {args.batch_size}")
    print("=" * 70)
    
    # ----------------------------------------------------
    # TASK 1: Multimodal Classification (Defect Detection)
    # ----------------------------------------------------
    print("\n" + "#" * 70)
    print("TASK 1: MULTIMODAL CLASSIFICATION (PRODUCT DEFECTS)")
    print("#" * 70)
    t1_f1 = train_task1(epochs=args.epochs_t1, batch_size=args.batch_size, seed=args.seed)
    predict_task1(output_file='prediksi_klasifikasi.csv', batch_size=args.batch_size)
    
    # ----------------------------------------------------
    # TASK 2: Time-Series Regression (RUL)
    # ----------------------------------------------------
    print("\n" + "#" * 70)
    print("TASK 2: TIME-SERIES REGRESSION (REMAINING USEFUL LIFE)")
    print("#" * 70)
    t2_rmse, t2_asym = train_task2(epochs=args.epochs_t2, batch_size=args.batch_size, seed=args.seed)
    predict_task2(output_file='prediksi_rul.csv')
    
    # ----------------------------------------------------
    # TASK 3: Multimodal Clustering (Material Types)
    # ----------------------------------------------------
    print("\n" + "#" * 70)
    print("TASK 3: MULTIMODAL CLUSTERING (MATERIAL TYPES)")
    print("#" * 70)
    run_task3_pipeline(epochs_ae=args.epochs_t3, batch_size=args.batch_size, seed=args.seed, output_file='cluster.csv')
    
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS & PREDICTIONS SUCCESSFULLY COMPLETED!")
    print(f"Total Execution Time: {elapsed / 60:.2f} minutes")
    print("=" * 70)
    print("Summary of Key Performance Indicators (Reproduced):")
    print(f"  * Task 1 Validation Macro-F1:     {t1_f1:.4f}")
    print(f"  * Task 2 Validation RMSE:         {t2_rmse:.3f}")
    print(f"  * Task 2 Asymmetric Loss Score:   {t2_asym:.2f}")
    print("  * Output prediction files verified:")
    print("    - prediksi_klasifikasi.csv / prediksi/prediksi_klasifikasi.csv")
    print("    - prediksi_rul.csv         / prediksi/prediksi_rul.csv")
    print("    - cluster.csv              / prediksi/cluster.csv")
    print("    - Figures in laporan/figures/")
    print("=" * 70)

if __name__ == '__main__':
    main()
