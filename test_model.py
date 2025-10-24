"""
Standalone Test Script for Time-LLM
Loads trained checkpoint, runs inference, and plots results
"""

import argparse
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from accelerate import Accelerator
from torch import nn
from tqdm import tqdm

from models import Autoformer, DLinear, TimeLLM
from data_provider.data_factory import data_provider
from utils.tools import load_content

# ==================== CONFIGURATION ====================
# Set your configuration here
CONFIG = {
    # Checkpoint path - CHANGE THIS TO YOUR ACTUAL CHECKPOINT PATH
    'checkpoint_path': './checkpoints/long_term_forecast_PPData_96_96_TimeLLM_PPData_ftM_sl96_ll48_pl96_dm16_nh8_el2_dl1_df32_fc3_ebtimeF_test_0-TimeLLM-PPData/checkpoint',

    # Data configuration
    'data': 'PPData',
    'root_path': './dataset/',
    'data_path': 'PPData.csv',
    'target': 'OT',
    'features': 'M',  # M: multivariate, S: univariate, MS: multivariate predict univariate

    # Model configuration
    'model': 'TimeLLM',
    'model_id': 'PPData_96_96',
    'model_comment': 'TimeLLM-PPData',

    # Sequence configuration
    'seq_len': 96,
    'label_len': 48,
    'pred_len': 96,

    # Model architecture
    'enc_in': 43,
    'dec_in': 43,
    'c_out': 43,
    'd_model': 16,
    'd_ff': 32,
    'n_heads': 8,
    'e_layers': 2,
    'd_layers': 1,
    'factor': 3,
    'dropout': 0.1,

    # LLM configuration
    'llm_model': 'LLAMA',
    'llm_dim': 4096,
    'llm_layers': 32,

    # Other settings
    'embed': 'timeF',
    'freq': 'h',
    'activation': 'gelu',
    'output_attention': False,
    'patch_len': 16,
    'stride': 8,
    'batch_size': 16,
    'num_workers': 4,
    'use_amp': False,
    'percent': 100,
    'des': 'test',

    # Output settings
    'save_results': True,
    'results_dir': './test_results/',
    'plot_samples': 5,  # Number of samples to plot
}

# ==================== MAIN FUNCTION ====================

def main():
    # Convert config to argparse-like object
    class Args:
        pass

    args = Args()
    for key, value in CONFIG.items():
        setattr(args, key, value)

    # Setup
    print('='*80)
    print('Time-LLM Testing Script')
    print('='*80)

    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint_path):
        print(f"\n❌ ERROR: Checkpoint not found at: {args.checkpoint_path}")
        print("\nAvailable checkpoints:")
        if os.path.exists('./checkpoints'):
            for root, dirs, files in os.walk('./checkpoints'):
                for file in files:
                    if file == 'checkpoint':
                        print(f"  - {os.path.join(root, file)}")
        else:
            print("  No checkpoints directory found!")
        return

    print(f"\n✓ Checkpoint found: {args.checkpoint_path}")

    # Setup accelerator (for single GPU or CPU)
    accelerator = Accelerator()
    device = accelerator.device

    # Load test data
    print('\n[1/5] Loading test dataset...')
    test_data, test_loader = data_provider(args, 'test')
    print(f'  Test samples: {len(test_data)}')

    # Build model
    print('\n[2/5] Building model...')
    if args.model == 'Autoformer':
        model = Autoformer.Model(args).float()
    elif args.model == 'DLinear':
        model = DLinear.Model(args).float()
    else:
        model = TimeLLM.Model(args).float()

    # Load content for TimeLLM
    args.content = load_content(args)

    # Load checkpoint
    print(f'\n[3/5] Loading checkpoint...')
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    print('  ✓ Model loaded successfully')

    # Run inference
    print('\n[4/5] Running inference on test set...')
    preds_scaled = []
    trues_scaled = []

    criterion = nn.MSELoss()
    mae_metric = nn.L1Loss()

    total_loss = []
    total_mae_loss = []

    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(test_loader, desc='  Testing')):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            # Decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)

            # Forward pass
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.output_attention:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                if args.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:]

            pred = outputs.detach().cpu().numpy()
            true = batch_y.detach().cpu().numpy()

            preds_scaled.append(pred)
            trues_scaled.append(true)

            # Calculate loss
            loss = criterion(outputs, batch_y)
            mae_loss = mae_metric(outputs, batch_y)

            total_loss.append(loss.item())
            total_mae_loss.append(mae_loss.item())

    # Concatenate results
    preds_scaled = np.concatenate(preds_scaled, axis=0)
    trues_scaled = np.concatenate(trues_scaled, axis=0)

    # Inverse transform to get original scale
    print('\n[5/5] Inverse transforming to original scale...')
    preds_original = test_data.inverse_transform(preds_scaled.reshape(-1, preds_scaled.shape[-1])).reshape(preds_scaled.shape)
    trues_original = test_data.inverse_transform(trues_scaled.reshape(-1, trues_scaled.shape[-1])).reshape(trues_scaled.shape)

    # Calculate metrics on scaled data
    mse_scaled = np.mean((preds_scaled - trues_scaled) ** 2)
    mae_scaled = np.mean(np.abs(preds_scaled - trues_scaled))
    rmse_scaled = np.sqrt(mse_scaled)

    # Calculate metrics on original data
    mse_original = np.mean((preds_original - trues_original) ** 2)
    mae_original = np.mean(np.abs(preds_original - trues_original))
    rmse_original = np.sqrt(mse_original)

    # Print results
    print('\n' + '='*80)
    print('TEST RESULTS')
    print('='*80)
    print(f'\nScaled Data Metrics (Normalized):')
    print(f'  MSE:  {mse_scaled:.7f}')
    print(f'  MAE:  {mae_scaled:.7f}')
    print(f'  RMSE: {rmse_scaled:.7f}')
    print(f'\nOriginal Scale Metrics:')
    print(f'  MSE:  {mse_original:.7f}')
    print(f'  MAE:  {mae_original:.7f}')
    print(f'  RMSE: {rmse_original:.7f}')
    print(f'\nData shapes:')
    print(f'  Predictions: {preds_scaled.shape}')
    print(f'  Ground truth: {trues_scaled.shape}')
    print('='*80)

    # Get OT column index (last column based on your CSV structure)
    ot_index = -1  # OT is the last column

    # Plot results
    print('\n[6/6] Generating plots...')

    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    num_samples = min(args.plot_samples, len(preds_scaled))

    # Plot 1: Scaled data (multiple samples)
    fig, axes = plt.subplots(num_samples, 1, figsize=(15, 4*num_samples))
    if num_samples == 1:
        axes = [axes]

    for i in range(num_samples):
        axes[i].plot(trues_scaled[i, :, ot_index], label='Actual (Scaled)', linewidth=2, marker='o')
        axes[i].plot(preds_scaled[i, :, ot_index], label='Predicted (Scaled)', linewidth=2, marker='x')
        axes[i].set_title(f'Sample {i+1} - OT Prediction (Scaled/Normalized Data)')
        axes[i].set_xlabel('Time Step')
        axes[i].set_ylabel('OT Value (Scaled)')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    scaled_path = os.path.join(args.results_dir, f'{args.data}_scaled_predictions.png')
    plt.savefig(scaled_path, dpi=300, bbox_inches='tight')
    print(f'  ✓ Saved scaled plot: {scaled_path}')
    plt.close()

    # Plot 2: Original scale data (multiple samples)
    fig, axes = plt.subplots(num_samples, 1, figsize=(15, 4*num_samples))
    if num_samples == 1:
        axes = [axes]

    for i in range(num_samples):
        axes[i].plot(trues_original[i, :, ot_index], label='Actual (Original)', linewidth=2, marker='o', color='green')
        axes[i].plot(preds_original[i, :, ot_index], label='Predicted (Original)', linewidth=2, marker='x', color='red')
        axes[i].set_title(f'Sample {i+1} - OT Prediction (Original Scale)')
        axes[i].set_xlabel('Time Step')
        axes[i].set_ylabel('OT Value (Original Scale)')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    original_path = os.path.join(args.results_dir, f'{args.data}_original_predictions.png')
    plt.savefig(original_path, dpi=300, bbox_inches='tight')
    print(f'  ✓ Saved original scale plot: {original_path}')
    plt.close()

    # Plot 3: Side-by-side comparison for first sample
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Scaled
    axes[0].plot(trues_scaled[0, :, ot_index], label='Actual', linewidth=2, marker='o')
    axes[0].plot(preds_scaled[0, :, ot_index], label='Predicted', linewidth=2, marker='x')
    axes[0].set_title('OT Prediction - Scaled/Normalized Data')
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('OT Value (Scaled)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Original
    axes[1].plot(trues_original[0, :, ot_index], label='Actual', linewidth=2, marker='o', color='green')
    axes[1].plot(preds_original[0, :, ot_index], label='Predicted', linewidth=2, marker='x', color='red')
    axes[1].set_title('OT Prediction - Original Scale')
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('OT Value (Original Scale)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    comparison_path = os.path.join(args.results_dir, f'{args.data}_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f'  ✓ Saved comparison plot: {comparison_path}')
    plt.close()

    # Plot 4: Error distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scaled errors
    errors_scaled = preds_scaled[:, :, ot_index].flatten() - trues_scaled[:, :, ot_index].flatten()
    axes[0].hist(errors_scaled, bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_title('Prediction Error Distribution (Scaled Data)')
    axes[0].set_xlabel('Error')
    axes[0].set_ylabel('Frequency')
    axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Original errors
    errors_original = preds_original[:, :, ot_index].flatten() - trues_original[:, :, ot_index].flatten()
    axes[1].hist(errors_original, bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[1].set_title('Prediction Error Distribution (Original Scale)')
    axes[1].set_xlabel('Error')
    axes[1].set_ylabel('Frequency')
    axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    error_path = os.path.join(args.results_dir, f'{args.data}_error_distribution.png')
    plt.savefig(error_path, dpi=300, bbox_inches='tight')
    print(f'  ✓ Saved error distribution plot: {error_path}')
    plt.close()

    # Save numerical results
    if args.save_results:
        result_file = os.path.join(args.results_dir, f'{args.data}_{args.model_id}_results.npz')
        np.savez(result_file,
                 predictions_scaled=preds_scaled,
                 ground_truth_scaled=trues_scaled,
                 predictions_original=preds_original,
                 ground_truth_original=trues_original,
                 mse_scaled=mse_scaled,
                 mae_scaled=mae_scaled,
                 rmse_scaled=rmse_scaled,
                 mse_original=mse_original,
                 mae_original=mae_original,
                 rmse_original=rmse_original)
        print(f'\n  ✓ Results saved to: {result_file}')

    print('\n' + '='*80)
    print('✓ Testing completed successfully!')
    print(f'  All plots saved to: {args.results_dir}')
    print('='*80)


if __name__ == '__main__':
    main()
