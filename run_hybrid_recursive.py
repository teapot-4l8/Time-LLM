"""
混合递归预测脚本
- 使用真实的OT值进行第一次预测
- 后续预测中，只更新OT列为预测值，其他42列保持真实值
- 虽然pred_len=6，但只使用第一步预测值进行递归
"""

import argparse
import torch
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt

from models import Autoformer, DLinear, TimeLLM
from data_provider.data_factory import data_provider
import random
import numpy as np
import os

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

from utils.tools import load_content

plt.switch_backend('agg')

parser = argparse.ArgumentParser(description='Time-LLM Hybrid Recursive Prediction for PPData - OT Only')

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# basic config
parser.add_argument('--task_name', type=str, required=True, default='short_term_forecast')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model_comment', type=str, required=True, default='none')
parser.add_argument('--model', type=str, required=True, default='GPT2')
parser.add_argument('--seed', type=int, default=2021, help='random seed')
parser.add_argument('--checkpoint_path', type=str, required=True, help='path to trained checkpoint file')

# data loader
parser.add_argument('--data', type=str, required=True, default='PPData')
parser.add_argument('--root_path', type=str, default='./dataset')
parser.add_argument('--data_path', type=str, default='PPData.csv')
parser.add_argument('--features', type=str, default='M')
parser.add_argument('--target', type=str, default='OT', help='target feature to predict and visualize')
parser.add_argument('--loader', type=str, default='modal')
parser.add_argument('--freq', type=str, default='s')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

# forecasting task
parser.add_argument('--seq_len', type=int, default=90)
parser.add_argument('--label_len', type=int, default=15)
parser.add_argument('--pred_len', type=int, default=6)
parser.add_argument('--seasonal_patterns', type=str, default='Monthly')

# model define
parser.add_argument('--enc_in', type=int, default=43)
parser.add_argument('--dec_in', type=int, default=43)
parser.add_argument('--c_out', type=int, default=43)
parser.add_argument('--d_model', type=int, default=16)
parser.add_argument('--n_heads', type=int, default=8)
parser.add_argument('--e_layers', type=int, default=2)
parser.add_argument('--d_layers', type=int, default=1)
parser.add_argument('--d_ff', type=int, default=32)
parser.add_argument('--moving_avg', type=int, default=25)
parser.add_argument('--factor', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--embed', type=str, default='timeF')
parser.add_argument('--activation', type=str, default='gelu')
parser.add_argument('--output_attention', action='store_true')
parser.add_argument('--patch_len', type=int, default=16)
parser.add_argument('--stride', type=int, default=8)
parser.add_argument('--prompt_domain', type=int, default=0)
parser.add_argument('--llm_model', type=str, default='LLAMA')
parser.add_argument('--llm_dim', type=int, default=4096)
parser.add_argument('--llm_layers', type=int, default=6)

# testing
parser.add_argument('--num_workers', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=1, help='必须为1以便进行递归预测')
parser.add_argument('--des', type=str, default='test')
parser.add_argument('--use_amp', action='store_true', default=False)
parser.add_argument('--percent', type=int, default=100)
parser.add_argument('--recursive_steps', type=int, default=200, help='递归预测的步数')
parser.add_argument('--use_gpu', action='store_true', default=True)
parser.add_argument('--gpu', type=int, default=0)

args = parser.parse_args()

# Update seed
random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Setup paths
folder_path = './test_results_hybrid_recursive/'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Set device
if args.use_gpu and torch.cuda.is_available():
    device = torch.device(f'cuda:{args.gpu}')
    print(f'Using GPU: {torch.cuda.get_device_name(args.gpu)}')
else:
    device = torch.device('cpu')
    print('Using CPU')

# Load test data
print('='*80)
print('LOADING TEST DATA')
print('='*80)
test_data, test_loader = data_provider(args, 'test')
print(f'Test samples: {len(test_data)}')
print(f'Number of features: {args.c_out}')
print(f'Target feature: {args.target}')

# Get the full dataset for accessing raw data
print('\nAccessing raw test data...')
# The test_data object should have the raw data
# We need to access it differently depending on the data loader implementation
# Let's get the full data array

# Build model
print('\n' + '='*80)
print('BUILDING MODEL')
print('='*80)
if args.model == 'Autoformer':
    model = Autoformer.Model(args).float()
elif args.model == 'DLinear':
    model = DLinear.Model(args).float()
else:
    model = TimeLLM.Model(args).float()

args.content = load_content(args)

# Load checkpoint
print('\n' + '='*80)
print('LOADING CHECKPOINT')
print('='*80)
print(f'Checkpoint path: {args.checkpoint_path}')
if not os.path.exists(args.checkpoint_path):
    print(f'ERROR: Checkpoint not found')
    exit(1)

checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint)
print('✓ Checkpoint loaded successfully')

# Move to device
model = model.to(device)
if args.use_gpu and torch.cuda.is_available():
    model = model.to(torch.bfloat16)
    print('✓ Model converted to bfloat16')

# Start hybrid recursive prediction
print('\n' + '='*80)
print('HYBRID RECURSIVE PREDICTION')
print('='*80)
print(f'Recursive steps: {args.recursive_steps}')
print(f'Prediction length: {args.pred_len} (but only use first step)')
print(f'Sequence length: {args.seq_len}')
print(f'Only updating OT column, keeping other 42 features as ground truth')
print('='*80)

model.eval()

# Get the first batch to start
first_batch = next(iter(test_loader))
batch_x, batch_y, batch_x_mark, batch_y_mark = first_batch

# We need access to the raw data beyond the first window
# Let's access the test_data directly
# Get the scaled data from test_data
if hasattr(test_data, 'data_x'):
    # For datasets that store data_x
    full_data_scaled = test_data.data_x  # Shape: (total_timesteps, num_features)
elif hasattr(test_data, 'data'):
    # For datasets that store 'data'
    full_data_scaled = test_data.data
else:
    print("ERROR: Cannot access raw data from test_data object")
    exit(1)

print(f'\nFull test data shape: {full_data_scaled.shape}')

# Also get the time marks if available
if hasattr(test_data, 'data_stamp'):
    full_time_marks = test_data.data_stamp
else:
    # Create dummy time marks if not available
    full_time_marks = np.zeros((full_data_scaled.shape[0], 4))  # Assuming 4 time features

print(f'Full time marks shape: {full_time_marks.shape}')

# OT column index (last column, index -1 or 42)
ot_index = -1
num_features = args.c_out

# Initialize storage for predictions
predictions_ot_scaled = []
ground_truth_ot_scaled = []

# Start from index 0
start_idx = 0

# We'll perform recursive_steps predictions
print(f'\nStarting recursive prediction from index {start_idx}...')

with torch.no_grad():
    for step in tqdm(range(args.recursive_steps), desc='Recursive Prediction'):
        # Current window: [start_idx : start_idx + seq_len]
        current_idx = start_idx + step

        # Check if we have enough data
        if current_idx + args.seq_len + args.pred_len > full_data_scaled.shape[0]:
            print(f'\nReached end of data at step {step}')
            break

        # Get input sequence [current_idx : current_idx + seq_len]
        seq_x = full_data_scaled[current_idx : current_idx + args.seq_len].copy()  # (seq_len, num_features)
        seq_x_mark = full_time_marks[current_idx : current_idx + args.seq_len].copy()  # (seq_len, time_features)

        # Get target sequence for decoder input and ground truth
        seq_y = full_data_scaled[current_idx + args.seq_len - args.label_len :
                                  current_idx + args.seq_len + args.pred_len].copy()  # (label_len + pred_len, num_features)
        seq_y_mark = full_time_marks[current_idx + args.seq_len - args.label_len :
                                      current_idx + args.seq_len + args.pred_len].copy()  # (label_len + pred_len, time_features)

        # Convert to tensors and add batch dimension
        batch_x = torch.from_numpy(seq_x).float().unsqueeze(0).to(device)  # (1, seq_len, num_features)
        batch_y = torch.from_numpy(seq_y).float().unsqueeze(0).to(device)  # (1, label_len + pred_len, num_features)
        batch_x_mark = torch.from_numpy(seq_x_mark).float().unsqueeze(0).to(device)  # (1, seq_len, time_features)
        batch_y_mark = torch.from_numpy(seq_y_mark).float().unsqueeze(0).to(device)  # (1, label_len + pred_len, time_features)

        # Create decoder input
        dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)

        # Model prediction
        if args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        # Extract predictions for pred_len steps
        f_dim = -1 if args.features == 'MS' else 0
        pred_full = outputs[:, -args.pred_len:, f_dim:]  # (1, pred_len, num_features)

        # Only use the FIRST step prediction (index 0)
        pred_step_0 = pred_full[:, 0, :]  # (1, num_features)

        # Extract OT prediction from the first step
        pred_ot_step_0 = pred_step_0[:, ot_index].cpu().numpy()  # (1,)

        # Get ground truth OT for the first predicted step
        ground_truth_idx = current_idx + args.seq_len  # The first step we're predicting
        true_ot_step_0 = full_data_scaled[ground_truth_idx, ot_index]

        # Store predictions and ground truth
        predictions_ot_scaled.append(pred_ot_step_0[0])
        ground_truth_ot_scaled.append(true_ot_step_0)

        # Update the data for next iteration
        # We update ONLY the OT column at position [ground_truth_idx]
        # with the predicted value for hybrid recursive prediction
        # Other columns remain as ground truth
        full_data_scaled[ground_truth_idx, ot_index] = pred_ot_step_0[0]

# Convert to arrays
predictions_ot_scaled = np.array(predictions_ot_scaled)
ground_truth_ot_scaled = np.array(ground_truth_ot_scaled)

print(f'\nHybrid recursive prediction completed!')
print(f'Generated {len(predictions_ot_scaled)} predictions')
print(f'Predictions shape: {predictions_ot_scaled.shape}')
print(f'Ground truth shape: {ground_truth_ot_scaled.shape}')

# Inverse transform to original scale
print('\n' + '='*80)
print('INVERSE TRANSFORMING TO ORIGINAL SCALE')
print('='*80)

# Reshape for inverse transform: need to match the format (samples, features)
# Since we only have OT, we need to create a full feature array with OT in the right position
# and fill other positions with zeros (or we can use the scaler for just OT)

# Create dummy full feature arrays
dummy_features = np.zeros((len(predictions_ot_scaled), num_features))
dummy_features[:, ot_index] = predictions_ot_scaled
predictions_original_full = test_data.inverse_transform(dummy_features)
predictions_ot_original = predictions_original_full[:, ot_index]

dummy_features_true = np.zeros((len(ground_truth_ot_scaled), num_features))
dummy_features_true[:, ot_index] = ground_truth_ot_scaled
ground_truth_original_full = test_data.inverse_transform(dummy_features_true)
ground_truth_ot_original = ground_truth_original_full[:, ot_index]

print(f'Original scale predictions shape: {predictions_ot_original.shape}')
print(f'Original scale ground truth shape: {ground_truth_ot_original.shape}')

# Calculate metrics
mse_scaled = np.mean((predictions_ot_scaled - ground_truth_ot_scaled) ** 2)
mae_scaled = np.mean(np.abs(predictions_ot_scaled - ground_truth_ot_scaled))
rmse_scaled = np.sqrt(mse_scaled)

mse_original = np.mean((predictions_ot_original - ground_truth_ot_original) ** 2)
mae_original = np.mean(np.abs(predictions_ot_original - ground_truth_ot_original))
rmse_original = np.sqrt(mse_original)

# Print results
print('\n' + '='*80)
print(f'HYBRID RECURSIVE PREDICTION RESULTS - {args.target} ONLY')
print('='*80)
print(f'\nScaled Data Metrics (Normalized):')
print(f'  MSE:  {mse_scaled:.7f}')
print(f'  MAE:  {mae_scaled:.7f}')
print(f'  RMSE: {rmse_scaled:.7f}')
print(f'\nOriginal Scale Metrics:')
print(f'  MSE:  {mse_original:.7f}')
print(f'  MAE:  {mae_original:.7f}')
print(f'  RMSE: {rmse_original:.7f}')
print('='*80)

# Save results
result_file = os.path.join(folder_path, f'{args.data}_{args.model_id}_hybrid_recursive_results.npz')
np.savez(result_file,
         predictions_scaled=predictions_ot_scaled,
         ground_truth_scaled=ground_truth_ot_scaled,
         predictions_original=predictions_ot_original,
         ground_truth_original=ground_truth_ot_original,
         mse_scaled=mse_scaled,
         mae_scaled=mae_scaled,
         rmse_scaled=rmse_scaled,
         mse_original=mse_original,
         mae_original=mae_original,
         rmse_original=rmse_original)

print(f'\n✓ Results saved to {result_file}')

# Generate plots
print('\n' + '='*80)
print(f'GENERATING PLOTS FOR {args.target}')
print('='*80)

# Plot 1: Full continuous prediction - Scaled
print('\n[1/4] Plotting full continuous prediction (scaled)...')
fig, ax = plt.subplots(1, 1, figsize=(20, 6))

time_axis = np.arange(len(predictions_ot_scaled))

ax.plot(time_axis, ground_truth_ot_scaled, label='Ground Truth (Scaled)',
        linewidth=2, color='#2E86AB', alpha=0.8)
ax.plot(time_axis, predictions_ot_scaled, label='Hybrid Recursive Prediction (Scaled)',
        linewidth=2, color='#A23B72', alpha=0.8, linestyle='--')
ax.fill_between(time_axis,
                 ground_truth_ot_scaled,
                 predictions_ot_scaled,
                 alpha=0.2, color='gray', label='Prediction Error')
ax.set_title(f'{args.target} Hybrid Recursive Prediction - Scaled Data ({len(predictions_ot_scaled)} steps)',
             fontsize=14, fontweight='bold', pad=10)
ax.set_xlabel('Recursive Step', fontsize=12)
ax.set_ylabel(f'{args.target} Value (Scaled)', fontsize=12)
ax.legend(fontsize=11, loc='best')
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
scaled_path = os.path.join(folder_path, f'{args.data}_{args.model_id}_hybrid_recursive_scaled.png')
plt.savefig(scaled_path, dpi=300, bbox_inches='tight')
print(f'  ✓ Saved: {scaled_path}')
plt.close()

# Plot 2: Full continuous prediction - Original scale
print('[2/4] Plotting full continuous prediction (original scale)...')
fig, ax = plt.subplots(1, 1, figsize=(20, 6))

ax.plot(time_axis, ground_truth_ot_original, label='Ground Truth (Original)',
        linewidth=2, color='#18A558', alpha=0.8)
ax.plot(time_axis, predictions_ot_original, label='Hybrid Recursive Prediction (Original)',
        linewidth=2, color='#F18F01', alpha=0.8, linestyle='--')
ax.fill_between(time_axis,
                 ground_truth_ot_original,
                 predictions_ot_original,
                 alpha=0.2, color='gray', label='Prediction Error')
ax.set_title(f'{args.target} Hybrid Recursive Prediction - Original Scale ({len(predictions_ot_original)} steps)',
             fontsize=14, fontweight='bold', pad=10)
ax.set_xlabel('Recursive Step', fontsize=12)
ax.set_ylabel(f'{args.target} Value (Original Scale)', fontsize=12)
ax.legend(fontsize=11, loc='best')
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
original_path = os.path.join(folder_path, f'{args.data}_{args.model_id}_hybrid_recursive_original.png')
plt.savefig(original_path, dpi=300, bbox_inches='tight')
print(f'  ✓ Saved: {original_path}')
plt.close()

# Plot 3: Side-by-side comparison
print('[3/4] Plotting side-by-side comparison...')
fig, axes = plt.subplots(2, 1, figsize=(20, 10))

# Scaled
axes[0].plot(time_axis, ground_truth_ot_scaled, label='Ground Truth',
            linewidth=2, color='#2E86AB', alpha=0.8)
axes[0].plot(time_axis, predictions_ot_scaled, label='Prediction',
            linewidth=2, color='#A23B72', alpha=0.8, linestyle='--')
axes[0].fill_between(time_axis,
                     ground_truth_ot_scaled,
                     predictions_ot_scaled,
                     alpha=0.2, color='gray', label='Error')
axes[0].set_title(f'{args.target} Hybrid Recursive Prediction - Scaled/Normalized Data',
                 fontsize=13, fontweight='bold', pad=10)
axes[0].set_xlabel('Recursive Step', fontsize=11)
axes[0].set_ylabel(f'{args.target} Value (Scaled)', fontsize=11)
axes[0].legend(fontsize=11, loc='best')
axes[0].grid(True, alpha=0.3, linestyle='--')

# Original
axes[1].plot(time_axis, ground_truth_ot_original, label='Ground Truth',
            linewidth=2, color='#18A558', alpha=0.8)
axes[1].plot(time_axis, predictions_ot_original, label='Prediction',
            linewidth=2, color='#F18F01', alpha=0.8, linestyle='--')
axes[1].fill_between(time_axis,
                     ground_truth_ot_original,
                     predictions_ot_original,
                     alpha=0.2, color='gray', label='Error')
axes[1].set_title(f'{args.target} Hybrid Recursive Prediction - Original Scale',
                 fontsize=13, fontweight='bold', pad=10)
axes[1].set_xlabel('Recursive Step', fontsize=11)
axes[1].set_ylabel(f'{args.target} Value (Original Scale)', fontsize=11)
axes[1].legend(fontsize=11, loc='best')
axes[1].grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
comparison_path = os.path.join(folder_path, f'{args.data}_{args.model_id}_hybrid_recursive_comparison.png')
plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
print(f'  ✓ Saved: {comparison_path}')
plt.close()

# Plot 4: Error analysis
print('[4/4] Plotting error analysis...')
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Scaled error distribution
errors_scaled = predictions_ot_scaled - ground_truth_ot_scaled
axes[0, 0].hist(errors_scaled, bins=50, edgecolor='black', alpha=0.7, color='#2E86AB')
axes[0, 0].set_title(f'{args.target} Prediction Error Distribution (Scaled)',
                     fontsize=12, fontweight='bold', pad=10)
axes[0, 0].set_xlabel('Prediction Error', fontsize=10)
axes[0, 0].set_ylabel('Frequency', fontsize=10)
axes[0, 0].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error', alpha=0.8)
axes[0, 0].axvline(x=np.mean(errors_scaled), color='green', linestyle='-.', linewidth=2,
                   label=f'Mean: {np.mean(errors_scaled):.4f}', alpha=0.8)
axes[0, 0].legend(fontsize=9)
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Original error distribution
errors_original = predictions_ot_original - ground_truth_ot_original
axes[0, 1].hist(errors_original, bins=50, edgecolor='black', alpha=0.7, color='#F18F01')
axes[0, 1].set_title(f'{args.target} Prediction Error Distribution (Original)',
                     fontsize=12, fontweight='bold', pad=10)
axes[0, 1].set_xlabel('Prediction Error', fontsize=10)
axes[0, 1].set_ylabel('Frequency', fontsize=10)
axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error', alpha=0.8)
axes[0, 1].axvline(x=np.mean(errors_original), color='green', linestyle='-.', linewidth=2,
                   label=f'Mean: {np.mean(errors_original):.4f}', alpha=0.8)
axes[0, 1].legend(fontsize=9)
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Scaled error over time
axes[1, 0].plot(time_axis, errors_scaled, linewidth=1.5, color='#2E86AB', alpha=0.7)
axes[1, 0].axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.8)
axes[1, 0].set_title('Error Over Time (Scaled)',
                     fontsize=12, fontweight='bold', pad=10)
axes[1, 0].set_xlabel('Recursive Step', fontsize=10)
axes[1, 0].set_ylabel('Prediction Error', fontsize=10)
axes[1, 0].grid(True, alpha=0.3)

# Original error over time
axes[1, 1].plot(time_axis, errors_original, linewidth=1.5, color='#F18F01', alpha=0.7)
axes[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.8)
axes[1, 1].set_title('Error Over Time (Original)',
                     fontsize=12, fontweight='bold', pad=10)
axes[1, 1].set_xlabel('Recursive Step', fontsize=10)
axes[1, 1].set_ylabel('Prediction Error', fontsize=10)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
error_path = os.path.join(folder_path, f'{args.data}_{args.model_id}_hybrid_recursive_error_analysis.png')
plt.savefig(error_path, dpi=300, bbox_inches='tight')
print(f'  ✓ Saved: {error_path}')
plt.close()

print('\n' + '='*80)
print(f'✓ All plots saved to: {folder_path}')
print('='*80)

print(f'\n✓ Hybrid recursive prediction completed for {args.target}!')
print(f'\nSummary:')
print(f'  - Performed {len(predictions_ot_scaled)} recursive predictions')
print(f'  - Only OT column was updated with predictions')
print(f'  - Other 42 features remained as ground truth')
print(f'  - Used only first step (index 0) of each pred_len={args.pred_len} prediction')
