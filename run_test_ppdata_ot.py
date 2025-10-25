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

parser = argparse.ArgumentParser(description='Time-LLM Testing for PPData - OT Only')

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
parser.add_argument('--pred_len', type=int, default=30)
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
parser.add_argument('--batch_size', type=int, default=24)
parser.add_argument('--des', type=str, default='test')
parser.add_argument('--use_amp', action='store_true', default=False)
parser.add_argument('--percent', type=int, default=100)
parser.add_argument('--plot_samples', type=int, default=10, help='number of samples to plot')
parser.add_argument('--continuous_steps', type=int, default=100, help='number of continuous time steps to plot')
parser.add_argument('--use_gpu', action='store_true', default=True)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--load_results', type=str, default=None)
parser.add_argument('--save_results', action='store_true', default=True)

args = parser.parse_args()

# Update seed
random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Setup paths
folder_path = './test_results/'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

result_file = os.path.join(folder_path, f'{args.data}_{args.model_id}_inference.npz')

# Check if loading existing results
if args.load_results and os.path.exists(args.load_results):
    print('='*80)
    print('LOADING SAVED TEST RESULTS')
    print('='*80)
    print(f'Loading from: {args.load_results}')

    loaded = np.load(args.load_results)
    preds_scaled = loaded['predictions_scaled']
    trues_scaled = loaded['ground_truth_scaled']

    print(f'✓ Loaded predictions shape: {preds_scaled.shape}')
    print(f'✓ Loaded ground truth shape: {trues_scaled.shape}')

    print('\nLoading test data for inverse transform...')
    test_data, _ = data_provider(args, 'test')

    skip_inference = True
else:
    skip_inference = False

# Set device
if args.use_gpu and torch.cuda.is_available():
    device = torch.device(f'cuda:{args.gpu}')
    print(f'Using GPU: {torch.cuda.get_device_name(args.gpu)}')
else:
    device = torch.device('cpu')
    print('Using CPU')

if not skip_inference:
    # Load test data
    print('='*80)
    print('LOADING TEST DATA')
    print('='*80)
    test_data, test_loader = data_provider(args, 'test')
    print(f'Test samples: {len(test_data)}')
    print(f'Number of features: {args.c_out}')
    print(f'Target feature: {args.target}')

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

    # Test
    print('\n' + '='*80)
    print('RUNNING INFERENCE')
    print('='*80)
    model.eval()

    preds = []
    trues = []

    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(test_loader), desc='Testing', total=len(test_loader)):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)

            if args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)

            pred = outputs.detach().cpu().numpy()
            true = batch_y.detach().cpu().numpy()

            preds.append(pred)
            trues.append(true)

    preds_scaled = np.concatenate(preds, axis=0)
    trues_scaled = np.concatenate(trues, axis=0)

    print(f'\nPredictions shape: {preds_scaled.shape}')
    print(f'Ground truth shape: {trues_scaled.shape}')

    if args.save_results:
        print(f'\n✓ Saving raw inference results to {result_file}')
        np.savez(result_file,
                 predictions_scaled=preds_scaled,
                 ground_truth_scaled=trues_scaled)
else:
    print('\n✓ Skipped inference, using loaded results')

# Reshape for multivariate
if args.features == 'M':
    num_features = args.c_out
    total_samples = preds_scaled.shape[0]

    if total_samples % num_features != 0:
        print(f'\nWARNING: Adjusting data size...')
        valid_samples = (total_samples // num_features) * num_features
        preds_scaled = preds_scaled[:valid_samples]
        trues_scaled = trues_scaled[:valid_samples]
        total_samples = valid_samples

    num_samples = total_samples // num_features

    print(f'\nReshaping data:')
    print(f'  Total entries: {total_samples}')
    print(f'  Number of features: {num_features}')
    print(f'  Number of samples: {num_samples}')
    print(f'  Prediction length: {args.pred_len}')

    preds_scaled = preds_scaled.squeeze(-1)
    trues_scaled = trues_scaled.squeeze(-1)

    preds_scaled = preds_scaled.reshape(num_features, num_samples, args.pred_len)
    trues_scaled = trues_scaled.reshape(num_features, num_samples, args.pred_len)

    preds_scaled = preds_scaled.transpose(1, 2, 0)
    trues_scaled = trues_scaled.transpose(1, 2, 0)

    print(f'Reshaped predictions: {preds_scaled.shape}')
    print(f'Reshaped ground truth: {trues_scaled.shape}')

# Inverse transform
print('\n' + '='*80)
print('INVERSE TRANSFORMING TO ORIGINAL SCALE')
print('='*80)
preds_original = test_data.inverse_transform(preds_scaled.reshape(-1, preds_scaled.shape[-1])).reshape(preds_scaled.shape)
trues_original = test_data.inverse_transform(trues_scaled.reshape(-1, trues_scaled.shape[-1])).reshape(trues_scaled.shape)

# Find OT feature index
# OT is the last column (index -1 or 42 for 0-indexed)
ot_index = -1

print(f'\nExtracting OT feature (index: {ot_index})')

# Extract only OT predictions
preds_scaled_ot = preds_scaled[:, :, ot_index]
trues_scaled_ot = trues_scaled[:, :, ot_index]
preds_original_ot = preds_original[:, :, ot_index]
trues_original_ot = trues_original[:, :, ot_index]

print(f'OT predictions shape: {preds_scaled_ot.shape}')

# Calculate metrics for OT only
mse_scaled = np.mean((preds_scaled_ot - trues_scaled_ot) ** 2)
mae_scaled = np.mean(np.abs(preds_scaled_ot - trues_scaled_ot))
rmse_scaled = np.sqrt(mse_scaled)

mse_original = np.mean((preds_original_ot - trues_original_ot) ** 2)
mae_original = np.mean(np.abs(preds_original_ot - trues_original_ot))
rmse_original = np.sqrt(mse_original)

# Print results
print('\n' + '='*80)
print(f'TEST RESULTS - {args.target} ONLY')
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
print(f'  Predictions: {preds_scaled_ot.shape}')
print(f'  Ground truth: {trues_scaled_ot.shape}')
print('='*80)

# Save results
final_result_file = os.path.join(folder_path, f'{args.data}_{args.model_id}_OT_results.npz')
np.savez(final_result_file,
         predictions_scaled=preds_scaled_ot,
         ground_truth_scaled=trues_scaled_ot,
         predictions_original=preds_original_ot,
         ground_truth_original=trues_original_ot,
         mse_scaled=mse_scaled,
         mae_scaled=mae_scaled,
         rmse_scaled=rmse_scaled,
         mse_original=mse_original,
         mae_original=mae_original,
         rmse_original=rmse_original)

print(f'\n✓ Results saved to {final_result_file}')
print(f'  - To reload inference: --load_results {result_file}')

# Generate plots
print('\n' + '='*80)
print(f'GENERATING PLOTS FOR {args.target}')
print('='*80)

num_samples = min(args.plot_samples, len(preds_scaled_ot))
print(f'Plotting {num_samples} samples...')

# Plot 1: Multiple samples - Scaled
print('\n[1/4] Plotting scaled predictions for multiple samples...')
fig, axes = plt.subplots(num_samples, 1, figsize=(15, 3*num_samples))
if num_samples == 1:
    axes = [axes]

for i in range(num_samples):
    time_steps = np.arange(args.pred_len)
    axes[i].plot(time_steps, trues_scaled_ot[i, :], label='Actual (Scaled)',
                 linewidth=2.5, marker='o', markersize=7, color='#2E86AB', alpha=0.8)
    axes[i].plot(time_steps, preds_scaled_ot[i, :], label='Predicted (Scaled)',
                 linewidth=2.5, marker='x', markersize=9, color='#A23B72', alpha=0.8)
    axes[i].set_title(f'Sample {i+1} - {args.target} Prediction (Scaled Data)',
                     fontsize=13, fontweight='bold', pad=10)
    axes[i].set_xlabel('Time Step (seconds)', fontsize=11)
    axes[i].set_ylabel(f'{args.target} Value (Scaled)', fontsize=11)
    axes[i].legend(fontsize=10, loc='best')
    axes[i].grid(True, alpha=0.3, linestyle='--')
    axes[i].set_xlim(-0.5, args.pred_len-0.5)

plt.tight_layout()
scaled_path = os.path.join(folder_path, f'{args.data}_{args.model_id}_OT_scaled.png')
plt.savefig(scaled_path, dpi=300, bbox_inches='tight')
print(f'  ✓ Saved: {scaled_path}')
plt.close()

# Plot 2: Multiple samples - Original scale
print('[2/4] Plotting original scale predictions for multiple samples...')
fig, axes = plt.subplots(num_samples, 1, figsize=(15, 3*num_samples))
if num_samples == 1:
    axes = [axes]

for i in range(num_samples):
    time_steps = np.arange(args.pred_len)
    axes[i].plot(time_steps, trues_original_ot[i, :], label='Actual (Original)',
                 linewidth=2.5, marker='o', markersize=7, color='#18A558', alpha=0.8)
    axes[i].plot(time_steps, preds_original_ot[i, :], label='Predicted (Original)',
                 linewidth=2.5, marker='x', markersize=9, color='#F18F01', alpha=0.8)
    axes[i].set_title(f'Sample {i+1} - {args.target} Prediction (Original Scale)',
                     fontsize=13, fontweight='bold', pad=10)
    axes[i].set_xlabel('Time Step (seconds)', fontsize=11)
    axes[i].set_ylabel(f'{args.target} Value (Original Scale)', fontsize=11)
    axes[i].legend(fontsize=10, loc='best')
    axes[i].grid(True, alpha=0.3, linestyle='--')
    axes[i].set_xlim(-0.5, args.pred_len-0.5)

plt.tight_layout()
original_path = os.path.join(folder_path, f'{args.data}_{args.model_id}_OT_original.png')
plt.savefig(original_path, dpi=300, bbox_inches='tight')
print(f'  ✓ Saved: {original_path}')
plt.close()

# Plot 3: Side-by-side comparison
print('[3/4] Plotting side-by-side comparison...')
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

time_steps = np.arange(args.pred_len)

# Scaled
axes[0].plot(time_steps, trues_scaled_ot[0, :], label='Actual',
            linewidth=2.5, marker='o', markersize=7, color='#2E86AB', alpha=0.8)
axes[0].plot(time_steps, preds_scaled_ot[0, :], label='Predicted',
            linewidth=2.5, marker='x', markersize=9, color='#A23B72', alpha=0.8)
axes[0].set_title(f'{args.target} Prediction - Scaled/Normalized Data',
                 fontsize=13, fontweight='bold', pad=10)
axes[0].set_xlabel('Time Step (seconds)', fontsize=11)
axes[0].set_ylabel(f'{args.target} Value (Scaled)', fontsize=11)
axes[0].legend(fontsize=11, loc='best')
axes[0].grid(True, alpha=0.3, linestyle='--')
axes[0].set_xlim(-0.5, args.pred_len-0.5)

# Original
axes[1].plot(time_steps, trues_original_ot[0, :], label='Actual',
            linewidth=2.5, marker='o', markersize=7, color='#18A558', alpha=0.8)
axes[1].plot(time_steps, preds_original_ot[0, :], label='Predicted',
            linewidth=2.5, marker='x', markersize=9, color='#F18F01', alpha=0.8)
axes[1].set_title(f'{args.target} Prediction - Original Scale',
                 fontsize=13, fontweight='bold', pad=10)
axes[1].set_xlabel('Time Step (seconds)', fontsize=11)
axes[1].set_ylabel(f'{args.target} Value (Original Scale)', fontsize=11)
axes[1].legend(fontsize=11, loc='best')
axes[1].grid(True, alpha=0.3, linestyle='--')
axes[1].set_xlim(-0.5, args.pred_len-0.5)

plt.tight_layout()
comparison_path = os.path.join(folder_path, f'{args.data}_{args.model_id}_OT_comparison.png')
plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
print(f'  ✓ Saved: {comparison_path}')
plt.close()

# Plot 4: Error distribution
print('[4/4] Plotting error distributions...')
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

errors_scaled = preds_scaled_ot.flatten() - trues_scaled_ot.flatten()
axes[0].hist(errors_scaled, bins=50, edgecolor='black', alpha=0.7, color='#2E86AB')
axes[0].set_title(f'{args.target} Prediction Error Distribution (Scaled Data)',
                 fontsize=13, fontweight='bold', pad=10)
axes[0].set_xlabel('Prediction Error', fontsize=11)
axes[0].set_ylabel('Frequency', fontsize=11)
axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2.5, label='Zero Error', alpha=0.8)
axes[0].axvline(x=np.mean(errors_scaled), color='green', linestyle='-.', linewidth=2,
                label=f'Mean Error: {np.mean(errors_scaled):.4f}', alpha=0.8)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3, axis='y')

errors_original = preds_original_ot.flatten() - trues_original_ot.flatten()
axes[1].hist(errors_original, bins=50, edgecolor='black', alpha=0.7, color='#F18F01')
axes[1].set_title(f'{args.target} Prediction Error Distribution (Original Scale)',
                 fontsize=13, fontweight='bold', pad=10)
axes[1].set_xlabel('Prediction Error', fontsize=11)
axes[1].set_ylabel('Frequency', fontsize=11)
axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2.5, label='Zero Error', alpha=0.8)
axes[1].axvline(x=np.mean(errors_original), color='green', linestyle='-.', linewidth=2,
                label=f'Mean Error: {np.mean(errors_original):.4f}', alpha=0.8)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
error_path = os.path.join(folder_path, f'{args.data}_{args.model_id}_OT_error_dist.png')
plt.savefig(error_path, dpi=300, bbox_inches='tight')
print(f'  ✓ Saved: {error_path}')
plt.close()

# Plot 5: Continuous time series (NEW!)
print(f'[5/6] Plotting continuous time series ({args.continuous_steps} steps)...')

# Calculate how many samples we can concatenate
max_continuous = min(args.continuous_steps // args.pred_len, len(preds_scaled_ot))

if max_continuous > 0:
    # Create continuous predictions by taking the first step of each prediction
    # This simulates a rolling prediction scenario
    continuous_pred_scaled = []
    continuous_true_scaled = []
    continuous_pred_original = []
    continuous_true_original = []

    for i in range(max_continuous):
        # Take the first prediction step from each sample
        continuous_pred_scaled.append(preds_scaled_ot[i, 0])
        continuous_true_scaled.append(trues_scaled_ot[i, 0])
        continuous_pred_original.append(preds_original_ot[i, 0])
        continuous_true_original.append(trues_original_ot[i, 0])

    continuous_pred_scaled = np.array(continuous_pred_scaled)
    continuous_true_scaled = np.array(continuous_true_scaled)
    continuous_pred_original = np.array(continuous_pred_original)
    continuous_true_original = np.array(continuous_true_original)

    fig, axes = plt.subplots(2, 1, figsize=(20, 10))

    time_axis = np.arange(len(continuous_pred_scaled))

    # Scaled
    axes[0].plot(time_axis, continuous_true_scaled, label='Actual (Scaled)',
                linewidth=2, color='#2E86AB', alpha=0.8)
    axes[0].plot(time_axis, continuous_pred_scaled, label='Predicted (Scaled)',
                linewidth=2, color='#A23B72', alpha=0.8, linestyle='--')
    axes[0].fill_between(time_axis,
                         continuous_true_scaled,
                         continuous_pred_scaled,
                         alpha=0.2, color='gray', label='Prediction Error')
    axes[0].set_title(f'{args.target} Continuous Prediction - Scaled Data ({len(continuous_pred_scaled)} time steps)',
                     fontsize=14, fontweight='bold', pad=10)
    axes[0].set_xlabel('Time Step (seconds)', fontsize=12)
    axes[0].set_ylabel(f'{args.target} Value (Scaled)', fontsize=12)
    axes[0].legend(fontsize=11, loc='best')
    axes[0].grid(True, alpha=0.3, linestyle='--')

    # Original
    axes[1].plot(time_axis, continuous_true_original, label='Actual (Original)',
                linewidth=2, color='#18A558', alpha=0.8)
    axes[1].plot(time_axis, continuous_pred_original, label='Predicted (Original)',
                linewidth=2, color='#F18F01', alpha=0.8, linestyle='--')
    axes[1].fill_between(time_axis,
                         continuous_true_original,
                         continuous_pred_original,
                         alpha=0.2, color='gray', label='Prediction Error')
    axes[1].set_title(f'{args.target} Continuous Prediction - Original Scale ({len(continuous_pred_original)} time steps)',
                     fontsize=14, fontweight='bold', pad=10)
    axes[1].set_xlabel('Time Step (seconds)', fontsize=12)
    axes[1].set_ylabel(f'{args.target} Value (Original Scale)', fontsize=12)
    axes[1].legend(fontsize=11, loc='best')
    axes[1].grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    continuous_path = os.path.join(folder_path, f'{args.data}_{args.model_id}_OT_continuous.png')
    plt.savefig(continuous_path, dpi=300, bbox_inches='tight')
    print(f'  ✓ Saved: {continuous_path}')
    plt.close()
else:
    print(f'  ⚠ Not enough samples for continuous plot')

# Plot 6: Multiple continuous predictions (zoomed in)
print(f'[6/6] Plotting detailed multi-step predictions...')

# Show 3 consecutive predictions with all their steps
num_detailed = min(3, len(preds_scaled_ot))

fig, axes = plt.subplots(num_detailed, 2, figsize=(18, 5*num_detailed))
if num_detailed == 1:
    axes = axes.reshape(1, -1)

for i in range(num_detailed):
    time_steps = np.arange(args.pred_len)

    # Scaled
    axes[i, 0].plot(time_steps, trues_scaled_ot[i, :], label='Actual',
                   linewidth=2.5, marker='o', markersize=8, color='#2E86AB', alpha=0.8)
    axes[i, 0].plot(time_steps, preds_scaled_ot[i, :], label='Predicted',
                   linewidth=2.5, marker='x', markersize=10, color='#A23B72', alpha=0.8)
    axes[i, 0].fill_between(time_steps,
                            trues_scaled_ot[i, :],
                            preds_scaled_ot[i, :],
                            alpha=0.2, color='gray')
    axes[i, 0].set_title(f'Prediction Window {i+1} - Scaled ({args.pred_len} steps)',
                        fontsize=12, fontweight='bold')
    axes[i, 0].set_xlabel('Time Step', fontsize=10)
    axes[i, 0].set_ylabel(f'{args.target} (Scaled)', fontsize=10)
    axes[i, 0].legend(fontsize=10)
    axes[i, 0].grid(True, alpha=0.3, linestyle='--')

    # Original
    axes[i, 1].plot(time_steps, trues_original_ot[i, :], label='Actual',
                   linewidth=2.5, marker='o', markersize=8, color='#18A558', alpha=0.8)
    axes[i, 1].plot(time_steps, preds_original_ot[i, :], label='Predicted',
                   linewidth=2.5, marker='x', markersize=10, color='#F18F01', alpha=0.8)
    axes[i, 1].fill_between(time_steps,
                            trues_original_ot[i, :],
                            preds_original_ot[i, :],
                            alpha=0.2, color='gray')
    axes[i, 1].set_title(f'Prediction Window {i+1} - Original Scale ({args.pred_len} steps)',
                        fontsize=12, fontweight='bold')
    axes[i, 1].set_xlabel('Time Step', fontsize=10)
    axes[i, 1].set_ylabel(f'{args.target} (Original)', fontsize=10)
    axes[i, 1].legend(fontsize=10)
    axes[i, 1].grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
detailed_path = os.path.join(folder_path, f'{args.data}_{args.model_id}_OT_detailed.png')
plt.savefig(detailed_path, dpi=300, bbox_inches='tight')
print(f'  ✓ Saved: {detailed_path}')
plt.close()

print('\n' + '='*80)
print(f'✓ All plots saved to: {folder_path}')
print('='*80)

print(f'\n✓ Testing completed for {args.target}!')
