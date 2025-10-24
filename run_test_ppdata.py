import argparse
import torch
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
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

parser = argparse.ArgumentParser(description='Time-LLM Testing for PPData')

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# basic config
parser.add_argument('--task_name', type=str, required=True, default='short_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model_comment', type=str, required=True, default='none', help='prefix when saving test results')
parser.add_argument('--model', type=str, required=True, default='GPT2',
                    help='model name, options: [Autoformer, DLinear, TimeLLM, GPT2]')
parser.add_argument('--seed', type=int, default=2021, help='random seed')
parser.add_argument('--checkpoint_path', type=str, required=True, help='path to trained checkpoint file')

# data loader
parser.add_argument('--data', type=str, required=True, default='PPData', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='PPData.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; '
                         'M:multivariate predict multivariate, S: univariate predict univariate, '
                         'MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--loader', type=str, default='modal', help='dataset type')
parser.add_argument('--freq', type=str, default='s',
                    help='freq for time features encoding, '
                         'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                         'you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=90, help='input sequence length')
parser.add_argument('--label_len', type=int, default=6, help='start token length')
parser.add_argument('--pred_len', type=int, default=6, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

# model define
parser.add_argument('--enc_in', type=int, default=43, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=43, help='decoder input size')
parser.add_argument('--c_out', type=int, default=43, help='output size')
parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=3, help='attn factor')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--prompt_domain', type=int, default=0, help='')
parser.add_argument('--llm_model', type=str, default='LLAMA', help='LLM model') # LLAMA, GPT2, BERT
parser.add_argument('--llm_dim', type=int, default=4096, help='LLM model dimension')# LLama7b:4096; GPT2-small:768; BERT-base:768
parser.add_argument('--llm_layers', type=int, default=6)

# testing
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--batch_size', type=int, default=24, help='batch size of test input data')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision', default=False)
parser.add_argument('--percent', type=int, default=100)
parser.add_argument('--plot_samples', type=int, default=5, help='number of samples to plot')
parser.add_argument('--plot_features', type=int, default=3, help='number of features to plot (for multivariate)')

args = parser.parse_args()

# Update seed based on args
random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin)

# Load test data
accelerator.print('='*80)
accelerator.print('LOADING TEST DATA')
accelerator.print('='*80)
test_data, test_loader = data_provider(args, 'test')
accelerator.print(f'Test samples: {len(test_data)}')
accelerator.print(f'Number of features: {args.c_out}')

# Build model
accelerator.print('\n' + '='*80)
accelerator.print('BUILDING MODEL')
accelerator.print('='*80)
if args.model == 'Autoformer':
    model = Autoformer.Model(args).float()
elif args.model == 'DLinear':
    model = DLinear.Model(args).float()
else:
    model = TimeLLM.Model(args).float()

# Load content for TimeLLM
args.content = load_content(args)

# Load checkpoint
accelerator.print('\n' + '='*80)
accelerator.print('LOADING CHECKPOINT')
accelerator.print('='*80)
accelerator.print(f'Checkpoint path: {args.checkpoint_path}')
if not os.path.exists(args.checkpoint_path):
    accelerator.print(f'ERROR: Checkpoint not found at {args.checkpoint_path}')
    accelerator.print('\nSearching for available checkpoints...')
    if os.path.exists('./checkpoints'):
        for root, dirs, files in os.walk('./checkpoints'):
            for file in files:
                if file == 'checkpoint':
                    accelerator.print(f'  Found: {os.path.join(root, file)}')
    else:
        accelerator.print('  ./checkpoints directory does not exist')
    exit(1)

checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint)
accelerator.print('✓ Checkpoint loaded successfully')

# Prepare model
test_loader, model = accelerator.prepare(test_loader, model)

# Test
accelerator.print('\n' + '='*80)
accelerator.print('RUNNING INFERENCE')
accelerator.print('='*80)
model.eval()

preds = []
trues = []
criterion = nn.MSELoss()
mae_metric = nn.L1Loss()

total_loss = []
total_mae_loss = []

with torch.no_grad():
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(test_loader), desc='Testing', disable=not accelerator.is_local_main_process):
        batch_x = batch_x.float().to(accelerator.device)
        batch_y = batch_y.float().to(accelerator.device)
        batch_x_mark = batch_x_mark.float().to(accelerator.device)
        batch_y_mark = batch_y_mark.float().to(accelerator.device)

        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(accelerator.device)

        # encoder - decoder
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

        outputs, batch_y = accelerator.gather_for_metrics((outputs, batch_y))

        f_dim = -1 if args.features == 'MS' else 0
        outputs = outputs[:, -args.pred_len:, f_dim:]
        batch_y = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device)

        pred = outputs.detach().cpu().numpy()
        true = batch_y.detach().cpu().numpy()

        preds.append(pred)
        trues.append(true)

        # Calculate loss
        loss = criterion(outputs, batch_y)
        mae_loss = mae_metric(outputs, batch_y)

        total_loss.append(loss.item())
        total_mae_loss.append(mae_loss.item())

# Concatenate all predictions
preds_scaled = np.concatenate(preds, axis=0)
trues_scaled = np.concatenate(trues, axis=0)

accelerator.print(f'Predictions shape: {preds_scaled.shape}')
accelerator.print(f'Ground truth shape: {trues_scaled.shape}')

# Inverse transform to get original scale
accelerator.print('\n' + '='*80)
accelerator.print('INVERSE TRANSFORMING TO ORIGINAL SCALE')
accelerator.print('='*80)
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
accelerator.print('\n' + '='*80)
accelerator.print('TEST RESULTS')
accelerator.print('='*80)
accelerator.print(f'\nScaled Data Metrics (Normalized):')
accelerator.print(f'  MSE:  {mse_scaled:.7f}')
accelerator.print(f'  MAE:  {mae_scaled:.7f}')
accelerator.print(f'  RMSE: {rmse_scaled:.7f}')
accelerator.print(f'\nOriginal Scale Metrics:')
accelerator.print(f'  MSE:  {mse_original:.7f}')
accelerator.print(f'  MAE:  {mae_original:.7f}')
accelerator.print(f'  RMSE: {rmse_original:.7f}')
accelerator.print(f'\nData shapes:')
accelerator.print(f'  Predictions: {preds_scaled.shape}')
accelerator.print(f'  Ground truth: {trues_scaled.shape}')
accelerator.print('='*80)

# Save results and generate plots
if accelerator.is_local_main_process:
    folder_path = './test_results/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    result_file = os.path.join(folder_path, f'{args.data}_{args.model_id}_{args.pred_len}_results.npz')
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

    accelerator.print(f'\n✓ Results saved to {result_file}')

    # Generate plots
    accelerator.print('\n' + '='*80)
    accelerator.print('GENERATING PLOTS')
    accelerator.print('='*80)

    num_samples = min(args.plot_samples, len(preds_scaled))
    num_features = min(args.plot_features, preds_scaled.shape[-1])

    accelerator.print(f'Plotting {num_samples} samples with {num_features} features...')

    # Plot 1: Multiple features comparison (Scaled data)
    accelerator.print('\n[1/6] Plotting scaled predictions for multiple features...')
    fig, axes = plt.subplots(num_features, 1, figsize=(15, 4*num_features))
    if num_features == 1:
        axes = [axes]

    for feat_idx in range(num_features):
        axes[feat_idx].plot(trues_scaled[0, :, feat_idx], label=f'Actual (Feature {feat_idx})',
                     linewidth=2, marker='o', markersize=6, color='blue')
        axes[feat_idx].plot(preds_scaled[0, :, feat_idx], label=f'Predicted (Feature {feat_idx})',
                     linewidth=2, marker='x', markersize=8, color='red')
        axes[feat_idx].set_title(f'Feature {feat_idx} - Scaled Prediction',
                         fontsize=12, fontweight='bold')
        axes[feat_idx].set_xlabel('Time Step', fontsize=10)
        axes[feat_idx].set_ylabel(f'Value (Scaled)', fontsize=10)
        axes[feat_idx].legend(fontsize=10)
        axes[feat_idx].grid(True, alpha=0.3)

    plt.tight_layout()
    scaled_features_path = os.path.join(folder_path, f'{args.data}_{args.model_id}_scaled_features.png')
    plt.savefig(scaled_features_path, dpi=300, bbox_inches='tight')
    accelerator.print(f'  ✓ Saved: {scaled_features_path}')
    plt.close()

    # Plot 2: Multiple features comparison (Original scale)
    accelerator.print('[2/6] Plotting original scale predictions for multiple features...')
    fig, axes = plt.subplots(num_features, 1, figsize=(15, 4*num_features))
    if num_features == 1:
        axes = [axes]

    for feat_idx in range(num_features):
        axes[feat_idx].plot(trues_original[0, :, feat_idx], label=f'Actual (Feature {feat_idx})',
                     linewidth=2, marker='o', markersize=6, color='green')
        axes[feat_idx].plot(preds_original[0, :, feat_idx], label=f'Predicted (Feature {feat_idx})',
                     linewidth=2, marker='x', markersize=8, color='red')
        axes[feat_idx].set_title(f'Feature {feat_idx} - Original Scale Prediction',
                         fontsize=12, fontweight='bold')
        axes[feat_idx].set_xlabel('Time Step', fontsize=10)
        axes[feat_idx].set_ylabel(f'Value (Original)', fontsize=10)
        axes[feat_idx].legend(fontsize=10)
        axes[feat_idx].grid(True, alpha=0.3)

    plt.tight_layout()
    original_features_path = os.path.join(folder_path, f'{args.data}_{args.model_id}_original_features.png')
    plt.savefig(original_features_path, dpi=300, bbox_inches='tight')
    accelerator.print(f'  ✓ Saved: {original_features_path}')
    plt.close()

    # Plot 3: First feature - multiple samples (Scaled)
    accelerator.print('[3/6] Plotting scaled predictions for multiple samples (Feature 0)...')
    fig, axes = plt.subplots(num_samples, 1, figsize=(15, 4*num_samples))
    if num_samples == 1:
        axes = [axes]

    for i in range(num_samples):
        axes[i].plot(trues_scaled[i, :, 0], label='Actual (Scaled)',
                     linewidth=2, marker='o', markersize=6, color='blue')
        axes[i].plot(preds_scaled[i, :, 0], label='Predicted (Scaled)',
                     linewidth=2, marker='x', markersize=8, color='red')
        axes[i].set_title(f'Sample {i+1} - Feature 0 Prediction (Scaled/Normalized Data)',
                         fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Time Step', fontsize=10)
        axes[i].set_ylabel(f'Value (Scaled)', fontsize=10)
        axes[i].legend(fontsize=10)
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    scaled_samples_path = os.path.join(folder_path, f'{args.data}_{args.model_id}_scaled_samples.png')
    plt.savefig(scaled_samples_path, dpi=300, bbox_inches='tight')
    accelerator.print(f'  ✓ Saved: {scaled_samples_path}')
    plt.close()

    # Plot 4: First feature - multiple samples (Original scale)
    accelerator.print('[4/6] Plotting original scale predictions for multiple samples (Feature 0)...')
    fig, axes = plt.subplots(num_samples, 1, figsize=(15, 4*num_samples))
    if num_samples == 1:
        axes = [axes]

    for i in range(num_samples):
        axes[i].plot(trues_original[i, :, 0], label='Actual (Original)',
                     linewidth=2, marker='o', markersize=6, color='green')
        axes[i].plot(preds_original[i, :, 0], label='Predicted (Original)',
                     linewidth=2, marker='x', markersize=8, color='red')
        axes[i].set_title(f'Sample {i+1} - Feature 0 Prediction (Original Scale)',
                         fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Time Step', fontsize=10)
        axes[i].set_ylabel(f'Value (Original Scale)', fontsize=10)
        axes[i].legend(fontsize=10)
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    original_samples_path = os.path.join(folder_path, f'{args.data}_{args.model_id}_original_samples.png')
    plt.savefig(original_samples_path, dpi=300, bbox_inches='tight')
    accelerator.print(f'  ✓ Saved: {original_samples_path}')
    plt.close()

    # Plot 5: Side-by-side comparison for first sample and first feature
    accelerator.print('[5/6] Plotting side-by-side comparison...')
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Scaled
    axes[0].plot(trues_scaled[0, :, 0], label='Actual',
                linewidth=2, marker='o', markersize=6, color='blue')
    axes[0].plot(preds_scaled[0, :, 0], label='Predicted',
                linewidth=2, marker='x', markersize=8, color='red')
    axes[0].set_title(f'Feature 0 Prediction - Scaled/Normalized Data',
                     fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Time Step', fontsize=10)
    axes[0].set_ylabel(f'Value (Scaled)', fontsize=10)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Original
    axes[1].plot(trues_original[0, :, 0], label='Actual',
                linewidth=2, marker='o', markersize=6, color='green')
    axes[1].plot(preds_original[0, :, 0], label='Predicted',
                linewidth=2, marker='x', markersize=8, color='red')
    axes[1].set_title(f'Feature 0 Prediction - Original Scale',
                     fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Time Step', fontsize=10)
    axes[1].set_ylabel(f'Value (Original Scale)', fontsize=10)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    comparison_path = os.path.join(folder_path, f'{args.data}_{args.model_id}_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    accelerator.print(f'  ✓ Saved: {comparison_path}')
    plt.close()

    # Plot 6: Error distribution for first feature
    accelerator.print('[6/6] Plotting error distributions...')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scaled errors
    errors_scaled = preds_scaled[:, :, 0].flatten() - trues_scaled[:, :, 0].flatten()
    axes[0].hist(errors_scaled, bins=50, edgecolor='black', alpha=0.7, color='blue')
    axes[0].set_title('Prediction Error Distribution (Scaled Data, Feature 0)',
                     fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Error', fontsize=10)
    axes[0].set_ylabel('Frequency', fontsize=10)
    axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Original errors
    errors_original = preds_original[:, :, 0].flatten() - trues_original[:, :, 0].flatten()
    axes[1].hist(errors_original, bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[1].set_title('Prediction Error Distribution (Original Scale, Feature 0)',
                     fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Error', fontsize=10)
    axes[1].set_ylabel('Frequency', fontsize=10)
    axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    error_path = os.path.join(folder_path, f'{args.data}_{args.model_id}_error_distribution.png')
    plt.savefig(error_path, dpi=300, bbox_inches='tight')
    accelerator.print(f'  ✓ Saved: {error_path}')
    plt.close()

    accelerator.print('\n' + '='*80)
    accelerator.print(f'✓ All plots saved to: {folder_path}')
    accelerator.print('='*80)

accelerator.print('\n✓ Testing completed!')
