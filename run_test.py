import argparse
import torch
import numpy as np
import os
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
from torch import nn
from tqdm import tqdm

from models import Autoformer, DLinear, TimeLLM
from data_provider.data_factory import data_provider
from utils.tools import load_content

parser = argparse.ArgumentParser(description='Time-LLM Testing')

# basic config
parser.add_argument('--task_name', type=str, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--model_id', type=str, required=True, help='model id')
parser.add_argument('--model_comment', type=str, required=True, help='prefix when saving test results')
parser.add_argument('--model', type=str, default='TimeLLM', help='model name')
parser.add_argument('--checkpoint_path', type=str, required=True, help='path to trained checkpoint file')

# data loader
parser.add_argument('--data', type=str, required=True, help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset', help='root path of the data file')
parser.add_argument('--data_path', type=str, required=True, help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

# model define
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--llm_model', type=str, default='LLAMA', help='LLM model')
parser.add_argument('--llm_dim', type=int, default=4096, help='LLM model dimension')
parser.add_argument('--llm_layers', type=int, default=32)

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of test input data')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision', default=False)
parser.add_argument('--percent', type=int, default=100)

args = parser.parse_args()

# Setup accelerator
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin)

# Load test data
accelerator.print('Loading test dataset...')
test_data, test_loader = data_provider(args, 'test')

# Build model
accelerator.print('Building model...')
if args.model == 'Autoformer':
    model = Autoformer.Model(args).float()
elif args.model == 'DLinear':
    model = DLinear.Model(args).float()
else:
    model = TimeLLM.Model(args).float()

# Load content for TimeLLM
args.content = load_content(args)

# Load checkpoint
accelerator.print(f'Loading checkpoint from {args.checkpoint_path}...')
if not os.path.exists(args.checkpoint_path):
    raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint_path}")

checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint)

# Prepare model
test_loader, model = accelerator.prepare(test_loader, model)

# Test
accelerator.print('Starting inference on test set...')
model.eval()

preds = []
trues = []
criterion = nn.MSELoss()
mae_metric = nn.L1Loss()

total_loss = []
total_mae_loss = []

with torch.no_grad():
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(test_loader)):
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
preds = np.concatenate(preds, axis=0)
trues = np.concatenate(trues, axis=0)

# Calculate metrics
mse = np.mean((preds - trues) ** 2)
mae = np.mean(np.abs(preds - trues))
rmse = np.sqrt(mse)

avg_loss = np.average(total_loss)
avg_mae_loss = np.average(total_mae_loss)

# Print results
accelerator.print('\n' + '='*50)
accelerator.print('TEST RESULTS')
accelerator.print('='*50)
accelerator.print(f'MSE Loss: {avg_loss:.7f}')
accelerator.print(f'MAE Loss: {avg_mae_loss:.7f}')
accelerator.print(f'RMSE: {rmse:.7f}')
accelerator.print(f'Predictions shape: {preds.shape}')
accelerator.print(f'Ground truth shape: {trues.shape}')
accelerator.print('='*50)

# Save results
if accelerator.is_local_main_process:
    folder_path = './test_results/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    result_file = os.path.join(folder_path, f'{args.data}_{args.model_id}_{args.pred_len}_results.npz')
    np.savez(result_file,
             predictions=preds,
             ground_truth=trues,
             mse=mse,
             mae=mae,
             rmse=rmse)

    accelerator.print(f'\nResults saved to {result_file}')

accelerator.print('\nTesting completed!')
