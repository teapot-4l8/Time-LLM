# run_test.py - Missing Arguments Added

## Summary
I've updated `run_test.py` to include ALL missing arguments from `run_main.py`, ensuring full compatibility between training and testing scripts.

## Arguments Added

### 1. **--seed**
- **Type**: int
- **Default**: 2021
- **Description**: Random seed for reproducibility
- **Usage**: Automatically sets random seeds for Python, NumPy, and PyTorch

### 2. **--loader**
- **Type**: str
- **Default**: 'modal'
- **Description**: Dataset type/loader specification

### 3. **--seasonal_patterns**
- **Type**: str
- **Default**: 'Monthly'
- **Description**: Subset for M4 dataset

### 4. **--moving_avg**
- **Type**: int
- **Default**: 25
- **Description**: Window size of moving average (used in some models like Autoformer)

### 5. **--prompt_domain**
- **Type**: int
- **Default**: 0
- **Description**: Prompt domain setting for Time-LLM

## Code Changes

### 1. **Import Added**
```python
import random  # Added for seed setting
```

### 2. **Seed Initialization Added**
```python
# Set random seed for reproducibility
random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
```

### 3. **Enhanced Argument Descriptions**
Updated help texts to match run_main.py format:
- `--features`: Full description of M/S/MS options
- `--freq`: Complete list of frequency options
- `--llm_model`: Comment with LLAMA, GPT2, BERT options
- `--llm_dim`: Comment with dimension mappings

## Complete Argument List

Now `run_test.py` has the same arguments as `run_main.py`:

### Basic Config
- task_name
- model_id
- model_comment
- model
- checkpoint_path *(unique to test)*
- seed ✅ *NEW*

### Data Loader
- data
- root_path
- data_path
- features
- target
- loader ✅ *NEW*
- freq
- checkpoints

### Forecasting Task
- seq_len
- label_len
- pred_len
- seasonal_patterns ✅ *NEW*

### Model Architecture
- enc_in
- dec_in
- c_out
- d_model
- n_heads
- e_layers
- d_layers
- d_ff
- moving_avg ✅ *NEW*
- factor
- dropout
- embed
- activation
- output_attention
- patch_len
- stride
- prompt_domain ✅ *NEW*
- llm_model
- llm_dim
- llm_layers

### Optimization/Testing
- num_workers
- batch_size
- des
- use_amp
- percent
- plot_samples *(unique to test)*

## Benefits

✅ **Full compatibility** - Test script now accepts all training arguments
✅ **Reproducibility** - Seed setting ensures consistent results
✅ **Model support** - All model-specific args (moving_avg, prompt_domain) now available
✅ **Dataset support** - M4 and other dataset-specific args supported
✅ **No breaking changes** - All existing functionality preserved

## Usage Example

You can now pass ALL the same arguments from training to testing:

```bash
# From GPT2_PPData.sh training script
accelerate launch run_main.py \
  --task_name short_term_forecast \
  --model GPT2 \
  --data PPData \
  --seq_len 90 --label_len 6 --pred_len 6 \
  --llm_layers 6 \
  ... (training args)

# Now for testing with EXACT same args
accelerate launch run_test.py \
  --checkpoint_path ./checkpoints/YOUR_CHECKPOINT/checkpoint \
  --task_name short_term_forecast \
  --model GPT2 \
  --data PPData \
  --seq_len 90 --label_len 6 --pred_len 6 \
  --llm_layers 6 \
  ... (same args!) \
  --plot_samples 5
```

## Verification

To verify all arguments are present, compare:

```bash
# Check run_main.py arguments
grep "parser.add_argument" run_main.py | wc -l

# Check run_test.py arguments
grep "parser.add_argument" run_test.py | wc -l
```

Both should have similar counts (test has +1 for checkpoint_path and +1 for plot_samples).
