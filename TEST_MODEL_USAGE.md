# Test Model Usage Guide

## Overview
The `test_model.py` script has been configured to match your `GPT2_PPData.sh` training script.

## Key Configuration Changes

### Model Settings
- **Model**: GPT2 (uses TimeLLM architecture with GPT2 backbone)
- **Task**: short_term_forecast (changed from long_term_forecast)
- **Model ID**: PPData_90_6

### Sequence Settings (Short-term Forecast)
- **seq_len**: 90 (input sequence length)
- **label_len**: 6 (start token length)
- **pred_len**: 6 (prediction length - SHORT TERM!)

### LLM Configuration
- **llm_model**: GPT2
- **llm_dim**: 4096
- **llm_layers**: 6

### Other Settings
- **freq**: 's' (seconds, not hours)
- **batch_size**: 24
- **description**: 'Power plant data'

## Expected Checkpoint Path

Based on your GPT2_PPData.sh configuration, your checkpoint should be at:
```
./checkpoints/short_term_forecast_PPData_90_6_GPT2_PPData_ftM_sl90_ll6_pl6_dm16_nh8_el2_dl1_df32_fc3_ebtimeF_Power plant data_0-GPT2-PPData/checkpoint
```

## How to Use

### Step 1: Find your checkpoint
```bash
find ./checkpoints -name "checkpoint" -type f
```

### Step 2: Update checkpoint path in test_model.py
Edit the `CONFIG['checkpoint_path']` at the top of `test_model.py` to match your actual checkpoint path.

### Step 3: Run the test
```bash
python test_model.py
```

## Output

The script will generate:

1. **Plots** (saved to `./test_results/`):
   - `PPData_scaled_predictions.png` - Predictions on scaled data
   - `PPData_original_predictions.png` - Predictions on original scale
   - `PPData_comparison.png` - Side-by-side comparison
   - `PPData_error_distribution.png` - Error histograms

2. **Metrics**:
   - MSE, MAE, RMSE on both scaled and original data

3. **Numerical Results**:
   - `PPData_PPData_90_6_results.npz` - All predictions and metrics

## Important Notes

⚠️ **Note about pred_len**: Your model predicts only **6 time steps** (short-term), not 96!

⚠️ **Note about llm_model**: Your `GPT2_PPData.sh` script doesn't explicitly set `--llm_model`. 
If it defaults to LLAMA during training, you may need to change `llm_model` to `'LLAMA'` in the CONFIG.

## Verification

To verify your training configuration matches the test configuration:
```bash
grep -E "model|seq_len|label_len|pred_len|llm" ./scripts/GPT2_PPData.sh
```
