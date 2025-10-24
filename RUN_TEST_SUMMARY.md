# run_test.py - Enhanced Testing Script with Plotting

## Overview
I've updated `run_test.py` to include comprehensive plotting functionality that visualizes **OT (target) predictions before and after scaling**.

## New Features Added

### 1. **Automatic Inverse Scaling**
- Predictions and ground truth are automatically inverse transformed from scaled space back to original scale
- Both scaled and original metrics are calculated and reported

### 2. **Comprehensive Metrics**
The script now reports metrics for **both scaled and original data**:

```
TEST RESULTS
================================================================================
Scaled Data Metrics (Normalized):
  MSE:  X.XXXXXXX
  MAE:  X.XXXXXXX
  RMSE: X.XXXXXXX

Original Scale Metrics:
  MSE:  XXX.XXXXXXX
  MAE:  XXX.XXXXXXX
  RMSE: XXX.XXXXXXX
```

### 3. **Four Types of Plots Generated**

#### Plot 1: Scaled Predictions (Multiple Samples)
- **File**: `{data}_{model_id}_scaled_predictions.png`
- Shows actual vs predicted on **normalized/scaled data**
- Displays multiple samples (default: 5)
- Blue line with circles = Actual (Scaled)
- Red line with X markers = Predicted (Scaled)

#### Plot 2: Original Scale Predictions (Multiple Samples)
- **File**: `{data}_{model_id}_original_predictions.png`
- Shows actual vs predicted on **original scale**
- Displays multiple samples (default: 5)
- Green line with circles = Actual (Original)
- Red line with X markers = Predicted (Original)

#### Plot 3: Side-by-Side Comparison
- **File**: `{data}_{model_id}_comparison.png`
- Side-by-side comparison of **scaled vs original** for first sample
- Left panel: Scaled data
- Right panel: Original scale data

#### Plot 4: Error Distribution
- **File**: `{data}_{model_id}_error_distribution.png`
- Histograms showing prediction error distributions
- Left panel: Scaled data errors
- Right panel: Original scale errors
- Red dashed line shows zero error

### 4. **Enhanced Data Saving**
Results file now includes:
- `predictions_scaled` - Predictions on scaled data
- `ground_truth_scaled` - Actual values on scaled data
- `predictions_original` - Predictions on original scale
- `ground_truth_original` - Actual values on original scale
- All metrics for both scales

## New Command-Line Argument

### `--plot_samples`
- **Type**: int
- **Default**: 5
- **Description**: Number of samples to plot
- **Usage**: `--plot_samples 10` to plot 10 samples

## Usage

### Basic Usage
```bash
bash scripts/test_PPData.sh
```

### Custom Number of Plot Samples
Edit `test_PPData.sh` and change:
```bash
plot_samples=10  # Change from 5 to 10
```

Or run directly:
```bash
python run_test.py \
  --checkpoint_path ./checkpoints/YOUR_CHECKPOINT/checkpoint \
  --data PPData \
  --data_path PPData.csv \
  --model GPT2 \
  --model_id PPData_90_6 \
  --seq_len 90 \
  --label_len 6 \
  --pred_len 6 \
  --enc_in 43 --dec_in 43 --c_out 43 \
  --plot_samples 10
```

## Output Files

After running the test, you'll find in `./test_results/`:

1. **Data File**: `PPData_PPData_90_6_results.npz`
   - Contains all predictions and metrics

2. **Plot Files**:
   - `PPData_PPData_90_6_scaled_predictions.png` - Scaled predictions
   - `PPData_PPData_90_6_original_predictions.png` - Original scale predictions
   - `PPData_PPData_90_6_comparison.png` - Side-by-side comparison
   - `PPData_PPData_90_6_error_distribution.png` - Error distributions

## Key Points

✅ **Automatic inverse transformation** - No manual scaling needed
✅ **Both scales shown** - See predictions in normalized and original units
✅ **Multiple samples** - Compare predictions across different test samples
✅ **Error analysis** - Visualize prediction error distributions
✅ **High quality** - All plots saved at 300 DPI
✅ **OT focused** - Plots specifically show the target variable (OT)

## Technical Details

- **Target variable**: OT (last column in dataset)
- **Scaling method**: StandardScaler (z-score normalization)
- **Inverse transform**: Uses `test_data.inverse_transform()`
- **Plot resolution**: 300 DPI
- **Figure sizes**: Optimized for clarity and detail

## Example Output

When you run the script, you'll see:

```
Loading test dataset...
Building model...
Loading checkpoint from ./checkpoints/...
Starting inference on test set...
  Testing: 100%|████████████| 50/50 [00:30<00:00,  1.65it/s]

Inverse transforming to original scale...

================================================================================
TEST RESULTS
================================================================================

Scaled Data Metrics (Normalized):
  MSE:  0.0234567
  MAE:  0.1234567
  RMSE: 0.1531234

Original Scale Metrics:
  MSE:  123.4567890
  MAE:  9.8765432
  RMSE: 11.1111111

Data shapes:
  Predictions: (100, 6, 43)
  Ground truth: (100, 6, 43)
================================================================================

Results saved to ./test_results/PPData_PPData_90_6_results.npz

================================================================================
GENERATING PLOTS
================================================================================
Plotting 5 samples...

[1/4] Plotting scaled predictions...
  ✓ Saved: ./test_results/PPData_PPData_90_6_scaled_predictions.png
[2/4] Plotting original scale predictions...
  ✓ Saved: ./test_results/PPData_PPData_90_6_original_predictions.png
[3/4] Plotting side-by-side comparison...
  ✓ Saved: ./test_results/PPData_PPData_90_6_comparison.png
[4/4] Plotting error distributions...
  ✓ Saved: ./test_results/PPData_PPData_90_6_error_distribution.png

================================================================================
✓ All plots saved to: ./test_results/
================================================================================

Testing completed!
```
