# 混合递归预测指南

## 概述

这个脚本实现了一个**混合递归预测**策略，专门用于预测OT列，同时保持其他42个特征列为真实值。

## 核心策略

### 混合递归的含义

在传统的递归预测中，所有预测值都会被用于下一步预测。但在**混合递归预测**中：

- **只有OT列使用预测值**
- **其他42个特征列保持真实值**

这种方法更符合实际应用场景，因为在实时预测中，我们通常只需要预测某一个关键指标（OT），而其他传感器数据仍然可以实时获取。

## 工作流程

### 步骤1: 初始预测
```
时间步 0-89: 所有43列使用真实值
     ↓
  模型预测
     ↓
时间步 90: 预测6步 [90, 91, 92, 93, 94, 95]
     ↓
  只取第一步 (step 0)
     ↓
时间步 90 的 OT 预测值
```

### 步骤2: 混合数据并递归
```
时间步 1-90:
  - OT列 (index 42): 使用步骤1的预测值 (仅时间步90)
  - 其他42列: 保持真实值
     ↓
  模型预测
     ↓
时间步 91: 预测6步 [91, 92, 93, 94, 95, 96]
     ↓
  只取第一步 (step 0)
     ↓
时间步 91 的 OT 预测值
```

### 步骤3: 继续递归
```
时间步 2-91:
  - OT列 (index 42): 时间步90-91使用预测值
  - 其他42列: 保持真实值
     ↓
  模型预测
     ↓
时间步 92: 预测6步 [92, 93, 94, 95, 96, 97]
     ↓
  只取第一步 (step 0)
     ↓
时间步 92 的 OT 预测值
```

### 重复N次
继续这个过程，直到达到 `recursive_steps` 参数指定的步数（默认200步）。

## 关键特性

### 1. 只使用第一步预测
虽然模型的 `pred_len=6`（预测6步），但我们**只使用第0步**（第一步）的预测值：
- 这样可以保持预测的连续性
- 避免预测窗口的跳跃
- 每次递归只前进1步

### 2. 混合数据策略
```python
# 伪代码示例
for step in range(recursive_steps):
    # 准备输入窗口
    input_window = data[step : step + 90, :]  # 所有43列

    # 进行预测
    prediction = model.predict(input_window)  # 形状: (6, 43)

    # 只取第一步
    first_step_pred = prediction[0, :]  # 形状: (43,)

    # 只更新OT列（index=-1或42）
    data[step + 90, -1] = first_step_pred[-1]  # 只更新OT
    # 其他42列保持不变（仍然是真实值）
```

### 3. 数据尺度处理
- 预测在**归一化（scaled）数据**上进行
- 结果会被**逆变换到原始尺度**
- 两种尺度的结果都会被保存和可视化

## 运行方法

### 使用Shell脚本（推荐）
```bash
bash ./scripts/GPT2_PPData_hybrid_recursive.sh
```

### 直接运行Python脚本
```bash
python run_hybrid_recursive.py \
  --task_name short_term_forecast \
  --root_path ./dataset/ \
  --data_path PPData.csv \
  --model_id PPData_90_6_hybrid_recursive \
  --model GPT2 \
  --data PPData \
  --features M \
  --target OT \
  --seq_len 90 \
  --label_len 6 \
  --pred_len 6 \
  --checkpoint_path <你的checkpoint路径> \
  --recursive_steps 200 \
  --use_gpu
```

## 重要参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--recursive_steps` | 递归预测的步数 | 200 |
| `--seq_len` | 输入序列长度 | 90 |
| `--pred_len` | 预测长度（但只用第0步） | 6 |
| `--batch_size` | 批次大小（必须为1） | 1 |
| `--checkpoint_path` | 模型检查点路径 | 必需 |

## 输出结果

### 1. 数值结果文件
保存位置: `./test_results_hybrid_recursive/`

文件: `PPData_<model_id>_hybrid_recursive_results.npz`

包含内容:
- `predictions_scaled`: 归一化尺度的预测值
- `ground_truth_scaled`: 归一化尺度的真实值
- `predictions_original`: 原始尺度的预测值
- `ground_truth_original`: 原始尺度的真实值
- `mse_scaled`, `mae_scaled`, `rmse_scaled`: 归一化尺度的指标
- `mse_original`, `mae_original`, `rmse_original`: 原始尺度的指标

### 2. 可视化图表

#### 图1: 归一化尺度的完整预测
`*_hybrid_recursive_scaled.png`
- 显示200步（或指定步数）的连续预测
- 归一化数据
- 包含真实值、预测值和误差区域

#### 图2: 原始尺度的完整预测
`*_hybrid_recursive_original.png`
- 显示200步（或指定步数）的连续预测
- 原始数据尺度
- 包含真实值、预测值和误差区域

#### 图3: 对比图
`*_hybrid_recursive_comparison.png`
- 上半部分: 归一化尺度
- 下半部分: 原始尺度
- 方便直接对比两种尺度

#### 图4: 误差分析
`*_hybrid_recursive_error_analysis.png`
- 左上: 归一化误差分布直方图
- 右上: 原始尺度误差分布直方图
- 左下: 归一化误差随时间变化
- 右下: 原始尺度误差随时间变化

## 与标准测试的区别

| 特性 | 标准测试 | 混合递归预测 |
|------|----------|--------------|
| 输入数据 | 所有特征都是真实值 | OT列逐步被预测值替换，其他列保持真实值 |
| 预测方式 | 每个窗口独立预测 | 递归预测，前一步的OT预测用于下一步 |
| 使用的预测步数 | 所有6步 | 只使用第0步（第一步） |
| 批次大小 | 可以是24等 | 必须是1 |
| 输出长度 | 取决于测试集大小 | 由recursive_steps指定 |

## 典型输出示例

```
================================================================================
HYBRID RECURSIVE PREDICTION
================================================================================
Recursive steps: 200
Prediction length: 6 (but only use first step)
Sequence length: 90
Only updating OT column, keeping other 42 features as ground truth
================================================================================

Full test data shape: (12168, 43)
Full time marks shape: (12168, 4)

Starting recursive prediction from index 0...
Recursive Prediction: 100%|████████████████| 200/200 [00:45<00:00,  4.42it/s]

Hybrid recursive prediction completed!
Generated 200 predictions
Predictions shape: (200,)
Ground truth shape: (200,)

================================================================================
HYBRID RECURSIVE PREDICTION RESULTS - OT ONLY
================================================================================

Scaled Data Metrics (Normalized):
  MSE:  0.0012345
  MAE:  0.0234567
  RMSE: 0.0351234

Original Scale Metrics:
  MSE:  12.3456789
  MAE:  2.3456789
  RMSE: 3.5123456
================================================================================

✓ Results saved to ./test_results_hybrid_recursive/PPData_<model_id>_hybrid_recursive_results.npz

================================================================================
GENERATING PLOTS FOR OT
================================================================================

[1/4] Plotting full continuous prediction (scaled)...
  ✓ Saved: *_hybrid_recursive_scaled.png
[2/4] Plotting full continuous prediction (original scale)...
  ✓ Saved: *_hybrid_recursive_original.png
[3/4] Plotting side-by-side comparison...
  ✓ Saved: *_hybrid_recursive_comparison.png
[4/4] Plotting error analysis...
  ✓ Saved: *_hybrid_recursive_error_analysis.png

================================================================================
✓ All plots saved to: ./test_results_hybrid_recursive/
================================================================================

✓ Hybrid recursive prediction completed for OT!

Summary:
  - Performed 200 recursive predictions
  - Only OT column was updated with predictions
  - Other 42 features remained as ground truth
  - Used only first step (index 0) of each pred_len=6 prediction
```

## 常见问题

### Q1: 为什么batch_size必须是1？
因为递归预测需要使用前一步的预测结果，所以必须逐步进行，不能批量处理。

### Q2: 为什么只使用第0步预测？
为了保持预测的连续性。如果使用所有6步，会导致预测窗口跳跃，无法形成连续的时间序列。

### Q3: 为什么只更新OT列？
这是混合递归预测的核心策略。在实际应用中，其他传感器数据（如压力、温度等）可以实时获取，只有OT需要预测。

### Q4: 如何增加预测步数？
修改 `--recursive_steps` 参数。例如，`--recursive_steps 500` 将预测500步。

### Q5: 如果我想预测其他列怎么办？
修改脚本中的 `ot_index` 变量为目标列的索引。例如，预测第30列就设置 `ot_index = 30`。

## 注意事项

1. **内存使用**: 长时间递归预测会消耗较多内存，建议根据你的硬件调整 `recursive_steps`
2. **GPU支持**: 建议使用GPU加速，通过 `--use_gpu` 参数启用
3. **检查点路径**: 确保 `checkpoint_path` 指向正确的训练好的模型
4. **数据可用性**: 确保测试数据有足够的长度来支持你指定的 `recursive_steps`

## 扩展应用

### 纯递归预测（所有列都用预测值）
如果你想实现纯递归预测（所有43列都使用预测值），可以修改脚本的这部分：

```python
# 当前的混合递归（只更新OT）
full_data_scaled[ground_truth_idx, ot_index] = pred_ot_step_0[0]

# 改为纯递归（更新所有列）
full_data_scaled[ground_truth_idx, :] = pred_step_0.cpu().numpy()[0, :]
```

### 自定义混合策略
你可以选择性地更新某些列。例如，只保留前10列为真实值，其他列使用预测值：

```python
# 只更新OT列和其他某些列
full_data_scaled[ground_truth_idx, ot_index] = pred_ot_step_0[0]
full_data_scaled[ground_truth_idx, 20:30] = pred_step_0.cpu().numpy()[0, 20:30]
# 其他列保持真实值
```

## 总结

混合递归预测是一种实用的预测策略，特别适合以下场景：
- 只需要预测某个关键指标
- 其他特征可以实时获取
- 需要长期连续预测
- 想要避免误差累积（因为大部分特征保持真实值）

通过这种方式，你可以在保持较高预测精度的同时，实现长期的连续预测。
