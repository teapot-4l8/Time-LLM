# OT特征预测 - 快速指南

## ⚠️ 重要说明

**你的模型是用 `pred_len=6` 训练的，所以现在只能预测6步（6秒）。**

如果想预测更多步数（如30秒、60秒），需要先重新训练模型。

## 当前可以做什么

### ✅ 立即运行（预测6秒）

```bash
bash ./scripts/GPT2_PPData_test_OT.sh
```

这会：
- 使用现有模型预测6秒
- 只显示OT特征的结果
- 生成10个样本的对比图

## 如何预测更多时间步

### 步骤1：训练新模型（预测30秒）

```bash
bash ./scripts/GPT2_PPData_train_long.sh
```

这个训练脚本配置为：
- `pred_len=30`：预测30秒
- `label_len=15`：标签15秒
- `seq_len=90`：输入90秒历史数据

**训练时间**：取决于你的GPU，大约几十分钟到几小时

### 步骤2：找到新模型的checkpoint路径

训练完成后，新模型会保存在：
```
./checkpoints/short_term_forecast_PPData_90_30_GPT2_PPData_ftM_sl90_ll15_pl30_dm16_nh8_el2_dl1_df32_fc3_ebtimeF_Power plant data - longer prediction_0-GPT2-PPData-Long/checkpoint
```

### 步骤3：更新测试脚本

编辑 `scripts/GPT2_PPData_test_OT.sh`：

```bash
# 第9行：更新checkpoint路径
checkpoint_path="./checkpoints/short_term_forecast_PPData_90_30_GPT2_PPData_ftM_sl90_ll15_pl30_.../checkpoint"

# 第14-15行：更新pred_len和label_len
pred_len=30
label_len=15
```

### 步骤4：运行测试

```bash
bash ./scripts/GPT2_PPData_test_OT.sh
```

现在会预测30秒！

## 不同预测长度的配置

编辑 `scripts/GPT2_PPData_train_long.sh`，可以选择：

```bash
# 短期预测（6秒）- 已有模型
pred_len=6
label_len=6

# 中期预测（30秒）- 推荐
pred_len=30
label_len=15

# 长期预测（1分钟）
pred_len=60
label_len=30

# 更长期预测（2分钟）
pred_len=120
label_len=60
```

## 输出示例

运行测试后，会生成：

### 控制台输出
```
TEST RESULTS - OT ONLY
================================================================================

Scaled Data Metrics (Normalized):
  MSE:  0.0123456
  MAE:  0.0987654
  RMSE: 0.1111111

Original Scale Metrics:
  MSE:  123.4567890
  MAE:  9.8765432
  RMSE: 11.1111111

Data shapes:
  Predictions: (282, 6)      # 282个样本，每个6步
  Ground truth: (282, 6)
```

### 图表文件（在 ./test_results/）
1. `*_OT_scaled.png` - 归一化OT预测（10个样本）
2. `*_OT_original.png` - 原始尺度OT预测（10个样本）
3. `*_OT_comparison.png` - 并排对比图
4. `*_OT_error_dist.png` - 误差分布直方图

## 为什么pred_len必须匹配？

神经网络的输出层大小在训练时就固定了：
- 训练时 `pred_len=6` → 输出层形状 `(6, ...)`
- 测试时 `pred_len=30` → 需要输出层形状 `(30, ...)`
- ❌ 形状不匹配 → 无法加载模型

**解决方案**：训练时用什么pred_len，测试时就必须用相同的pred_len。

## 常见问题

### Q1: 我能用6步模型预测30步吗？
**A**: 不能。必须重新训练一个pred_len=30的模型。

### Q2: 训练需要多久？
**A**: 取决于GPU和数据量，通常：
- 1个GPU: 30分钟 - 2小时
- 多个GPU: 更快

### Q3: 能否预测任意长度？
**A**: 理论上可以，但：
- 短期（<60秒）：通常效果好
- 中期（60-300秒）：效果中等
- 长期（>300秒）：效果可能不好

### Q4: 只想看OT，为什么还要用所有43个特征训练？
**A**: 因为其他特征可能包含有用信息来预测OT。多变量模型通常比单变量效果更好。

## 快速参考

### 现在立即测试（6秒预测）
```bash
bash ./scripts/GPT2_PPData_test_OT.sh
```

### 训练30秒预测模型
```bash
# 1. 训练
bash ./scripts/GPT2_PPData_train_long.sh

# 2. 训练完成后，记录checkpoint路径
# 3. 更新 scripts/GPT2_PPData_test_OT.sh 中的：
#    - checkpoint_path
#    - pred_len=30
#    - label_len=15

# 4. 测试
bash ./scripts/GPT2_PPData_test_OT.sh
```

## 文件清单

```
脚本文件:
├── scripts/GPT2_PPData_test_OT.sh          # OT测试脚本（当前pred_len=6）
├── scripts/GPT2_PPData_train_long.sh       # 训练长预测模型（pred_len=30）
└── scripts/GPT2_PPData.sh                  # 原始训练脚本（pred_len=6）

Python文件:
├── run_test_ppdata_ot.py                   # OT专用测试程序
├── run_test_ppdata_simple.py               # 通用测试程序
└── run_main.py                             # 训练程序

输出文件:
└── test_results/
    ├── PPData_PPData_90_6_OT_*.png         # 可视化图表
    └── PPData_PPData_90_6_OT_results.npz   # 数值结果
```

## 下一步

1. **先运行现有测试**（6秒预测）：
   ```bash
   bash ./scripts/GPT2_PPData_test_OT.sh
   ```

2. **查看结果**，如果6秒预测不够，执行步骤3

3. **训练新模型**（30秒或更长）：
   ```bash
   bash ./scripts/GPT2_PPData_train_long.sh
   ```

4. **更新测试脚本**并重新运行
