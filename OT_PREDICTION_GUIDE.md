# OT特征预测测试指南

## 概述

你现在有专门用于**只预测和可视化OT特征**的测试脚本，并且可以**自定义预测的时间步数**。

## 文件说明

### 1. `run_test_ppdata_ot.py`
- 专门用于OT特征预测的测试脚本
- 从所有43个特征中只提取OT进行可视化
- 支持可配置的预测长度

### 2. `scripts/GPT2_PPData_test_OT.sh`
- OT预测的启动脚本
- 默认配置：预测30秒（`pred_len=30`）
- 可以轻松修改预测长度

## 快速开始

### 运行测试（使用现有的多变量模型）

```bash
bash ./scripts/GPT2_PPData_test_OT.sh
```

## 调整预测时间步数

编辑 `scripts/GPT2_PPData_test_OT.sh` 文件，修改第18-19行：

```bash
# 预测更多时间步
pred_len=30      # 当前：30秒
label_len=15     # 通常是pred_len的一半

# 示例配置：
# pred_len=60   # 预测1分钟
# pred_len=120  # 预测2分钟
# pred_len=300  # 预测5分钟
```

### 注意事项

⚠️ **重要**：你的模型是用 `pred_len=6` 训练的。如果你想预测更长的时间步（如30、60步），需要：

**选项A：使用现有模型（可能效果不佳）**
- 直接运行，但模型可能在长预测上表现不好
- 适合快速测试

**选项B：重新训练模型（推荐）**
- 需要先用更长的`pred_len`重新训练模型
- 参考下面的"重新训练"部分

## 当前配置 vs 你想要的

### 当前训练配置
```bash
--seq_len 90        # 输入：90秒的历史数据
--label_len 6       # 标签：6秒
--pred_len 6        # 预测：6秒
--features M        # 多变量：使用所有43个特征
--target OT         # 目标特征名称（但实际预测所有43个）
```

### OT测试配置
```bash
--seq_len 90        # 输入：90秒的历史数据
--label_len 15      # 标签：15秒（可调）
--pred_len 30       # 预测：30秒（可调）
--features M        # 仍然使用多变量模型
--target OT         # 只可视化OT特征
```

## 输出结果

运行后会生成：

### 数据文件
```
test_results/
├── PPData_PPData_90_30_OT_inference.npz    # 原始推理结果
└── PPData_PPData_90_30_OT_results.npz      # OT特征完整结果
```

### 可视化图表（4张）
```
1. *_OT_scaled.png          # 多个样本的归一化OT预测
2. *_OT_original.png        # 多个样本的原始尺度OT预测
3. *_OT_comparison.png      # 归一化 vs 原始尺度并排对比
4. *_OT_error_dist.png      # 误差分布直方图
```

### 评估指标
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
  Predictions: (282, 30)     # 282个样本，每个30步
  Ground truth: (282, 30)
================================================================================
```

## 重新训练（用于更长预测）

如果你想要预测更长的时间步并获得好的效果，需要重新训练模型。

### 创建新的训练脚本

创建 `scripts/GPT2_PPData_train_long.sh`：

```bash
model_name=GPT2
train_epochs=10
learning_rate=0.001

batch_size=24
d_model=16
d_ff=32

comment='GPT2-PPData-Long'

# 新配置：预测更长时间
seq_len=90      # 输入90秒历史
label_len=15    # 标签15秒
pred_len=30     # 预测30秒（或60、120等）

accelerate launch --mixed_precision bf16  run_main.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path PPData.csv \
  --model_id PPData_${seq_len}_${pred_len} \
  --model $model_name \
  --data PPData \
  --features M \
  --target OT \
  --freq 's' \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $pred_len \
  --factor 3 \
  --enc_in 43 \
  --dec_in 43 \
  --c_out 43 \
  --llm_dim 4096 \
  --des 'Power plant data - longer prediction' \
  --itr 1 \
  --llm_layers 6 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --model_comment $comment
```

然后运行：
```bash
bash ./scripts/GPT2_PPData_train_long.sh
```

训练完成后，更新测试脚本中的`checkpoint_path`指向新模型。

## 只预测单个特征（S模式）

如果你想要**只用OT特征训练和预测**（单变量），需要创建新的训练配置：

### 单变量训练脚本

创建 `scripts/GPT2_PPData_train_OT_only.sh`：

```bash
model_name=GPT2
train_epochs=10
learning_rate=0.001

batch_size=24
d_model=16
d_ff=32

comment='GPT2-PPData-OT-Univariate'

accelerate launch --mixed_precision bf16  run_main.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path PPData.csv \
  --model_id PPData_OT_90_30 \
  --model $model_name \
  --data PPData \
  --features S \              # S = 单变量
  --target OT \               # 只使用OT特征
  --freq 's' \
  --seq_len 90 \
  --label_len 15 \
  --pred_len 30 \
  --factor 3 \
  --enc_in 1 \                # 只有1个输入特征
  --dec_in 1 \                # 只有1个解码特征
  --c_out 1 \                 # 只输出1个特征
  --llm_dim 4096 \
  --des 'Power plant OT univariate' \
  --itr 1 \
  --llm_layers 6 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --model_comment $comment
```

## 可调参数

在 `scripts/GPT2_PPData_test_OT.sh` 中：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `pred_len` | 30 | 预测时间步数（秒） |
| `label_len` | 15 | 标签长度（通常是pred_len的一半） |
| `--plot_samples` | 10 | 绘制的样本数量 |
| `--use_gpu` | 开启 | 使用GPU |

## 时间步数说明

由于 `freq='s'`（秒级），时间步直接对应秒数：

- `pred_len=6` → 预测6秒
- `pred_len=30` → 预测30秒
- `pred_len=60` → 预测1分钟
- `pred_len=120` → 预测2分钟
- `pred_len=300` → 预测5分钟

## 比较：当前方案 vs 其他方案

### 当前方案：使用多变量模型，只可视化OT
✅ **优点**：
- 立即可用，不需要重新训练
- 模型利用了所有43个特征的信息
- 可能效果更好（如果其他特征有助于预测OT）

❌ **缺点**：
- pred_len不匹配（训练6步，测试30步）
- 需要处理所有43个特征的数据

### 方案B：重新训练长预测多变量模型
✅ **优点**：
- pred_len匹配
- 模型针对长预测优化
- 仍然利用所有特征

❌ **缺点**：
- 需要重新训练（耗时）

### 方案C：训练单变量OT模型
✅ **优点**：
- 简单直接
- 更快的训练和推理
- 只关注OT特征

❌ **缺点**：
- 丢失其他特征的信息
- 可能效果不如多变量

## 推荐流程

1. **先运行当前脚本**（快速测试）
   ```bash
   bash ./scripts/GPT2_PPData_test_OT.sh
   ```

2. **查看结果**
   - 检查预测质量
   - 如果效果可接受，就使用这个方案

3. **如果效果不好**
   - 重新训练一个 `pred_len=30`（或更长）的多变量模型
   - 然后再运行测试

4. **如果想要更简单的模型**
   - 训练单变量OT模型（features='S'）

## 故障排查

### 问题：预测效果不好
**原因**：模型用6步训练，测试用30步
**解决**：
1. 先尝试 `pred_len=6` 看效果
2. 如果6步效果好，30步不好 → 需要重新训练

### 问题：找不到OT特征
**检查**：`args.target='OT'` 是否正确
**验证**：OT应该是第43列（最后一列）

## 更新日志

### v1.0
- ✅ 专门的OT预测脚本
- ✅ 可配置预测长度
- ✅ 4种可视化图表
- ✅ 只显示OT的指标
- ✅ 改进的图表样式
