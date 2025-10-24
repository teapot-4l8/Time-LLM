# PPData测试脚本使用说明

## 文件说明

### 1. `run_test_ppdata_simple.py`
简化的测试脚本，支持保存和加载测试结果

### 2. `scripts/GPT2_PPData_test_simple.sh`
测试启动脚本

## 主要功能

### ✅ 模型推理
- 加载训练好的LLAMA模型
- 在测试集上进行预测
- 自动处理bfloat16类型转换
- 支持GPU/CPU运行

### ✅ 数据处理
- 正确处理多变量预测（43个特征）
- 自动重塑数据维度
- 归一化数据反变换

### ✅ 结果保存/加载（新功能）
- **自动保存推理结果**：避免重复运行模型
- **快速加载已有结果**：直接生成可视化图表
- 两种结果文件：
  - `*_inference.npz`：原始推理结果（用于快速加载）
  - `*_results.npz`：完整结果（包含所有指标和反归一化数据）

### ✅ 可视化
生成6种对比图：
1. 多特征归一化数据对比
2. 多特征原始尺度数据对比
3. 多样本归一化数据对比
4. 多样本原始尺度数据对比
5. 归一化vs原始尺度并排对比
6. 误差分布直方图

### ✅ 评估指标
- 归一化数据：MSE, MAE, RMSE
- 原始尺度数据：MSE, MAE, RMSE

## 使用方法

### 第一次运行（完整推理）

```bash
bash ./scripts/GPT2_PPData_test_simple.sh
```

**步骤**：
1. 加载模型和数据
2. 运行推理（~2分钟）
3. 保存原始推理结果到 `test_results/PPData_PPData_90_6_6_inference.npz`
4. 数据重塑和反归一化
5. 计算指标
6. 生成6张可视化图表
7. 保存完整结果到 `test_results/PPData_PPData_90_6_6_results.npz`

### 后续运行（加载已有结果）

如果你已经运行过一次测试，想要重新生成图表或查看结果，可以直接加载已保存的推理结果：

```bash
python run_test_ppdata_simple.py \
  --task_name short_term_forecast \
  --root_path ./dataset/ \
  --data_path PPData.csv \
  --model_id PPData_90_6 \
  --model GPT2 \
  --data PPData \
  --features M \
  --freq 's' \
  --seq_len 90 \
  --label_len 6 \
  --pred_len 6 \
  --factor 3 \
  --enc_in 43 \
  --dec_in 43 \
  --c_out 43 \
  --llm_model LLAMA \
  --llm_dim 4096 \
  --des 'Power plant data' \
  --llm_layers 6 \
  --d_model 16 \
  --d_ff 32 \
  --batch_size 24 \
  --model_comment GPT2-PPData \
  --checkpoint_path "./checkpoints/your_checkpoint_path/checkpoint" \
  --plot_samples 5 \
  --plot_features 3 \
  --load_results ./test_results/PPData_PPData_90_6_6_inference.npz
```

**优点**：
- ⚡ 跳过模型加载和推理（节省~3-4分钟）
- 🔄 可以调整可视化参数（`--plot_samples`, `--plot_features`）
- 📊 快速重新生成图表

## 输出文件

### `./test_results/` 目录结构

```
test_results/
├── PPData_PPData_90_6_6_inference.npz          # 原始推理结果（加载用）
├── PPData_PPData_90_6_6_results.npz            # 完整结果（所有指标）
├── PPData_PPData_90_6_scaled_features.png      # 图1：归一化多特征
├── PPData_PPData_90_6_original_features.png    # 图2：原始尺度多特征
├── PPData_PPData_90_6_scaled_samples.png       # 图3：归一化多样本
├── PPData_PPData_90_6_original_samples.png     # 图4：原始尺度多样本
├── PPData_PPData_90_6_comparison.png           # 图5：并排对比
└── PPData_PPData_90_6_error_distribution.png   # 图6：误差分布
```

## 可调参数

在 `scripts/GPT2_PPData_test_simple.sh` 中：

- `checkpoint_path`: 模型checkpoint路径（必须修改）
- `--plot_samples 5`: 绘制样本数量
- `--plot_features 3`: 绘制特征数量
- `--use_gpu`: 使用GPU（删除此行则使用CPU）
- `--batch_size 24`: 批次大小

## Bug修复说明

### 1. 修复reshape错误
**问题**：数据维度计算错误，`12168 ≠ 43 × 282`
**解决**：
- 添加数据验证和调整
- 正确的reshape操作：squeeze → reshape → transpose
- 详细的形状打印信息

### 2. 添加保存/加载功能
**新增参数**：
- `--load_results`: 加载已保存的推理结果
- `--save_results`: 自动保存推理结果（默认开启）

**工作流程**：
```
首次运行: 模型推理 → 保存inference.npz → 处理 → 保存results.npz → 生成图表
后续运行: 加载inference.npz → 处理 → 保存results.npz → 生成图表
```

## 故障排查

### 问题1：reshape错误
**症状**：`ValueError: cannot reshape array`
**解决**：脚本会自动调整数据大小，如果看到警告信息，检查数据完整性

### 问题2：dtype不匹配
**症状**：`RuntimeError: Input type ... and weight type ... should be the same`
**解决**：确保使用 `--use_gpu` 参数（GPU上会自动转换为bfloat16）

### 问题3：模型架构不匹配
**症状**：`Missing key(s) in state_dict`
**解决**：确保 `--llm_model LLAMA` 参数正确（不是GPT2）

## 性能提示

- **首次运行时间**：~5-6分钟（模型加载2分钟 + 推理2分钟 + 处理1分钟）
- **加载结果时间**：~1分钟（仅数据处理和可视化）
- **内存需求**：~8GB GPU / ~16GB RAM（CPU模式）

## 示例输出

```
================================================================================
TEST RESULTS
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
  Predictions: (283, 6, 43)
  Ground truth: (283, 6, 43)
================================================================================

✓ Final results saved to ./test_results/PPData_PPData_90_6_6_results.npz
  - To reload inference results next time, use: --load_results ./test_results/PPData_PPData_90_6_6_inference.npz
```

## 更新日志

### v2.0 (当前版本)
- ✅ 修复多变量数据reshape错误
- ✅ 添加推理结果保存/加载功能
- ✅ 改进错误处理和数据验证
- ✅ 添加详细的形状信息打印

### v1.0
- 初始版本
- 基本推理和可视化功能
