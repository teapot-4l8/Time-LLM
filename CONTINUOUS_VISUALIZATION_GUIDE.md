# 连续时间序列可视化说明

## 问题解决

**你的问题**：图表上只显示6个点（因为每个预测窗口只有6步），看不到长时间的连续趋势。

**解决方案**：我添加了连续时间序列可视化功能，可以显示200步（或更多）的连续预测！

## 新增功能

### 之前（4张图）
1. 多个预测窗口（每个6步）
2. 原始尺度预测窗口
3. 并排对比
4. 误差分布

### 现在（6张图）
1-4. 保留原有图表
5. **连续时间序列图**（新！）- 显示200步连续预测
6. **详细预测窗口图**（新！）- 显示3个连续预测窗口的所有细节

## 连续时间序列的工作原理

### 你有282个样本
每个样本都是一个6步的预测：
```
样本1: [t0, t1, t2, t3, t4, t5]
样本2: [t6, t7, t8, t9, t10, t11]
样本3: [t12, t13, t14, t15, t16, t17]
...
```

### 连续可视化
我们取每个样本的第一步，连接成一条长线：
```
连续线: [t0, t6, t12, t18, ..., t200]
```

这样你可以看到200步（200秒）的连续预测趋势！

## 使用方法

### 运行测试
```bash
bash ./scripts/GPT2_PPData_test_OT.sh
```

### 调整连续时间步数

编辑 `scripts/GPT2_PPData_test_OT.sh` 第24行：

```bash
continuous_steps=200      # 显示200秒的连续预测

# 你可以改为：
continuous_steps=100      # 100秒
continuous_steps=300      # 5分钟
continuous_steps=600      # 10分钟
```

⚠️ **注意**：`continuous_steps` 必须是 `pred_len` 的倍数
- 当前 `pred_len=6`
- 所以 `continuous_steps` 应该是 6, 12, 18, 24, ..., 200, 300 等

## 输出图表

### 图5：连续时间序列图
`*_OT_continuous.png`

这是一张**大宽图**，显示：
- 上半部分：归一化数据的连续预测（200步）
- 下半部分：原始尺度数据的连续预测（200步）
- 阴影区域：预测误差

**特点**：
- 可以看到长期趋势
- 适合观察整体预测表现
- 一眼看出模型是否跟踪实际值

### 图6：详细预测窗口图
`*_OT_detailed.png`

显示3个连续的预测窗口，每个都是完整的6步：
- 左列：归一化数据
- 右列：原始尺度数据
- 每个窗口都有marker标记

**特点**：
- 可以看到每一步的预测细节
- 适合观察短期预测质量
- 显示预测的逐步变化

## 示例输出

### 控制台输出
```
GENERATING PLOTS FOR OT
================================================================================
Plotting 10 samples...

[1/6] Plotting scaled predictions for multiple samples...
  ✓ Saved: ./test_results/PPData_PPData_90_6_OT_OT_scaled.png

[2/6] Plotting original scale predictions for multiple samples...
  ✓ Saved: ./test_results/PPData_PPData_90_6_OT_OT_original.png

[3/6] Plotting side-by-side comparison...
  ✓ Saved: ./test_results/PPData_PPData_90_6_OT_OT_comparison.png

[4/6] Plotting error distributions...
  ✓ Saved: ./test_results/PPData_PPData_90_6_OT_OT_error_dist.png

[5/6] Plotting continuous time series (200 steps)...
  ✓ Saved: ./test_results/PPData_PPData_90_6_OT_OT_continuous.png  # 新！

[6/6] Plotting detailed multi-step predictions...
  ✓ Saved: ./test_results/PPData_PPData_90_6_OT_OT_detailed.png    # 新！
```

### 文件列表
```
test_results/
├── PPData_PPData_90_6_OT_OT_scaled.png          # 10个预测窗口（归一化）
├── PPData_PPData_90_6_OT_OT_original.png        # 10个预测窗口（原始）
├── PPData_PPData_90_6_OT_OT_comparison.png      # 并排对比
├── PPData_PPData_90_6_OT_OT_error_dist.png      # 误差分布
├── PPData_PPData_90_6_OT_OT_continuous.png      # 连续时间序列（200步）✨新
├── PPData_PPData_90_6_OT_OT_detailed.png        # 详细窗口（3个）✨新
├── PPData_PPData_90_6_OT_inference.npz          # 推理数据
└── PPData_PPData_90_6_OT_OT_results.npz         # 结果数据
```

## 参数说明

### 在 `scripts/GPT2_PPData_test_OT.sh` 中

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `pred_len` | 6 | 每个预测窗口的步数（必须匹配训练） |
| `label_len` | 6 | 标签长度（必须匹配训练） |
| `plot_samples` | 10 | 图1-2中显示多少个预测窗口 |
| `continuous_steps` | 200 | 图5中显示多少个连续时间步 |

## 理解数据

### 你的测试集
- 总样本数：282个
- 每个样本：6步预测
- 总时间点：282 × 6 = 1692步（但有重叠）

### 连续可视化
- 取前33个样本（200 ÷ 6 ≈ 33）
- 每个取第1步
- 形成200步的连续线

### 为什么不是所有282个样本？
因为我们只可视化 `continuous_steps` 指定的长度。如果你想看更多：
```bash
continuous_steps=1680     # 几乎所有数据（282 × 6）
```

## 可视化对比

### 图1-2：多个独立窗口
```
窗口1: ------
窗口2:       ------
窗口3:             ------
...
```
- 看不出连续性
- 适合对比不同窗口的预测质量

### 图5：连续时间序列
```
连续线: ----------------------------------
```
- 可以看出长期趋势
- 适合评估整体跟踪能力

### 图6：详细窗口
```
窗口1: •--•--•--•--•--•
窗口2:                •--•--•--•--•--•
窗口3:                               •--•--•--•--•--•
```
- 每个点都清晰可见
- 适合分析预测的逐步变化

## 快速开始

### 1. 运行测试
```bash
bash ./scripts/GPT2_PPData_test_OT.sh
```

### 2. 查看图表
```bash
# 在 test_results/ 目录查看图片
# 重点关注这两张新图：
# - *_OT_continuous.png   (连续200步)
# - *_OT_detailed.png     (3个详细窗口)
```

### 3. 调整可视化长度（可选）
编辑 `scripts/GPT2_PPData_test_OT.sh`：
```bash
continuous_steps=300   # 看更长的时间（5分钟）
plot_samples=20        # 显示更多窗口
```

## 常见问题

### Q1: 为什么连续图只有33个样本（200步 ÷ 6）？
**A**: 因为 `continuous_steps=200`。如果想显示所有282个样本：
```bash
continuous_steps=1692  # 282 × 6
```

### Q2: 连续图是如何生成的？
**A**: 从每个预测窗口取第一步，连接成连续线。这模拟了"滚动预测"的场景。

### Q3: 我可以看到所有6步的连续预测吗？
**A**: 图6（详细窗口图）显示了3个窗口的完整6步。如果想看更多窗口，修改代码中的 `num_detailed = min(3, ...)` 改为更大的数字。

### Q4: 哪张图最有用？
**A**:
- **图5（连续图）**：看长期趋势和整体表现
- **图6（详细图）**：看短期预测的准确性
- **图4（误差分布）**：看预测误差的统计特性

## 更新日志

### v2.0（当前）
- ✅ 添加连续时间序列图（200步）
- ✅ 添加详细预测窗口图（3个窗口）
- ✅ 可配置 `continuous_steps` 参数
- ✅ 改进图表样式和填充区域

### v1.0
- 4张基本图表
- OT特征提取
- 归一化和原始尺度对比
