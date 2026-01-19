# Prune-Only Experiment Quick Start Guide

## 🚀 快速开始

### 1. 验证环境（必须先运行）
```bash
python DoesOrderMatter/test_prune_setup.py
```

### 2. 运行实验

#### 选项A: 完整实验（推荐）
运行所有5个稀疏度（30%, 40%, 50%, 60%, 70%）
```bash
python DoesOrderMatter/prune_only.py
```

#### 选项B: 单个稀疏度测试（快速验证）
```bash
# 测试50%稀疏度
python DoesOrderMatter/prune_only.py --sparsity 0.5
```

#### 选项C: 自定义稀疏度
```bash
# 只测试30%和50%
python DoesOrderMatter/prune_only.py --sparsity 0.3 0.5
```

## 📊 查看结果

结果保存在：
- `DoesOrderMatter/results/prune_only_*.json` - 完整数据
- `DoesOrderMatter/results/prune_only_*.txt` - 可读摘要
- `DoesOrderMatter/logs/prune_sp*_*.log` - 详细日志

## ⏱️ 预计时间

- **单个稀疏度**: ~30-40分钟
- **完整实验(5个稀疏度)**: ~2.5-3.5小时

## 🎯 实验目标

评估Wanda剪枝在不同稀疏度下的性能：

| 稀疏度 | 参数保留 | 预期性能下降 |
|--------|----------|--------------|
| 30% | 70%参数 | 轻微 (~1-2%) |
| 40% | 60%参数 | 小 (~3-5%) |
| 50% | 50%参数 | 中等 (~5-8%) |
| 60% | 40%参数 | 明显 (~10-15%) |
| 70% | 30%参数 | 显著 (~20%+) |

## 📈 与Baseline对比

完成后可以对比`baseline_fp16.json`和`prune_only_*.json`中的结果。

### Baseline FP16结果（参考）
```
WikiText2 PPL: 5.50
C4 PPL: 7.15
PTB PPL: 37.92
HellaSwag: 73.0%
PIQA: 78.3%
ARC-Easy: 69.3%
BoolQ: 71.1%
```

## 🔧 核心技术细节

- **Calibration**: pileval数据集（128样本，与AWQ一致）
- **Pruning Metric**: |W| × ||X||₂ (权重绝对值 × 激活范数)
- **Granularity**: Per-output channel (逐输出通道)
- **Evaluation**: 每个稀疏度重新加载模型，避免累积效应

## ⚠️ 常见问题

### Q: 如何确认pileval数据集已缓存？
```bash
python DoesOrderMatter/test_prune_setup.py
```
看到 "✅ Pileval dataset loaded" 即表示正常。

### Q: 可以中断后继续吗？
不可以。每个稀疏度是独立的，中断需要重新开始。
建议先用单个稀疏度测试：`--sparsity 0.5`

### Q: GPU内存不足怎么办？
- LLaMA2-7B FP16需要约14GB显存
- 如果不够，考虑使用更小的模型或设置`CUDA_VISIBLE_DEVICES`

### Q: 离线模式失败？
1. 确保datasets已缓存：`ls ~/.cache/huggingface/datasets/`
2. 临时关闭离线模式：`--offline False`（需要网络）

## 📝 实验检查清单

运行前确认：
- [ ] 环境验证通过（test_prune_setup.py）
- [ ] GPU可用且显存充足（nvidia-smi）
- [ ] pileval数据集已缓存
- [ ] baseline_fp16.json已存在（用于对比）

运行后检查：
- [ ] 每个稀疏度都有对应的日志文件
- [ ] prune_only_*.json包含所有稀疏度结果
- [ ] prune_only_*.txt摘要可读

## 🔄 下一步

完成Prune-only后：
1. 分析性能-稀疏度曲线
2. 创建`quant_only.py`（AWQ量化实验）
3. 组合实验：Prune→Quant vs Quant→Prune

---

**需要帮助？** 查看详细文档：`PRUNE_ONLY_README.md`
