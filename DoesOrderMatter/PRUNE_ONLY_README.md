# Prune-Only Experiment Setup

## 概述

本文档说明如何运行Wanda剪枝实验，测试不同稀疏度对LLaMA2-7B模型性能的影响。

## 环境要求

已验证在当前`awq`环境中可以直接运行，无需额外创建环境。

### 关键依赖
- PyTorch 2.2.2
- Transformers 4.46.0
- Datasets 4.4.1
- lm-eval 0.3.0
- awq (本地安装)

## 文件结构

```
DoesOrderMatter/
├── prune_only.py           # 主实验脚本
├── test_prune_setup.py     # 环境验证脚本
├── baseline.py             # FP16基线（已完成）
└── results/                # 结果输出目录

wanda/
├── lib/
│   ├── prune.py           # Wanda核心算法（已修改支持pileval）
│   ├── layerwrapper.py    # 激活统计收集
│   ├── data.py            # 数据加载（已添加pileval支持）
│   └── prune_opt.py       # OPT模型支持（保留）
└── README.md              # Wanda原始文档
```

## 实验配置

### 剪枝参数
- **方法**: Wanda (magnitude × activation)
- **类型**: Unstructured (非结构化)
- **稀疏度**: 30%, 40%, 50%, 60%, 70%
- **校准数据集**: pileval (与AWQ保持一致)
- **校准样本数**: 128
- **随机种子**: 42

### 评估指标
- **困惑度 (PPL)**: WikiText2, C4, PTB
- **零样本准确率**: HellaSwag, PIQA, ARC-Easy, BoolQ

## 使用方法

### 步骤1: 环境验证

运行测试脚本确保环境配置正确：

```bash
cd d:\A_Project\llm-awq
python DoesOrderMatter/test_prune_setup.py
```

预期输出：
```
[1/5] Testing Wanda imports... ✅
[2/5] Testing evaluation imports... ✅
[3/5] Checking PyTorch and CUDA... ✅
[4/5] Checking pileval dataset cache... ✅
[5/5] Testing tokenizer... ✅
```

### 步骤2: 运行完整实验

测试所有5个稀疏度等级（30%, 40%, 50%, 60%, 70%）：

```bash
python DoesOrderMatter/prune_only.py
```

预计运行时间：约 2-3 小时（取决于GPU）

### 步骤3: 测试单个稀疏度

如果想先测试单个稀疏度：

```bash
# 测试50%稀疏度
python DoesOrderMatter/prune_only.py --sparsity 0.5

# 测试多个特定稀疏度
python DoesOrderMatter/prune_only.py --sparsity 0.3 0.5 0.7
```

## 输出结果

### 结果文件

实验完成后会生成：

```
DoesOrderMatter/results/
├── prune_only_YYYYMMDD_HHMMSS.json    # 完整JSON结果
└── prune_only_YYYYMMDD_HHMMSS.txt     # 人类可读摘要

DoesOrderMatter/logs/
├── prune_sp30_YYYYMMDD_HHMMSS.log     # 30%稀疏度日志
├── prune_sp40_YYYYMMDD_HHMMSS.log     # 40%稀疏度日志
├── prune_sp50_YYYYMMDD_HHMMSS.log     # 50%稀疏度日志
├── prune_sp60_YYYYMMDD_HHMMSS.log     # 60%稀疏度日志
└── prune_sp70_YYYYMMDD_HHMMSS.log     # 70%稀疏度日志
```

### JSON结果格式

```json
{
  "experiment": "prune_only",
  "model": "meta-llama/Llama-2-7b-hf",
  "timestamp": "2026-01-19T...",
  "config": {
    "sparsity_ratios": [0.3, 0.4, 0.5, 0.6, 0.7],
    "calib_dataset": "pileval",
    "calib_n_samples": 128,
    ...
  },
  "sparsity_results": {
    "sparsity_30": {
      "target_sparsity": 0.3,
      "actual_sparsity": 0.3001,
      "perplexity": {
        "wikitext2": 6.23,
        "c4": 8.15,
        "ptb": 45.32
      },
      "zeroshot": {
        "hellaswag": {"score": 70.5, "metric": "acc_norm"},
        "piqa": {"score": 76.8, "metric": "acc"},
        ...
      }
    },
    ...
  }
}
```

## 代码修改说明

### 1. wanda/lib/data.py
**修改**: 添加了`get_pileval()`函数和对pileval的支持

```python
def get_pileval(nsamples, seed, seqlen, tokenizer):
    """Load pileval dataset for calibration (compatible with AWQ)"""
    # 设置离线模式
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    
    # 加载pileval数据集
    traindata = load_dataset('mit-han-lab/pile-val-backup', split='validation')
    ...
```

### 2. wanda/lib/prune.py
**修改**: `prune_wanda()`函数默认使用pileval而非c4

```python
# 原代码：
dataloader, _ = get_loaders("c4", nsamples=args.nsamples, ...)

# 修改后：
calib_dataset = getattr(args, 'calib_dataset', 'pileval')
dataloader, _ = get_loaders(calib_dataset, nsamples=args.nsamples, ...)
```

## 对比分析

### 与Baseline FP16对比

完成后可以对比：
```python
# baseline_fp16.json
{
  "wikitext2": 5.50,
  "c4": 7.15,
  "ptb": 37.92,
  "hellaswag": 73.0,
  ...
}

# prune_only_*.json (例如 50%稀疏度)
{
  "wikitext2": ~6.8,   # 预期轻微上升
  "c4": ~8.5,
  "ptb": ~48.0,
  "hellaswag": ~68.0,  # 预期轻微下降
  ...
}
```

### 预期趋势

随着稀疏度增加：
- **PPL**: 逐渐上升（困惑度越高，性能越差）
- **准确率**: 逐渐下降
- **30-50%**: 性能下降较小（Wanda的sweet spot）
- **60-70%**: 性能下降明显加速

## 注意事项

### 1. 内存管理
- 每个稀疏度会重新加载模型（避免累积效应）
- 评估后自动清理GPU内存：`torch.cuda.empty_cache()`

### 2. 离线模式
- 默认启用离线模式（`offline_mode=True`）
- 确保pileval数据集已缓存
- 如遇网络问题，检查`HF_DATASETS_OFFLINE`环境变量

### 3. 可复现性
- 固定随机种子：`seed=42`
- 校准数据集固定：pileval
- 评估顺序固定

## 故障排除

### 问题1: ImportError
```
ModuleNotFoundError: No module named 'lib.prune'
```

**解决**: 检查sys.path设置，确保wanda文件夹在路径中

### 问题2: Dataset not found
```
FileNotFoundError: Dataset 'mit-han-lab/pile-val-backup' not found
```

**解决**: 
1. 关闭离线模式暂时下载：`--offline False`
2. 或运行 `python DoesOrderMatter/download_datasets.py`

### 问题3: CUDA OOM
```
RuntimeError: CUDA out of memory
```

**解决**: 
- 减少batch size（在lm_eval_adaptor中）
- 使用更小的模型测试
- 清理其他GPU进程

## 后续步骤

完成Prune-only实验后：

1. **分析结果**: 查看不同稀疏度的性能曲线
2. **选择最佳稀疏度**: 找到性能-效率平衡点
3. **准备Quant-only**: 创建`quant_only.py`进行量化实验
4. **组合实验**: 
   - Pipeline A: Prune → Quantize
   - Pipeline B: Quantize → Prune

## 参考

- Wanda论文: https://arxiv.org/abs/2306.11695
- AWQ论文: https://arxiv.org/abs/2306.00978
- LM-Eval Harness: https://github.com/EleutherAI/lm-evaluation-harness
