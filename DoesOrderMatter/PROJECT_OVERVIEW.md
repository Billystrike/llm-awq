# LLM Compression Order Study - Project Overview

## 🎯 研究目标

**核心问题**: 在LLM压缩中，Wanda剪枝和AWQ量化的应用顺序是否影响最终性能？

## 📊 实验设计

```
Phase 0: Baseline FP16 ✅ 已完成
   ↓
Phase 1: Single Compression ⏳ 进行中
   ├── Prune-only (Wanda)  ← 当前阶段
   └── Quant-only (AWQ)
   ↓
Phase 2: Combined Pipelines
   ├── Pipeline A: Prune → Quantize
   └── Pipeline B: Quantize → Prune
   ↓
Phase 3: Mechanism Analysis
   └── Why does order matter (if it does)?
```

## 📁 项目结构

```
llm-awq/
├── DoesOrderMatter/              # 实验代码
│   ├── baseline.py               ✅ FP16基线评估
│   ├── prune_only.py             🆕 Wanda剪枝实验
│   ├── test_prune_setup.py       🆕 环境验证
│   ├── download_datasets.py      ✅ 数据集下载
│   ├── QUICK_START.md            🆕 快速开始指南
│   ├── PRUNE_ONLY_README.md      🆕 剪枝实验文档
│   ├── results/                  # 结果输出
│   │   └── baseline_fp16.json    ✅ 基线结果
│   └── logs/                     # 实验日志
│
├── wanda/                        # Wanda剪枝库（已精简）
│   ├── lib/
│   │   ├── prune.py              ✅ 核心算法（已修改支持pileval）
│   │   ├── layerwrapper.py       ✅ 激活统计收集
│   │   ├── data.py               ✅ 数据加载（已添加pileval）
│   │   └── prune_opt.py          ✅ OPT模型支持
│   └── README.md                 # Wanda原始文档
│
└── awq/                          # AWQ量化库
    ├── quantize/                 # 量化算法
    └── utils/
        └── calib_data.py         # 校准数据加载
```

## ✅ 已完成工作

### Phase 0: Baseline (2026-01-16)
- ✅ 创建 `baseline.py` 完整评估框架
- ✅ 解决数据集加载问题（离线模式）
- ✅ 成功评估 LLaMA2-7B FP16
- ✅ 结果保存: `baseline_fp16.json`

**Baseline结果**:
```
WikiText2 PPL: 5.50
C4 PPL: 7.15
PTB PPL: 37.92
HellaSwag: 73.0% (acc_norm)
PIQA: 78.3%
ARC-Easy: 69.3%
BoolQ: 71.1%
```

### Phase 1A: Prune-only Setup (2026-01-19)
- ✅ 环境兼容性分析（确认无需新环境）
- ✅ 精简Wanda代码库（删除冗余文件）
- ✅ 修改 `wanda/lib/data.py` 支持pileval
- ✅ 修改 `wanda/lib/prune.py` 默认使用pileval
- ✅ 创建 `prune_only.py` 完整实验脚本
- ✅ 创建 `test_prune_setup.py` 环境验证
- ✅ 创建完整文档

## 🔄 当前状态

**下一步**: 运行Prune-only实验

### 立即执行
```bash
# 1. 验证环境
python DoesOrderMatter/test_prune_setup.py

# 2. 运行实验
python DoesOrderMatter/prune_only.py
```

## 🎯 实验参数总结

### 统一配置（所有实验一致）
| 参数 | 值 | 说明 |
|------|-----|------|
| 模型 | LLaMA2-7B | meta-llama/Llama-2-7b-hf |
| 随机种子 | 42 | 确保可复现性 |
| Calibration数据 | pileval | mit-han-lab/pile-val-backup |
| Calib样本数 | 128 | 校准数据量 |
| PPL数据集 | WikiText2, C4, PTB | 困惑度评估 |
| Zero-shot任务 | HellaSwag, PIQA, ARC-Easy, BoolQ | 零样本准确率 |

### Prune-only特定参数
| 参数 | 值 |
|------|-----|
| 剪枝方法 | Wanda |
| 稀疏度 | 30%, 40%, 50%, 60%, 70% |
| 剪枝类型 | Unstructured |
| Calib序列长度 | 512 (由model.seqlen决定) |

### Quant-only特定参数（待创建）
| 参数 | 值 |
|------|-----|
| 量化方法 | AWQ |
| 位宽 | W4A16 |
| Group Size | 128 |
| 版本 | GEMM |

## 📊 预期结果对比

### 性能退化预期

| 配置 | WikiText2 PPL | HellaSwag | 参数量 | 备注 |
|------|---------------|-----------|--------|------|
| **FP16 (Baseline)** | 5.50 | 73.0% | 100% | ✅ 已完成 |
| **Prune 30%** | ~5.8 | ~72.0% | 70% | 轻微下降 |
| **Prune 50%** | ~6.8 | ~68.0% | 50% | 中等下降 |
| **Prune 70%** | ~9.5 | ~58.0% | 30% | 显著下降 |
| **Quant W4A16** | ~5.7 | ~72.5% | 100% (4bit) | 待测试 |
| **Prune50→Quant** | ? | ? | 50% (4bit) | Pipeline A |
| **Quant→Prune50** | ? | ? | 50% (4bit) | Pipeline B |

## 🔬 关键研究问题

1. **性能对比**: Prune-only vs Quant-only 哪个更好？
2. **顺序效应**: Pipeline A vs Pipeline B 性能是否有差异？
3. **机制分析**: 如果有差异，是什么导致的？
   - 激活分布变化
   - 权重重要性重排
   - 通道间依赖关系

## 📈 评估指标

### 性能指标
- **困惑度 (PPL)**: 越低越好（生成质量）
- **准确率 (Acc)**: 越高越好（任务性能）

### 效率指标
- **参数量**: 模型大小
- **推理速度**: 前向传播时间
- **内存占用**: GPU显存需求

## 🛠️ 技术栈

- **深度学习框架**: PyTorch 2.2.2
- **模型库**: Transformers 4.46.0
- **评估工具**: lm-eval 0.3.0
- **数据集**: HuggingFace Datasets 4.4.1
- **剪枝**: Wanda (magnitude × activation)
- **量化**: AWQ (activation-aware weight quantization)

## 📝 实验日志规范

### 文件命名
- 基线: `baseline_fp16.json`
- 剪枝: `prune_only_YYYYMMDD_HHMMSS.json`
- 量化: `quant_only_YYYYMMDD_HHMMSS.json`
- 组合A: `prune_then_quant_YYYYMMDD_HHMMSS.json`
- 组合B: `quant_then_prune_YYYYMMDD_HHMMSS.json`

### 结果格式
所有实验结果统一使用JSON格式，包含：
```json
{
  "experiment": "实验类型",
  "model": "模型名称",
  "timestamp": "时间戳",
  "config": {...},
  "results": {
    "perplexity": {...},
    "zeroshot": {...}
  }
}
```

## 🎓 科学严谨性保证

### 可复现性
- ✅ 固定随机种子
- ✅ 统一校准数据
- ✅ 相同评估协议
- ✅ 详细参数记录

### 公平性
- ✅ 相同基准模型
- ✅ 相同评估数据
- ✅ 相同硬件环境
- ✅ 独立重复实验

### 完整性
- ✅ 多个评估指标
- ✅ 多个稀疏度/精度
- ✅ 详细日志记录
- ✅ 中间结果保存

## 📚 参考文献

### 核心论文
1. **Wanda**: Sun et al., "A Simple and Effective Pruning Approach for Large Language Models", 2023
   - https://arxiv.org/abs/2306.11695

2. **AWQ**: Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration", 2023
   - https://arxiv.org/abs/2306.00978

3. **LLaMA2**: Touvron et al., "Llama 2: Open Foundation and Fine-Tuned Chat Models", 2023
   - https://arxiv.org/abs/2307.09288

## 🚀 下一步行动

### 立即执行（Phase 1A）
1. ✅ 运行环境验证: `python DoesOrderMatter/test_prune_setup.py`
2. ⏳ 运行Prune-only: `python DoesOrderMatter/prune_only.py`
3. ⏳ 分析结果，选择最佳稀疏度

### 短期计划（Phase 1B）
1. 创建 `quant_only.py`
2. 测试AWQ量化（W4A16）
3. 对比Prune-only vs Quant-only

### 中期计划（Phase 2）
1. 实现Pipeline A: Prune → Quantize
2. 实现Pipeline B: Quantize → Prune
3. 对比两种Pipeline性能差异

### 长期计划（Phase 3）
1. 激活分布可视化
2. 权重重要性分析
3. 通道依赖性研究
4. 撰写研究报告

## 💡 重要提醒

- 每次实验前运行 `test_prune_setup.py` 验证环境
- 保持所有实验的calibration配置一致（pileval, 128 samples）
- 每个实验独立加载模型，避免累积效应
- 定期备份results文件夹
- 详细记录任何异常或意外结果

---

**项目状态**: Phase 1A - 准备运行Prune-only实验  
**最后更新**: 2026-01-19  
**负责人**: Research Team
