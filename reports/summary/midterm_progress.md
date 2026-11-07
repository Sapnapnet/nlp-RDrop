# R-Drop / MRPC 中期实验进度记录

记录日期：2026-11-07  
状态：正式训练已完成；中期结果基于 MRPC validation/dev，不代表独立 test 结果。

## 原文目标位置

原文：[R-Drop: Regularized Dropout for Neural Networks, NeurIPS 2021 PDF](../../../NeurIPS-2021-r-drop-regularized-dropout-for-neural-networks-Paper.pdf)。重点位置：

- 第 1 页摘要与引言：论文问题背景和方法概述。
- 第 2-3 页第 2.1 节及公式 (2)-(4)：同一样本经两种 dropout 子模型前向，以双向 KL 一致性损失约束输出，并与两次 NLL/交叉熵合并。
- 第 4 页算法 1 与第 2.2 节：训练流程和计算成本说明。论文指出每步计算成本增加；其算法描述将重复输入拼接后在同一批次前向。
- 第 6 页第 3.3 节及表 3：GLUE 语言理解实验背景。

本组复现聚焦 GLUE MRPC 二分类，使用 accuracy、正类 F1、不同随机种子的波动和训练时间进行比较。

## 本组设置差异

- 编码器固定为 `bert-base-cased`，MRPC 使用完整训练集 3,668 条和 validation/dev 408 条；本实验不使用 GLUE test split。数据处理、来源和行数核验见[输入协议](../../configs/input_protocol.json)与[数据审计](../../data/mrpc/audit.json)。
- 普通微调使用一次随机前向和一次交叉熵；R-Drop 使用两次独立随机前向，损失为 `0.5 * (CE1 + CE2) + alpha * 0.5 * (KL12 + KL21)`。损失按样本取均值，KL 在类别维求和后按 batch 取均值；论文原文的 NLL 写法为两次负对数似然之和。本组采用均值化形式并在[训练协议](../../configs/training_protocol_v1.json)中明确记录，因此系数尺度和原文未必可直接等同。
- 比较条件为普通微调，以及 R-Drop `alpha = 0、0.5、1.0`；种子为 `13、21、42`。每次训练均为 2,296 次优化器更新、同一训练样本访问计划和相同编码器初始化种子；R-Drop 每个 microbatch 做两次编码器前向。
- 本地交付包含 MRPC parquet/jsonl 数据和 BERT tokenizer/config；预训练权重约 436 MB，记录在资产清单中，但未包含在此项目目录/交付包内。复现实验需要另行取得该权重。详见[资产来源与哈希清单](../../provenance/assets.json)及[项目交付说明](../../README.md)。

## 已有并列结果

以下为三个种子的均值 ± 样本标准差（ddof=1）。分数和时间均来自各自正式运行汇总；时间为训练循环 wall time。

| 条件 | Accuracy (%) | F1 (%) | 训练时间 (s) | 更新步数/种子 | 编码器前向/种子 |
|---|---:|---:|---:|---:|---:|
| 普通微调 | 83.0882 ± 1.2255 | 88.2273 ± 0.5931 | 332.65 ± 1.26 | 2,296 | 4,583 |
| R-Drop α=0 | 83.2516 ± 1.4976 | 88.3561 ± 0.8016 | 607.02 ± 27.56 | 2,296 | 9,166 |
| R-Drop α=0.5 | 83.7418 ± 0.8608 | 88.7046 ± 0.3188 | 584.93 ± 0.87 | 2,296 | 9,166 |
| R-Drop α=1.0 | 83.2516 ± 1.0204 | 88.2993 ± 0.4086 | 590.78 ± 5.47 | 2,296 | 9,166 |

α=0.5 相对同种子普通微调的配对平均差为 Accuracy `+0.6536` 个百分点、F1 `+0.4773` 个百分点；平均训练时间约为普通微调的 `1.76` 倍。当前 n=3 的结果用于描述本组实验，不构成统计显著性结论。编码器前向次数和样本通过量是实际工作量计数，不是精确 FLOPs；同时报告训练 wall time，避免将性能变化描述为“免费提升”。

汇总证据：[普通微调三种子汇总](baseline_results.md) · [R-Drop 九组正式结果](../rdrop_formal/batch_20261104T082306144395Z/rdrop_results.md) · [R-Drop 机器可读结果](../rdrop_formal/batch_20261104T082306144395Z/rdrop_results.json)

## 本组逐条运行输出

每行链接至该次运行的原始 `result.json`、validation/dev 预测和逐步训练更新记录。普通微调 seed 13 与 seeds 21/42 的证据保存在不同批次目录。

| 条件 | Seed | 结果 | Dev 预测 | 训练更新记录 |
|---|---:|---|---|---|
| 普通微调 | 13 | [result.json](../baseline/seed13_20261104_122810_CN_kwnuwi7m/result.json) | [predictions_dev.jsonl](../baseline/seed13_20261104_122810_CN_kwnuwi7m/predictions_dev.jsonl) | [training_updates.jsonl](../baseline/seed13_20261104_122810_CN_kwnuwi7m/training_updates.jsonl) |
| 普通微调 | 21 | [result.json](../baseline_batch/step6_baseline_seeds_21_42/seed21/result.json) | [predictions_dev.jsonl](../baseline_batch/step6_baseline_seeds_21_42/seed21/predictions_dev.jsonl) | [training_updates.jsonl](../baseline_batch/step6_baseline_seeds_21_42/seed21/training_updates.jsonl) |
| 普通微调 | 42 | [result.json](../baseline_batch/step6_baseline_seeds_21_42/seed42/result.json) | [predictions_dev.jsonl](../baseline_batch/step6_baseline_seeds_21_42/seed42/predictions_dev.jsonl) | [training_updates.jsonl](../baseline_batch/step6_baseline_seeds_21_42/seed42/training_updates.jsonl) |
| R-Drop α=0 | 13 | [result.json](../rdrop_formal/batch_20261104T082306144395Z/rdrop_alpha0/seed13/result.json) | [predictions_dev.jsonl](../rdrop_formal/batch_20261104T082306144395Z/rdrop_alpha0/seed13/predictions_dev.jsonl) | [training_updates.jsonl](../rdrop_formal/batch_20261104T082306144395Z/rdrop_alpha0/seed13/training_updates.jsonl) |
| R-Drop α=0 | 21 | [result.json](../rdrop_formal/batch_20261104T082306144395Z/rdrop_alpha0/seed21/result.json) | [predictions_dev.jsonl](../rdrop_formal/batch_20261104T082306144395Z/rdrop_alpha0/seed21/predictions_dev.jsonl) | [training_updates.jsonl](../rdrop_formal/batch_20261104T082306144395Z/rdrop_alpha0/seed21/training_updates.jsonl) |
| R-Drop α=0 | 42 | [result.json](../rdrop_formal/batch_20261104T082306144395Z/rdrop_alpha0/seed42/result.json) | [predictions_dev.jsonl](../rdrop_formal/batch_20261104T082306144395Z/rdrop_alpha0/seed42/predictions_dev.jsonl) | [training_updates.jsonl](../rdrop_formal/batch_20261104T082306144395Z/rdrop_alpha0/seed42/training_updates.jsonl) |
| R-Drop α=0.5 | 13 | [result.json](../rdrop_formal/batch_20261104T082306144395Z/rdrop_alpha0.5/seed13/result.json) | [predictions_dev.jsonl](../rdrop_formal/batch_20261104T082306144395Z/rdrop_alpha0.5/seed13/predictions_dev.jsonl) | [training_updates.jsonl](../rdrop_formal/batch_20261104T082306144395Z/rdrop_alpha0.5/seed13/training_updates.jsonl) |
| R-Drop α=0.5 | 21 | [result.json](../rdrop_formal/batch_20261104T082306144395Z/rdrop_alpha0.5/seed21/result.json) | [predictions_dev.jsonl](../rdrop_formal/batch_20261104T082306144395Z/rdrop_alpha0.5/seed21/predictions_dev.jsonl) | [training_updates.jsonl](../rdrop_formal/batch_20261104T082306144395Z/rdrop_alpha0.5/seed21/training_updates.jsonl) |
| R-Drop α=0.5 | 42 | [result.json](../rdrop_formal/batch_20261104T082306144395Z/rdrop_alpha0.5/seed42/result.json) | [predictions_dev.jsonl](../rdrop_formal/batch_20261104T082306144395Z/rdrop_alpha0.5/seed42/predictions_dev.jsonl) | [training_updates.jsonl](../rdrop_formal/batch_20261104T082306144395Z/rdrop_alpha0.5/seed42/training_updates.jsonl) |
| R-Drop α=1.0 | 13 | [result.json](../rdrop_formal/batch_20261104T082306144395Z/rdrop_alpha1/seed13/result.json) | [predictions_dev.jsonl](../rdrop_formal/batch_20261104T082306144395Z/rdrop_alpha1/seed13/predictions_dev.jsonl) | [training_updates.jsonl](../rdrop_formal/batch_20261104T082306144395Z/rdrop_alpha1/seed13/training_updates.jsonl) |
| R-Drop α=1.0 | 21 | [result.json](../rdrop_formal/batch_20261104T082306144395Z/rdrop_alpha1/seed21/result.json) | [predictions_dev.jsonl](../rdrop_formal/batch_20261104T082306144395Z/rdrop_alpha1/seed21/predictions_dev.jsonl) | [training_updates.jsonl](../rdrop_formal/batch_20261104T082306144395Z/rdrop_alpha1/seed21/training_updates.jsonl) |
| R-Drop α=1.0 | 42 | [result.json](../rdrop_formal/batch_20261104T082306144395Z/rdrop_alpha1/seed42/result.json) | [predictions_dev.jsonl](../rdrop_formal/batch_20261104T082306144395Z/rdrop_alpha1/seed42/predictions_dev.jsonl) | [training_updates.jsonl](../rdrop_formal/batch_20261104T082306144395Z/rdrop_alpha1/seed42/training_updates.jsonl) |

## 两个失败或边界案例

1. **α=0 不是普通微调的数值对照。** 它只移除了 KL 项，仍保留双次随机前向和两份交叉熵；每种子仍有 9,166 次编码器前向，平均训练时间约 607 秒。因此它是 R-Drop 路径的零系数实现检查，不应用来替代单前向 baseline。
2. **当前最佳系数不等于普遍或显著提升。** α=0.5 在三个种子上的平均 Accuracy/F1 最高，但只有三个种子且 dev 被用于比较；α=1.0 相对 baseline 的配对平均增益较小（Accuracy +0.1634、F1 +0.0720 个百分点）。结论应限定为当前 MRPC dev 与当前协议下的观察，不推广到独立测试集或其他任务。

## 下一项单因素实验

将正则系数作为唯一变化因素，在同一 MRPC 划分、BERT 编码器、训练协议、更新步数和种子 `13/21/42` 下补跑 `alpha=0.25` 与 `alpha=0.75`；保持单前向 baseline 与既有 `alpha=0/0.5/1.0` 结果作为参照。逐种子记录 Accuracy、F1、训练 wall time、优化器更新数和编码器前向/样本通过量。这样可细化当前 `0.5` 附近的系数敏感性；新增结果仍属于 dev 比较，不作显著性或免费增益结论。
