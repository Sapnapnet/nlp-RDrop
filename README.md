# B 阶段 R-Drop / MRPC 最终交付说明

## 一、交付目的

本包用于 B 向下一位组员交接 R-Drop / MRPC 实验代码、正式实验结果和原始证据。
主要面向实验结果分析、中期报告撰写和后续成果整理。

## 二、实验完成情况

- 数据集：MRPC。
- 验证集：408 条，报告名称为 dev。
- Alpha：0、0.5、1.0。
- 随机种子：13、21、42。
- 正式实验：9/9 PASS。
- 每组优化器更新：2296。
- 使用最后一次更新的 checkpoint。

## 三、R-Drop 正式结果

| Alpha | Accuracy (%) | F1 (%) | 训练时间 (s) |
|---:|---:|---:|---:|
| 0 | 83.2516 ± 1.4976 | 88.3561 ± 0.8016 | 607.02 ± 27.56 |
| 0.5 | 83.7418 ± 0.8608 | 88.7046 ± 0.3188 | 584.93 ± 0.87 |
| 1 | 83.2516 ± 1.0204 | 88.2993 ± 0.4086 | 590.78 ± 5.47 |

## 四、重要结果解释

- Alpha=0.5 在当前三个种子的平均 Accuracy 和 F1 上表现最好。
- Alpha=0.5 相对配对 Baseline 的 Accuracy 平均差值为 +0.6536 个百分点。
- Alpha=0.5 相对配对 Baseline 的 F1 平均差值为 +0.4773 个百分点。
- Alpha=0 仍然执行两次随机前向和两次 CE，不能当作普通单前向 Baseline。
- ± 为三个种子的样本标准差，不是标准误或置信区间。
- 指标来自 validation/dev，不是独立 test 集。
- 当前结果不能直接证明统计显著性。

## 五、重要文件位置

| 目录 | 内容 |
|---|---|
| `src/` | 模型、训练和实验核心代码 |
| `scripts/` | 实验运行、预检和汇总脚本 |
| `configs/` | 数据、训练和实验配置 |
| `data/mrpc/` | MRPC 原始数据及处理后的数据 |
| `vendor/` | 作者相关 Hugging Face 源码；已排除 Fairseq 和 ViT 无关源码 |
| `patches/` | 兼容性修改补丁 |
| `provenance/` | 来源、版本和哈希记录 |
| `reports/rdrop_formal/` | 九组正式实验的报告、日志和预测 |
| `reports/summary/` | Baseline 三种子汇总 |
| `reports/baseline/` | Baseline seed 13 证据 |
| `reports/baseline_batch/` | Baseline seed 21/42 证据 |
| `docs/` | B 对接和训练协议相关文档 |
| `tests/` | 项目测试代码 |

## 六、交付范围和限制

本包是实验结果分析与报告交接包，不是开箱即用的完整训练环境。

为避免压缩包过大，以下内容没有包含：

- BERT 预训练模型权重。
- 九组 R-Drop 训练后的模型权重。
- Baseline 训练后的模型权重。
- `.venv` Python 虚拟环境。
- `wheelhouse` 离线依赖安装包。
- 与 B 阶段无关的 A 阶段模型备份。

checkpoint_manifest.json 等记录仍保留，用于查看原模型文件的哈希证据。
若需要独立重新训练、模型推理或重新加载 checkpoint，必须另外取得对应权重和运行依赖。

## 七、建议接手顺序

1. 先阅读本 README。
2. 阅读 `reports/rdrop_formal/` 下的 `rdrop_results.md`。
3. 阅读 `reports/summary/baseline_results.md`。
4. 对照 `result.json`、`formal_report.json` 和 `predictions_dev.jsonl` 查看原始证据。
5. 阅读 `docs/` 下的 B 对接和训练协议说明。
6. 根据正式结果开展分析和报告撰写。

## 八、结果文件说明

正式汇总以 `rdrop_results.json` 和 `rdrop_results.md` 为准。
Baseline 汇总以 `reports/summary/baseline_results.json` 和对应 Markdown 为准。

本 README 是当前交付状态说明，不应与旧阶段的历史进度记录混淆。
