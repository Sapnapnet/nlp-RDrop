# 第 2 步源码与依赖检查

总体状态：**PASS**

这是离线兼容性诊断，使用临时词表、合成句子和随机初始化的小型 BERT。
不代表 MRPC Baseline、正式 R-Drop 消融或完整训练已完成。

| 检查项 | 状态 | 说明 |
|---|---|---|
| source_integrity | PASS | Verified commit 9dcc8302eb9b0f112c38230864bbee807bbe1aa7; two dependency edits only. |
| pinned_versions | PASS | All 44 dependency versions match the offline lock. |
| author_fork_import | PASS | Transformers loads from the bundled author source; bert_kl and bert_cross_entropy exist. |
| tokenizer_compatibility | PASS | tokenizers 0.13.3: sentence pairs, padding, truncation, slow/fast parity and save/load passed. |
| dataset_trainer_interop | PASS | Arrow save/load, author Trainer initialization and DataLoader batch succeeded. |
| metrics | PASS | Known-label Accuracy and binary F1 checks passed (synthetic values, not MRPC scores). |
| author_tiny_bert_fp32 | PASS | Two author BERT forwards, finite loss/gradients and update passed. |
| author_tiny_bert_fp16_amp | PASS | Two author BERT forwards, finite loss/gradients and update passed. |

详细包版本、实际导入路径和诊断结果见同目录 dependency_report.json。
