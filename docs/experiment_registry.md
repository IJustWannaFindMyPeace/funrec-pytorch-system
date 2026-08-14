# FunRec 实验注册表

## 状态定义

- `planned`：已提出、未运行。
- `running`：存在未完成的冻结门槛。
- `completed`：工件、指标和验证均完整。
- `rejected`：实验无效或口径不可信，并保留原因。

## 实验总表

| ID | 状态 | 运行 Git SHA | Seed | 主要改动 | Validation | Test | 结论 |
|---|---|---|---:|---|---|---|---|
| MIGRATION-ACCEPTANCE | completed | e69ccda | 42 | TensorFlow → PyTorch 工程迁移 | Recall@10=0.100662；DeepFM AUC=0.854670 | 不适用 | 链路验收，不与 V0 直接比较 |
| EXP-000 | running | d65c9d6 | 42 | 三段时序切分、冲突修复、收敛校准、独立 Test | Retrieval loss=6.282928（Epoch 14）；DeepFM AUC=0.866237（Epoch 5） | Recall@10=0.137086；NDCG@10=0.070298；DeepFM AUC=0.841382 | 离线 V0 已封存；待同工件在线验证后完成 |

## EXP-000 事实记录

### 数据

- 原始交互 1,000,209；用户 6,040；电影 3,883。
- Retrieval Train/Validation/Test：982,089 / 6,040 / 6,040。
- Ranking Train/Validation/Test：1,703,230 / 225,347 / 448,147。
- 六类 Ranking 跨 split 冲突均为 0。

### 训练

- YouTubeDNN：实际运行 17 epochs，Epoch 14 Validation loss 最优。
- DeepFM：实际运行 8 epochs，Epoch 5 Validation AUC 最优。
- 初始默认 3 epochs 被 Validation 证据判定为欠训练；Test 未参与预算调整。

### Test

- Test 只运行一次，后续禁止覆盖或用其选择方案。
- Retrieval Recall@10 为 0.137086，是 popularity 的 7.0769 倍（相对提升 607.69%）。
- DeepFM Test ROC-AUC 为 0.841382，较 Validation 低 0.024855。
- 原始 Test SHA256：`2B88A39F61C18CC46562295E114273F0D015F14EB2FA8842124C3F5DF8918B9E`。

### 剩余完成条件

- 在线服务加载冻结召回和排序工件。
- 在线返回 `youtube_dnn`/`deepfm` 策略并验证部署哈希。
- 完成后将 EXP-000 改为 `completed` 并创建 Baseline tag。

## 单次实验必须保存

- 实验 ID、假设、Git SHA、数据版本和切分统计。
- 完整命令、配置、seed、history、最佳 epoch 和停止原因。
- Validation/Test 指标及其候选集语义。
- checkpoint、模型哈希、部署工件和运行环境。
- 成功、失败、限制和下一步推理。
