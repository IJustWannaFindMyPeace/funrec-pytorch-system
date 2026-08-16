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
| EXP-000 | completed | d65c9d6 | 42 | 三段时序切分、冲突修复、收敛校准、独立 Test、在线同工件验证 | Retrieval loss=6.282928（Epoch 14）；DeepFM AUC=0.866237（Epoch 5） | Recall@10=0.137086；NDCG@10=0.070298；DeepFM AUC=0.841382 | V0 已封存；离线评估与在线服务核心工件哈希一致 |

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

### 在线验证

- 宿主机训练产物、部署目录与 Docker 容器内四个核心工件 SHA256 全部一致。
- 在线返回 `youtube_dnn` 和 `user_preference` 召回通道，排序策略为 `deepfm`。
- 请求返回 100 个召回候选、20 个不重复推荐结果，HTTP 状态为 200。
- `(3883, 16)` 离线向量在线变为 `(3884, 16)`，仅因索引 0 添加全零 padding/OOV 行。
- 验证代码版本：`fe394f7`；证据：`docs/results/baseline_v0_online_verification.json`。
- `EXP-000` 已满足 completed 定义；合并后创建 Baseline tag。

## ATTENTION-20（正式预注册）

- 状态：`validation_rejected`；正式训练与 Validation-only 评估已完成。
- 代码节点：`18dc8639203b77f2a4dc1db9976ad4c1ae33646e`；seed：42。
- 唯一核心变量：History-20 的 `personalized_attention`，相对于已拒绝的 History-20 `masked_mean`；其余训练协议固定不变。
- 完整配置与五项 Validation 准入门槛：`docs/results/attention20_preregistered_config.json`；结果证据：`docs/results/attention20_validation_results.json`。
- 选择边界：仅 Validation；封存 Test 禁止访问、反序列化和重新评估。
- 训练：Epoch 1 后因执行环境时限中断；从 checkpoint 无覆盖恢复，Epoch 23 因 patience=3 早停，按 Validation loss 选择 Epoch 20。
- 决策：Overall Recall@10=0.199007、NDCG@10=0.101495、AQ3 Recall@10=0.139721、Tail PQ0 Recall@10=0.120735 均通过；AQ0-AQ3 gap=0.120463 超过冻结上限 0.088300。仅通过 4/5，拒绝；Test 保持关闭。

## 单次实验必须保存

- 实验 ID、假设、Git SHA、数据版本和切分统计。
- 完整命令、配置、seed、history、最佳 epoch 和停止原因。
- Validation/Test 指标及其候选集语义。
- checkpoint、模型哈希、部署工件和运行环境。
- 成功、失败、限制和下一步推理。
