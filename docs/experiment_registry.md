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
- 机制诊断：使用被拒绝候选的 Epoch 20 checkpoint、仅 Validation、无重训。AQ3 的 movie attention 有更长有效历史（9.628 对 AQ0 的 8.226）且近期 5 项权重更低（0.702 对 0.756）；genre 通道近期权重接近均匀。该相关性不证明因果，也不支持删除旧历史。证据：`docs/results/attention20_attention_mechanism_summary.json`。

## 单次实验必须保存

- 实验 ID、假设、Git SHA、数据版本和切分统计。
- 完整命令、配置、seed、history、最佳 epoch 和停止原因。
- Validation/Test 指标及其候选集语义。
- checkpoint、模型哈希、部署工件和运行环境。
- 成功、失败、限制和下一步推理。

## 前瞻性探索治理规则

- 本规则不修改既有实验的五项 Validation 准入门槛；仅约束未来是否继续同一探索方向。
- 同一假设族连续两次正式预注册候选因同一决定性门槛失败时，关闭该假设族。
- 每个新假设族最多两次正式候选；第二次失败后必须转向不同因果层级，而不是继续增加同类模型复杂度。
- 新假设族的预注册必须说明目标门槛、Train-only 实现方式、预期失败模式和关闭条件。
- 若两个独立假设族均未产生 5/5 通过的候选，停止模型搜索，冻结可信 Baseline、失败迭代与机制证据，完成面试交付物。

### 已关闭：History-20 attention/pooling

- 单路 personalized attention：AQ0-AQ3 gap=0.120463，超过 0.088300 上限，4/5 通过后拒绝。
- 双时间尺度 attention：gap 降至 0.106017，但仍超过上限，4/5 通过后拒绝。结果：`docs/results/dualtimescale_attention20_validation_results.json`。
- 因两次连续正式变体均由同一 gap 门槛拒绝，本假设族关闭；不得继续提出 activity-conditioned 或更复杂的 pooling 变体。

### 协议无效事件：activity-balanced History-10

- `baseline-v1-activity-balanced-history10` 错误复用了含嵌入 Test 的 Baseline V0 训练 artifact；加载 pickle 即发生 Test 反序列化，即使训练循环未消费 Test。
- 所有该目录 checkpoint、日志和训练 history 均不可用于选型、报告或 PPT；Validation evaluator 在输出指标前拒绝该 artifact。
- 目录不得删除、覆盖、复用或再次评估。必须从不含 Test 的 History-10 Train/Validation 工件重新开始。证据：`docs/results/activity_balanced_history10_protocol_incident.json`。

### 预注册替代运行：activity-balanced History-10

- 原运行仅因工件含嵌入 Test 而协议无效；其训练曲线、checkpoint 与任何未输出指标均不作为候选证据。替代运行仍是该独立假设族的第 1 个正式候选，而不是依据结果追加的变体。
- 已冻结 `docs/results/activity_balanced_history10_rerun_preregistered_config.json`：五项 Validation 准入门槛、模型、超参和唯一核心变量完全不变；仅更换为 Train/Validation-only 工件，并在训练入口增加 fail-closed 校验。
- 状态：`validation_rejected`。有效替代运行在 Epoch 25 因 `patience=3` 早停，按 Validation loss 选择 Epoch 22（loss=6.422596）。Validation Recall@10=0.152980、NDCG@10=0.076987、AQ3=0.085828、AQ0-AQ3 gap=0.127706、Tail PQ0=0.049213；仅 NDCG 通过，故 1/5 拒绝。Test 未访问、未反序列化、未重评估。完整证据：`docs/results/activity_balanced_history10_rerun_validation_results.json`。
