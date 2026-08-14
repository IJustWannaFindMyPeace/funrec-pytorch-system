# Baseline V0 第一性审计

## 文档目的

本文件记录 FunRec PyTorch Baseline V0 的数据、训练、评估和报告语义审计。目标不是预设优化方案，而是确保后续每一次算法迭代都建立在可复现、不可反复窥视 Test 的事实基线上。

## 版本定位

- 迁移与工程基准：`e69ccda`
- 数据协议修复：`126641b`
- 正式实验代码：`d65c9d6`
- 实验分支：`experiment/baseline-v0`
- 迁移验收结果只证明链路可运行，不与正式 V0 做因果比较。
- 正式结果：`docs/baseline_results.md`

## 数据与切分审计

### Retrieval

- Train：每个用户更早交互形成的滑动窗口。
- Validation：每个用户倒数第二次交互。
- Test：每个用户最后一次交互。
- Validation/Test 各覆盖 6,040 个用户；跨 split 目标对重叠和时间违规均为 0。

### Ranking

- 用户内严格时序 Train/Validation/Test 三段切分。
- 标签阈值只由 Train 计算。
- Validation 用于 early stopping；Test 在配置冻结后只运行一次。
- Validation/Test 保留全部真实曝光负反馈。
- 随机负样本排除用户全量已知交互，用于闭集离线 false-negative sanitation；严格 point-in-time 采样保留为后续消融。

### 协议修复证据

| 审计项 | 修复前 | 修复后 |
|---|---:|---:|
| Validation 用户覆盖 | 5,871 / 6,040 | 6,040 / 6,040 |
| Test 用户覆盖 | 6,004 / 6,040 | 6,040 / 6,040 |
| Train 负样本与 Validation 正样本冲突 | 7,649 | 0 |
| Train 负样本与 Test 正样本冲突 | 14,963 | 0 |
| Validation 负样本与 Test 正样本冲突 | 2,027 | 0 |
| 三组任意用户—电影跨 split 重叠合计 | 36,396 | 0 |
| 时间边界违规 | 0 | 0 |
| 编码越界、OOV、split 内标签冲突 | 0 | 0 |

机器可读证据：`docs/results/baseline_v0_data_audit.json`。

## 训练审计

默认 3 epochs 不能作为收敛证据。YouTubeDNN 在 Epoch 3 后 Validation loss 继续下降，最终在 Epoch 14 最优并于 Epoch 17 early stop。DeepFM 在 Epoch 5 达到最高 Validation AUC，并于 Epoch 8 early stop。所有选择均发生在 Test 首次打开之前。

## 指标与报告语义审计

- Single-target 下 `Recall@K = HitRate@K`，不重复解读。
- `times_baseline = model / baseline`；`relative_improvement = (model - baseline) / baseline`。
- DeepFM ROC-AUC 基于采样候选，只表示正负区分能力。
- V0 未实现 Ranking 候选集 Top-K、GAUC、PR-AUC 和 Calibration；这些是后续评估能力，不伪装为已报告结果。
- 原始 Test 结果已封存；公开 JSON 脱敏本机路径，并通过原始 SHA256 追溯。

## 风险登记

| ID | 风险 | 严重度 | V0 处理或状态 |
|---|---|---|---|
| R1 | Validation 与 Test 复用 | 高 | 已修复：三段时序切分 |
| R2 | Test 参与 checkpoint 选择 | 高 | 已修复：独立 Test，只运行一次 |
| R3 | 早期随机负样本与未来正反馈冲突 | 高 | 已修复：闭集 sanitation；保留 point-in-time 消融 |
| R4 | ROC-AUC 与 Top-K 目标不一致 | 中 | 明确限制；Ranking Top-K 尚未实现 |
| R5 | Recall 与 HitRate 重复 | 低 | 文档明确 single-target 等价性 |
| R6 | 闭集词表不代表归纳泛化 | 中 | 明确 closed-catalog transductive 口径 |
| R7 | 倍数被误命名为 relative lift | 中 | 已修复：倍数与相对提升分字段 |
| R8 | Validation→Test AUC 下降 2.49pp | 中 | 待 Validation 分群诊断 |
| R9 | 单随机种子结果方差未知 | 中 | 最终候选方案阶段进行多 seed 复验 |

## 冻结门槛

- [x] Train/Validation/Test 独立。
- [x] 训练过程中不读取 Test。
- [x] 负采样信息边界明确。
- [x] 指标定义和报告语义有测试。
- [x] 完整训练配置已保存。
- [x] checkpoint、history、metrics、环境和哈希可追溯。
- [x] Validation 与 Test 严格区分。
- [x] Seed 42 正式训练与唯一 Test 完成。
- [ ] 在线服务加载同一组冻结工件并验证哈希。
- [ ] 最终候选方案与 V0 完成多随机种子复验。

## 后续分析原则

下一项优化必须来自 Validation 切片诊断，而不是根据 Test 反复试方案。优先检查用户历史长度、目标物品训练期热门度、类型偏好熵、召回漏失、负样本难度和时序漂移；再按证据强度、预期收益、实现成本和面试解释价值选择实验。
