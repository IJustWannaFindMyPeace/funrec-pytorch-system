# Baseline V0 第一性审计

## 文档目的

本文件记录 FunRec PyTorch Baseline V0 在正式训练前的数据、训练与评估审计。目标不是预设优化方案，而是先确认指标口径可信、实验可复现，并为后续分析驱动的算法迭代建立事实基线。

## 当前版本定位

- 迁移与工程基准提交：`e69ccda`
- 实验分支：`experiment/baseline-v0`
- 当前结果属于 **Migration Acceptance Results**，用于证明 TensorFlow 到 PyTorch 的迁移和端到端链路可运行。
- 当前结果尚不属于独立测试集上的 **Baseline V0 Test Results**。

| 项目 | 结果 | 当前口径 |
|---|---:|---|
| YouTubeDNN Recall@10 | 0.100662 | Validation |
| Popularity Recall@10 | 0.019536 | Validation |
| DeepFM AUC | 0.854670 | Sampled validation AUC |
| 后端测试 | 102 passed | 工程正确性 |
| 在线链路 | YouTubeDNN → DeepFM → 重排 | 本地端到端验证 |

## 数据与切分审计

### 召回数据

当前实现按 `timestamp` 和原始行号稳定排序，并在用户内部构建滑动窗口样本。最后一次交互作为评估目标，更早交互用于训练，方向上符合时序推荐评估。

主要问题：

1. 数据只有 `train` 和 `test` 两部分。
2. 训练脚本实际把 `samples["test"]` 用作 validation。
3. Early stopping 和最佳 checkpoint 都基于这部分数据。
4. 最终 Recall/NDCG 又在相同数据上报告。

正式协议：

- Train：除最后两次交互以外的滑动窗口样本。
- Validation：每个满足长度要求用户的倒数第二次交互。
- Test：每个满足长度要求用户的最后一次交互。
- Test 只能在模型和超参数确定后评估。

### 排序数据

当前实现已做到用户内时间排序、训练期阈值计算以及训练和评估分别负采样。

主要问题：

1. 仍然只有 train/test，且 test 被用于 checkpoint 选择。
2. 当前 AUC 基于采样后的正负样本集合，不代表全量物品排序能力。
3. 初次严格时序实现仅排除当期历史，导致部分未来正反馈被错误采为早期随机负样本。
4. 用户、电影与类别词表的拟合范围需要明确为闭集部署设定或严格归纳设定。

正式协议：

- 用户内时序 Train/Validation/Test 三段切分。
- 标签阈值只由 Train 计算。
- Validation 用于 early stopping。
- Test 在配置冻结后只运行一次。
- 随机负采样使用全量已知交互做离线 false-negative sanitation；该步骤不进入模型特征、标签阈值或 checkpoint 选择，并将严格 point-in-time 采样保留为后续消融。
- 同时报告分类指标与候选集 Top-K 排序指标。


### 真实工件审计与协议修复

首次真实工件审计发现，严格 point-in-time 随机负采样产生了明显的跨时间标签矛盾，同时评估构造只保留在窗口内至少有一个正反馈的用户。

| 审计项 | 修复前 | 修复后 |
|---|---:|---:|
| Validation 用户覆盖 | 5,871 / 6,040 | 6,040 / 6,040 |
| Test 用户覆盖 | 6,004 / 6,040 | 6,040 / 6,040 |
| Train 负样本与 Validation 正样本冲突 | 7,649 | 0 |
| Train 负样本与 Test 正样本冲突 | 14,963 | 0 |
| Validation 负样本与 Test 正样本冲突 | 2,027 | 0 |
| 三组任意 `(user, movie)` 跨 split 重叠合计 | 36,396 | 0 |
| 时间边界违规 | 0 | 0 |
| 编码越界、OOV、split 内标签冲突 | 0 | 0 |

修复策略：训练集继续采用受控困难负样本和随机负样本；Validation/Test 保留全部真实曝光负反馈；随机负样本候选排除用户全量已知交互，避免离线标签自相矛盾。该设置属于 MovieLens 闭集、离线 false-negative sanitation，不等价于线上能够提前知道未来行为。

机器可读证据：`docs/results/baseline_v0_data_audit.json`。对应协议修复提交：`126641b`。全量工程测试：`112 passed`。

## 指标语义审计

### Recall@K 与 HitRate@K

当前召回评估每个用户只有一个目标电影，因此 `Recall@K = HitRate@K`。这是 single-target leave-one-out 协议的数学结果，不是代码错误。

### DeepFM AUC

当前 `0.854670` 的严格表述是：DeepFM 在构造的正样本、曝光未点击负样本和随机负样本组成的验证集上的 ROC-AUC。它不能直接解释为全量电影推荐 Top-K 的提升。

## 风险登记

| ID | 风险 | 严重度 | Baseline V0 处理 |
|---|---|---|---|
| R1 | Validation 与 Test 复用 | 高 | 三段时序切分 |
| R2 | Test 参与 checkpoint 选择 | 高 | 独立 Test 入口 |
| R3 | Point-in-time 随机负样本与未来正反馈冲突 | 高 | 全量已知交互仅用于 false-negative sanitation，并保留严格时点采样消融 |
| R4 | AUC 与 Top-K 目标不一致 | 中 | 增加排序 Top-K 指标 |
| R5 | Single-target 下 Recall 与 HitRate 重复 | 低 | 文档明确，不重复解读 |
| R6 | 词表拟合范围口径不明确 | 中 | 区分闭集部署与严格归纳设置 |
| R7 | 现有指标命名为 test 可能误导 | 高 | 统一改称 migration validation |

## Baseline V0 冻结门槛

- [x] Train/Validation/Test 独立。
- [x] 训练过程中从不读取 Test 指标。
- [x] 负采样信息边界明确。
- [x] 指标定义与实现有测试。
- [ ] 完整训练命令和配置已保存。
- [ ] checkpoint、history、metrics 和环境信息可追溯。
- [ ] 端到端在线推理使用同一组冻结工件。
- [ ] Baseline 报告区分 validation 与 test。
- [ ] Baseline Seed 42 正式运行完成。
- [ ] Baseline 与最终最佳方案完成三随机种子复验。

## 后续分析原则

优化方向必须来自切片诊断，而不是预先套用方案。至少分析用户历史长度、物品热门度、类型偏好熵、召回漏失、召回到排序的漏斗损失、负样本难度、冷启动差异以及准确率/覆盖率/多样性/延迟权衡。
