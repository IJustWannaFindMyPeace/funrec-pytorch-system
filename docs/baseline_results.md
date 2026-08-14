# Baseline V0 正式结果

## 结果定位

`EXP-000` 是 FunRec PyTorch 迁移后的首个可信离线 Baseline。它不是模型优化实验：本阶段只修正数据协议、建立独立 Test、校准训练收敛并冻结可追溯工件。

- 实验代码提交：`d65c9d6`
- 数据协议修复提交：`126641b`
- Seed：42
- Python：3.11.15
- PyTorch：2.12.1+cu132
- GPU：NVIDIA GeForce RTX 5070
- Test 执行次数：1
- 原始 Test SHA256：`2B88A39F61C18CC46562295E114273F0D015F14EB2FA8842124C3F5DF8918B9E`

机器可读证据：

- `docs/results/baseline_v0_data_audit.json`
- `docs/results/baseline_v0_run_config.json`
- `docs/results/baseline_v0_model_hashes.json`
- `docs/results/baseline_v0_test_metrics.json`

## 数据协议

| 项目 | Train | Validation | Test |
|---|---:|---:|---:|
| 原始时序交互 | 705,170 | 97,383 | 197,656 |
| Retrieval 样本 | 982,089 | 6,040 | 6,040 |
| Ranking 采样样本 | 1,703,230 | 225,347 | 448,147 |
| Ranking 正样本 | 499,640 | 64,190 | 125,837 |

- 6,040 个用户均覆盖 Validation/Test。
- 原始交互跨 split 重叠、时间边界违规、Retrieval 目标对重叠均为 0。
- Ranking 六类跨 split 用户—电影冲突均为 0。
- 标签阈值只由训练期计算。
- 随机负样本使用全量已知交互进行离线 false-negative sanitation；这是 MovieLens 闭集实验处理，不代表线上提前知道未来。
- Validation 用于 checkpoint 选择；配置冻结后 Test 只运行一次。

## 训练收敛

| 模型 | 初始默认预算 | 最终上限 | 实际运行 | 选择指标 | 最佳 epoch | 最佳 Validation |
|---|---:|---:|---:|---|---:|---:|
| YouTubeDNN | 3 | 30 | 17 | Loss（越低越好） | 14 | 6.282928 |
| DeepFM | 3 | 30 | 8 | ROC-AUC（越高越好） | 5 | 0.866237 |

YouTubeDNN 在 Epoch 3 的 Validation loss 仍为 `6.743469` 且持续下降，因此默认 3 epochs 属于欠训练。扩大安全上限后，Epoch 14 达到最低 loss，Epoch 15–17 连续未改善并触发 early stopping。

DeepFM 的最低 Validation loss 出现在 Epoch 2（`0.413154`），但最高 Validation AUC 出现在 Epoch 5（`0.866237`）。V0 以排序判别能力为 checkpoint 目标，因此选择 Epoch 5；Epoch 6–8 未再提升后停止。

## Retrieval Test

协议：6,040 个 Test 用户；每用户一个 held-out 目标；3,883 个电影的闭集候选；过滤用户历史；Embedding 维度 16。

| K | YouTubeDNN Recall/HitRate | Popularity Recall/HitRate | YouTubeDNN NDCG | Popularity NDCG | 倍数（Recall） | 相对提升（Recall） |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.085430 | 0.009106 | 0.053784 | 0.005777 | 9.3818× | 838.18% |
| 10 | 0.137086 | 0.019371 | 0.070298 | 0.009065 | 7.0769× | 607.69% |
| 20 | 0.216391 | 0.042384 | 0.090259 | 0.014785 | 5.1055× | 410.55% |
| 50 | 0.364735 | 0.107947 | 0.119609 | 0.027670 | 3.3788× | 237.88% |

Recall@10 对应 YouTubeDNN 命中 `828/6040` 个用户，Popularity 命中 `117/6040`。由于每个用户只有一个目标，Recall@K 与 HitRate@K 数值完全相同，不能当作两项独立收益。

“7.0769×”表示 `model / baseline`；严格的相对提升是 `(model - baseline) / baseline = 607.69%`。两种表述不能混用。

## Ranking Test

| 指标 | Validation | Test | 差值（Test - Validation） |
|---|---:|---:|---:|
| ROC-AUC | 0.866237 | 0.841382 | -0.024855 |
| Loss | 0.417611（AUC 最优 epoch） | 0.464629 | +0.047018 |

Test 包含 448,147 个采样样本，其中正样本 125,837、负样本 322,310。该 AUC 衡量构造候选中的整体正负区分能力，不等价于全量电影 Top-K 排序收益，也不是线上 A/B Test。

Validation 到 Test 的 AUC 下降 2.49 个百分点，证明存在时间外推泛化差距；其来源可能包含用户活跃度、物品热度、偏好漂移或负样本难度变化，必须在 Validation 上做切片诊断后才能下结论。

## 不能从 V0 得出的结论

- 不能将数据协议修复产生的变化描述成模型算法提升。
- 不能把迁移验收阶段指标与 V0 指标直接做因果比较，因为切分、采样与训练预算不同。
- 不能用 Test 分群结果选择下一项优化；下一步假设必须来自 Validation 诊断。
- 当前未报告 Ranking GAUC、PR-AUC、候选集 NDCG、Calibration、Coverage、多样性或在线延迟分布。
- Seed 42 是正式单次 Baseline；最终候选方案仍需与 V0 做多随机种子复验。

## 冻结状态

离线数据、训练、checkpoint、模型哈希和唯一 Test 结果均已冻结。尚未完成的最后门槛是：让在线服务加载同一组冻结部署工件，并验证召回策略、排序策略和模型哈希一致。
