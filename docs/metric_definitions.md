# Baseline V0 指标定义

## 使用原则

- Validation 用于模型、训练预算和 checkpoint 选择。
- Test 在配置冻结后只运行一次，不能用于选择下一项优化。
- 所有指标记录样本数、候选集、K 值、历史过滤和采样方式。
- 离线回放不描述成线上 A/B Test。

## Retrieval

### Recall@K

相关目标中被 Top-K 命中的比例。V0 每用户只有一个目标，因此等于命中指示变量的用户均值。

### HitRate@K

至少命中一个目标的用户比例。V0 single-target 下与 Recall@K 完全相同，不作为独立证据重复汇报。

### NDCG@K

命中越靠前权重越高。Single-target 命中位置为 `rank` 时得分为 `1/log2(rank+1)`，未命中为 0。

### 倍数与相对提升

设模型指标为 `M`、基线为 `B`：

- 绝对提升：`M - B`。
- 基线倍数：`M / B`。
- 相对提升：`(M - B) / B`。
- 相对提升百分比：`((M - B) / B) × 100%`。

例如 Recall@10 的 `7.0769×` 等价于相对提升 `607.69%`，而不是 `707.69%`。

### V0 尚未报告

MRR、Coverage、长尾 Recall、多样性和新颖性已列为后续诊断指标，但 V0 正式 Test JSON 不包含这些数值。

## Ranking

### ROC-AUC

随机正样本得分高于随机负样本的概率。V0 在真实曝光负反馈与受控随机负样本组成的采样集合上计算，不代表全量电影 Top-K 排序效果。

### LogLoss

二分类交叉熵，衡量概率预测质量。V0 checkpoint 按 Validation AUC 而非最低 loss 选择，因此最低 loss epoch 与最佳 AUC epoch 可以不同。

### 计划补充但尚未在 V0 Test 报告

- PR-AUC：类别不平衡下的 Precision-Recall 曲线面积。
- GAUC：用户内 AUC 的加权平均。
- Ranking NDCG@K：同用户候选集内的排序质量。
- Calibration：Brier Score 或 ECE。

这些指标只能在下一轮预注册协议中加入，不能反复打开已经封存的 V0 Test 来选择方案。

## V0 评估协议

- Retrieval：6,040 用户，每用户一个目标，3,883 个电影，过滤历史，K=5/10/20/50。
- Ranking：448,147 个 Test 样本，其中正样本 125,837、负样本 322,310。
- Catalog：closed-catalog transductive。
- Seed：42。
- Test：唯一一次执行，原始文件由 SHA256 封存。

## 分层诊断指标

下一阶段只在 Validation 上按以下维度切片：

- 用户历史长度和活跃度。
- 目标电影训练期热门度（头部/腰部/长尾）。
- 用户类型偏好熵。
- 负样本难度。
- 时间区间与偏好漂移。
- 召回命中、排序提升和重排损失的漏斗阶段。
