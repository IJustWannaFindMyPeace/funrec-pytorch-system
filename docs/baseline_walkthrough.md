# PyTorch Baseline V1 Walkthrough

> 本文是代码、实验记录与面试 PPT 的共同事实源。优化只使用 Train/Validation，冻结 Test 不参与选型。

## 数据与目标

MovieLens 1M 包含 6,040 位用户、3,883 部电影和 1,000,209 条评分。电影编码 1–3,883，0 为 padding/OOV。每个用户按时间排序：首条用于形成历史，倒数第二条为 Validation 目标，最后一条为 Test 目标，因此 Retrieval 训练样本数为 `1,000,209 - 3 × 6,040 = 982,089`。

用户目标电影 `i_u*` 是评估窗口中该用户真实发生的下一次交互。`进入 Top-10` 指它出现在模型过滤已看历史后返回的十部最高分电影中。

## YouTubeDNN

每部电影就是一个类别。用户塔输出 `z_u∈R^16`，电影 `j` 的向量为 `v_j∈R^16`：

`s(u,j)=z_u^T v_j`

`P(j|u)=exp(s(u,j))/Σ_{m=1}^{3883} exp(s(u,m))`

真实下一部电影为 `i_u*` 时：`L=-log P(i_u*|u)`。当前实现对全部 3,883 个非 padding 类计算 exact full softmax，不使用 20 个训练负样本。随机均匀预测的损失约为 `log(3883)=8.264`，交叉熵范围 `[0,+∞)`。

训练时用户向量归一化、电影权重未归一化；导出和服务时两边归一化。这是待在 Validation 上验证的 train–serve scoring 差异。

## DeepFM embedding

九个 field 为 user_id、gender、age、occupation、zip_code、movie_id、genres、isAdult、startYear。

### 一阶 embedding

`linear_embeddings[field]` 是 `Embedding(vocab_size, 1)`。某个字段取值查表得到标量 `w_{f,x_f}`：

`s_linear(x)=Σ_f w_{f,x_f}`

它表示单个取值自身的全局贡献。

### 二阶 embedding 与共享

`feature_embeddings[field]` 是 `Embedding(vocab_size,16)`，取值向量记为 `e_{f,x_f}`。FM 二阶项：

`s_FM(x)=Σ_{f<g}<e_{f,x_f},e_{g,x_g}>`

同一批 16 维向量一条路径计算 FM 内积，另一条路径拼接后输入 DNN。共享发生在 FM 与 DNN 分支之间：二者没有各自维护一套 16 维表，反向传播会共同更新同一个 `feature_embeddings`。一阶的 1 维标量表则是独立参数。

最终 `s=b+s_linear+s_FM+s_DNN`，`p=sigmoid(s)`，三分支共用一次 pointwise BCE：

`L=-[y log p+(1-y)log(1-p)]`

范围 `[0,+∞)`；`p=0.5` 时为 `log 2≈0.693`。Baseline 没有 BPR/hinge pairwise 或 ListNet/ListMLE listwise。genres 只取第一个类型是工程简化，不是 DeepFM 理论限制。

## 指标

### Recall@K / HitRate@K

`Recall@K=(1/|U|)Σ_u |R_u^K∩G_u|/|G_u|`。V1 每用户只有一个目标，故等于 `命中用户数/总用户数`，也等于 HitRate@K，范围 `[0,1]`。Test Recall@10=`828/6040=0.137086`。

### NDCG@K

`DCG@K=Σ_{r=1}^K (2^{rel_r}-1)/log2(r+1)`，`NDCG=DCG/IDCG`。单目标时，目标在第 r 名则为 `1/log2(r+1)`，未进 Top-K 为 0。第 1/2/3/10 名分别为 1/0.631/0.5/0.289；范围 `[0,1]`。

### ROC-AUC

`AUC=P(s(x+)>s(x-))+0.5P(s(x+)=s(x-))`，范围 `[0,1]`；0.5 相当于随机。它衡量采样候选上的正负判别，不等于完整目录 Top-K。

Validation AUC=0.866237，Test AUC=0.841382，下降 `0.024855`，即 2.49 个百分点（相对下降约 2.87%）。这表示更晚窗口泛化变弱，但不能单独证明用户兴趣漂移。

## Validation-only 诊断结论

诊断代码始终读取 `validation`，并在证据中声明 `test_accessed=false`。V2 发现所有 Retrieval Validation 样本的长度 10 历史窗口均已饱和；这只是模型可见窗口，不是用户原始活跃度。按冻结协议重建的原始活动范围为 20–2,314 次，训练样本还存在明显用户集中：Retrieval/Ranking 中最活跃 10% 用户分别贡献 38.61%/37.83% 的训练样本。

V3 在 Validation 上得到 Retrieval Recall@10=`0.156954`、NDCG@10=`0.077737`；DeepFM ROC-AUC=`0.866237`、PR-AUC=`0.692331`、logloss=`0.417611`。Retrieval 的最低/最高活跃四分位 Recall@10 为 `0.198423/0.105123`；最低/最高目标热门度四分位为 `0.079396/0.192715`。这同时提示活跃用户表示和长尾召回问题，但单轴切片可能混杂。

V4 使用“原始用户活跃度 × Train 电影热门度”4×4 交叉矩阵。固定最热门电影组后，低/高活跃用户 Retrieval Recall@10 为 `0.251969/0.087963`，相对下降 65.1%；NDCG@10 为 `0.1292/0.0454`，相对下降 64.9%。因此活跃度退化不能只由目标电影更冷门解释。不过在最冷门组中该趋势不成立，说明活跃度与热门度存在交互，不能宣称历史截断是唯一原因。

DeepFM 在 16 个交叉格中都优于同格常数正例率预测器，但优势随活跃度上升而普遍缩小。整体 PR-AUC=`0.692331`，其无信息基线为正例率 `0.284850`；归一化 logloss=`0.698959`，即相对常数预测器改善 30.10%。热门度分组正例率差异很大，因此原始 PR-AUC 和 logloss 不能脱离各组基线直接横比。

兴趣漂移指偏好随时间改变，例如早期偏好喜剧、近期转向科幻；多兴趣指同一用户同时存在多个兴趣主题。当前证据只能说明高活跃用户更难建模，不能仅凭分组指标证明发生兴趣漂移。

## 第一次优化：预注册实验协议

第一个方向是 Retrieval 历史表示容量，而非先改 sampled softmax。保持数据切分、标签、embedding 维度、优化器、full softmax 与 checkpoint 选择规则不变，先比较长度 10/20/50 的近期历史平均池化；若增加长度无效，再在最佳长度上比较 recency-weighted pooling 和 attention pooling。

选择只使用 Validation。主要指标为整体 Recall@10；约束指标为最高活跃组 AQ3 Recall@10、AQ3–AQ0 差距、长尾 PQ0 Recall@10 和 NDCG@10。候选必须同时满足：整体 Recall@10 提升、AQ3 提升、活跃度差距缩小，且 PQ0 无明显退化。Test 继续封存。若长度增加只提高训练成本或稀释近期兴趣，则保留长度 10，并把该负结果写入故事。

Sampled softmax主要解决超大类别空间的计算成本；当前只有 3,883 类且 exact softmax 可承受，不能预设它提升质量。固定每用户样本数、长尾重加权和 train–serve 归一化一致性均保留为后续独立消融，避免一次修改多个变量。

## 代码索引

- `backend/modeling/youtubednn.py`
- `backend/modeling/deepfm.py`
- `backend/offline/feature/preprocess_retrieval.py`
- `backend/offline/feature/preprocess_ranking.py`
- `backend/offline/evaluation/retrieval.py`
- `backend/offline/evaluation/diagnose_validation.py`
- `backend/offline/evaluation/diagnose_validation_models.py`
- `backend/offline/evaluation/diagnose_validation_cross.py`
- `docs/results/baseline_v0_*.json`
