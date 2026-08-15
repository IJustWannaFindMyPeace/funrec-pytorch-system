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

## 兴趣漂移与诊断

兴趣漂移指用户偏好随时间改变，例如早期偏好喜剧、近期转向科幻。下一阶段严格只在 Validation 上分析历史长度、活跃度、偏好熵、目标热门度、年代、genre、负样本类型、召回命中和精排位置，再据此选择优化。

## 代码索引

- `backend/modeling/youtubednn.py`
- `backend/modeling/deepfm.py`
- `backend/offline/feature/preprocess_retrieval.py`
- `backend/offline/feature/preprocess_ranking.py`
- `backend/offline/evaluation/retrieval.py`
- `docs/results/baseline_v0_*.json`

