# FunRec V2：从评分契约修复到多路召回的算法故事线

## 目标与边界

V2 的目标不是追认某个已经观察到的分数，而是建立一条可以部署、可以解释、可以停止的推荐算法迭代链。V1 的封存 Test 继续关闭：V2 的开发、模型选择和算法决策只使用新的 Train/Validation 工件。

PPT 更新暂停，直到这条路线形成完整的工程、数学、实验和部署证据。

## V2 的起点：训练—服务评分契约

V1 的训练 logits 是

\[
s_{train}(u,i)=\hat{z}_u^\top v_i,
\]

其中用户向量 \(\hat{z}_u\) 归一化，电影向量 \(v_i\) 不归一化。导出和在线检索却使用

\[
s_{serve}(u,i)=\hat{z}_u^\top \hat{v}_i.
\]

这不是无害的实现细节：\(\|v_i\|\) 会改变排序。`training_raw` 的 Validation 实验证明该差异会提高总体 Recall/NDCG，却同时伤害 AQ3、活动差距和尾部。因此它被保留为有效的工程诊断，不能部署。

### V2 的修复契约

V2 统一使用 scaled cosine retrieval：

\[
\hat{z}_u=\frac{f_\theta(x_u)}{\|f_\theta(x_u)\|_2},\qquad
\hat{v}_i=\frac{v_i}{\|v_i\|_2},\qquad
s(u,i)=\tau\hat{z}_u^\top\hat{v}_i.
\]

\(\tau>0\) 是训练时的 logit scale；它对 Top-K 排序不产生影响，但避免把 softmax logits 限死在 \([-1,1]\)。训练、离线 Validation、向量导出和线上检索必须共享该契约及其版本元数据。

### 必须先完成的工程验收

1. `compute_full_logits` 与导出 item 向量的点积（乘以 \(\tau\)）逐元素一致。
2. 导出工件携带 `scoring_contract=scaled_cosine_v2`；在线加载器拒绝不匹配的 user/item 工件。
3. 训练、离线评估和服务端各有回归测试；任何包含 Test 的选择工件在训练或评估前 fail-closed。
4. 在新外部目录中重训 V2 aligned Baseline，按 Validation loss 早停；不覆盖 V1。

## V2 评价协议：先测量，再冻结门槛

V1 的五项门槛适用于 V1 的已部署评分契约，不能事后为 `training_raw` 放宽；它们也不应在没有测量 V2 aligned Baseline 前机械照搬为 V2 的业务合同。

V2 将按以下顺序执行：

1. 预先固定评分契约、切分、指标定义、早停规则和随机种子集合。
2. 训练 aligned Baseline V2，并只用 Validation 计算整体、AQ3、AQ0−AQ3 gap 与 Tail PQ0。
3. 对 Validation 用户做 paired bootstrap，报告指标差值及置信区间；对需要重训的候选至少使用 3 个 seed。
4. 在任何算法候选结果出现前，以 V2 Baseline、产品目标和可检测效应设定新的联合准入：总体提升、AQ3/尾部非劣、gap 不恶化。门槛和置信区间决策规则冻结后不得回改。

这样做把“严格”从任意的五个绝对数，升级为可解释的效应大小与不确定性约束；同时不追认已经看过的任何候选。

## 算法迭代顺序

### 阶段 A：统一双塔 Baseline V2

这是契约修复，不是通过改变历史长度或注意力获得的表面提升。它给出新的、可部署的质量锚点，并验证训练—评估—服务三者一致。

### 阶段 B：多路召回，先从可解释 I2I 开始

教材中的 ItemCF、Swing、Item2Vec 与双塔召回可形成互补通道。V2 首先实现 Train-only ItemCF：

\[
w_{ij}=\frac{\sum_{u\in U_{ij}}\frac{1}{\log(1+|I_u|)}}{\sqrt{N_iN_j}},\qquad
S(u,j)=\sum_{i\in H_u}r(i,u)w_{ij}.
\]

其中 \(U_{ij}\) 是同时交互过 \(i,j\) 的用户，\(N_i\) 是 Train 中物品频次，\(r(i,u)\) 是预先固定的近期权重。它只从 Train 共现统计建索引，天然适合解释“因为你最近看过 X，所以补充召回 Y”。

离线先评估每个通道的 Recall@K、覆盖率、与双塔候选的增量召回；再以固定配额合并候选并由已有 DeepFM 精排。这是真实的工业级多路召回，不是把多个模型硬拼。

若 ItemCF 的增量覆盖有证据，再以相同接口加入 Item2Vec 作为学习式 I2I 通道；它与 ItemCF 的对照能够回答“序列语义是否超过共现统计”。

### 阶段 C：更强模型只在诊断支持时引入

教材中的 MIND 比继续修改单向量 attention 更适配已观察到的高活跃用户难题：它用 \(K\) 个兴趣向量而不是把所有行为压成一个向量。

\[
\{z_u^{(1)},\ldots,z_u^{(K)}\}=\mathrm{Routing}(H_u),\qquad
s(u,i)=\max_k\ \tau\,z_u^{(k)\top}\hat v_i.
\]

它是新的多兴趣召回假设族，不是已经关闭的 History-20 attention/pooling 变体。只有在 V2 Baseline 的切片与多路召回诊断显示“单兴趣压缩”仍是主要瓶颈时，才立项、预注册并训练 MIND；否则不为了“更厉害”而上模型。

## 部署闭环

每个通过 V2 联合准入的候选都需要：新部署目录、模型与 item 向量的合约元数据/哈希、在线资源加载校验、双塔 + ItemCF 候选去重与来源记录、DeepFM 精排和重排的端到端 API 冒烟验证。V1 工件、封存 Test 与已有实验目录均不覆盖。

## 终止规则

- 每个算法假设族最多两个正式候选；失败原因相同则关闭该族。
- 任一候选不满足 V2 冻结的联合规则，不部署、不重新解释门槛。
- 连续两个独立算法方向未通过时，停止模型搜索，交付对照、诊断与部署证据。
