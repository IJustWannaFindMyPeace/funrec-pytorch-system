# FunRec PyTorch 算法项目故事

## 一句话版本

将 FunRec 的 YouTubeDNN 与 DeepFM 从 TensorFlow 迁移到 PyTorch并打通离线训练、部署、在线服务和前端；随后发现原评估存在 Validation/Test 复用和跨时间负样本矛盾，重建严格时序实验协议，冻结首个真实 Baseline，再用 Validation 分层诊断驱动召回—排序—重排迭代。

## 已完成

### 工程迁移

- PyTorch YouTubeDNN 和 DeepFM。
- CUDA 训练、checkpoint 恢复、本地部署和在线推理。
- Docker Compose 全栈与前端体验。
- 112 项后端测试。

### 可信实验协议

- 独立 Train/Validation/Test。
- 修复 36,396 个跨 split 用户—电影重叠及未来正样本误采问题。
- Validation/Test 用户覆盖恢复到 6,040/6,040。
- Test 只运行一次，并由 SHA256 封存。

### 正式离线 Baseline

- 默认 3 epochs 被证明欠训练；只依据 Validation 扩展训练预算。
- YouTubeDNN 在 Epoch 14 最优，Test Recall@10=`0.137086`、NDCG@10=`0.070298`。
- Recall@10 为 popularity 的 `7.0769×`，严格相对提升 `607.69%`。
- DeepFM 在 Epoch 5 最优，Validation AUC=`0.866237`，Test AUC=`0.841382`。
- Validation→Test AUC 下降 `0.024855`，形成下一阶段诊断问题，但尚不能武断归因。

### 在线同工件闭环

- 训练产物、部署目录和 Docker 容器内的 YouTubeDNN、物品向量、电影 ID、DeepFM 四个核心工件哈希一致。
- 在线请求实际经过 `youtube_dnn`/`user_preference` 召回、`deepfm` 精排以及类型和年代多样性重排，返回 20 个不重复结果。
- 在线矩阵多出的第 0 行是适配一基电影 ID 的全零 padding/OOV 向量，不改变真实电影数量和模型工件。
- `EXP-000` 已形成可追溯的离线评估—部署复制—容器挂载—在线推理闭环。

### Validation 诊断与受控迭代

- Validation 分群确认高活跃 AQ3 与长尾 PQ0 是 Retrieval 弱点；没有把这种相关性直接宣传为兴趣漂移因果。
- History-20 + masked mean 只通过 1/5 个预注册门槛，被拒绝；同 checkpoint 的遮罩诊断也不支持“旧历史简单等权稀释近期兴趣”。
- 唯一改变聚合机制的正式 Attention-20 在 Epoch 23 按 patience=3 早停，选择 Epoch 20 的最低 Validation loss checkpoint。
- Attention-20 的 Recall@10=`0.199007`、NDCG@10=`0.101495`、AQ3 Recall@10=`0.139721`、Tail PQ0 Recall@10=`0.120735` 均越过冻结门槛；但 AQ0−AQ3 gap=`0.120463` 超过上限 `0.088300`。
- 因此按全五项通过规则记录为 `validation_rejected`。Test 始终未访问；失败结果和一次工具会话超时后的无覆盖恢复均已保留。

## 尚未完成

- 在不改变既有门槛且不打开 Test 的前提下，提出并预注册下一项单变量候选。
- 仅对未来满足全部 Validation 准入门槛的候选开展多随机种子复验与端到端精度—延迟权衡。
- 生成、渲染并逐页检查真实面试 PPTX；其中必须把 Attention-20 标为失败的 Validation 候选，而不是成功指标。

## 面试叙事主线

1. **迁移**：统一 PyTorch 技术栈和 GPU 训练/推理。
2. **质疑指标**：发现原结果复用了评估集，不能支撑算法结论。
3. **重建协议**：三段时序切分、训练期标签阈值、Test 封闭。
4. **对抗性审计**：发现未来正反馈被采为早期负样本，以及评估用户丢失。
5. **修复数据**：跨 split 冲突归零、用户覆盖恢复完整。
6. **校准收敛**：默认 3 epochs 欠训练；YouTubeDNN/DeepFM 分别在 Epoch 14/5 early stop。
7. **冻结 V0**：真实 Recall/NDCG/AUC、日志、checkpoint、哈希和唯一 Test 全部可追溯。
8. **分析后优化**：只用 Validation 切片定位长尾、低活跃或时序泛化瓶颈，再做控制变量实验。
9. **落地**：部署冻结工件，验证容器内哈希与线上 YouTubeDNN→DeepFM 策略一致。
10. **诊断迭代**：只使用 Validation 建立用户、物品、时序和漏斗切片，再由证据选择第一个算法改进。

## 面试中必须主动说明的边界

- 数据协议修复不是算法提升。
- 迁移验收指标与 V0 的切分不同，不能直接比较百分比收益。
- Single-target 下 Recall 与 HitRate 相同。
- DeepFM ROC-AUC 是采样候选区分指标，不等于线上 Top-K 收益。
- 当前结果来自 Seed 42；最终方案需要多 seed。
- 离线 Test 不是线上 A/B Test。

## 证据规则

每个数字必须能够追溯到实验注册表、脱敏 metrics、run config、训练 history、checkpoint、模型哈希和 Git commit。失败实验和协议修复同样保留，因为它们体现了如何识别伪提升并建立可信结论。
