# Baseline V1 面试 PPT 骨架

1. 业务目标：从开源工程迁移到可复现 PyTorch 推荐链路。
2. 数据与防泄漏协议：MovieLens 1M、时序三段切分、Test 封存。
3. 系统漏斗：多路召回 → DeepFM 精排 → 多样性重排。
4. YouTubeDNN：用户塔、3,883 类 full softmax、在线向量检索。
5. DeepFM：一阶、共享二阶 embedding、FM 与 DNN 共用 BCE。
6. Baseline 结果：Recall/NDCG、Popularity 对照、AUC 与时序差距。
7. 数据与工程审计：跨 split 冲突归零、同工件哈希、在线闭环。
8. 诊断方法：模型窗口与原始活跃度分离，单轴切片升级为 4×4 交叉矩阵。
9. 诊断发现：高活跃退化在控制热门度后仍存在；长尾与活跃度存在交互。
10. 指标校准：PR-AUC 对照正例率，logloss 对照同组常数预测器熵。
11. 假设边界：证据支持“高活跃用户更难表示”，尚不能证明兴趣漂移是唯一原因。
12. 优化预注册：历史长度 10/20/50 → recency weighting → attention；Test 继续封存。
13. 迭代页模板：现象 → 假设 → 控制变量 → Validation → 机制 → 决策。

最终优化完成后统一视觉与叙事；每次有效实验追加一页，不用 Test 选择方向。
