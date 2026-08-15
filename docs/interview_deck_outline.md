# Baseline V1 面试 PPT 骨架

1. 业务目标：从开源工程迁移到可复现 PyTorch 推荐链路。
2. 数据与防泄漏协议：MovieLens 1M、时序三段切分、Test 封存。
3. 系统漏斗：多路召回 → DeepFM 精排 → 多样性重排。
4. YouTubeDNN：用户塔、3,883 类 full softmax、在线向量检索。
5. DeepFM：一阶、共享二阶 embedding、FM 与 DNN 共用 BCE。
6. Baseline 结果：Recall/NDCG、Popularity 对照、AUC 与时序差距。
7. 数据与工程审计：跨 split 冲突归零、同工件哈希、在线闭环。
8. 未决问题：召回、精排、时序泛化的 Validation-only 诊断矩阵。
9. 迭代页模板：现象 → 假设 → 控制变量 → Validation → 机制 → 决策。

最终优化完成后统一视觉与叙事；每次有效实验追加一页，不用 Test 选择方向。
