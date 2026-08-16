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
12. History-20 masked mean：预注册对照仅通过 1/5，作为失败实验保留；Test 继续封存。
13. 遮罩诊断：full-20 / recent-10-only / older-10-only 推翻“简单等权稀释”的解释。
14. Attention-20：唯一变量为个性化位置注意力；4/5 通过但 AQ0−AQ3 gap 超限，按预注册规则拒绝。
15. 迭代页模板：现象 → 假设 → 控制变量 → Validation → 机制 → 决策；明确区分真实结果、诊断相关性与未验证假设。

最终 PPT 必须呈现真实的 Attention-20 `validation_rejected` 结果；不使用 Test 选择方向，也不把后续候选的指标提前写入。

## 当前交付状态

- 已生成真实 14 页文件：`docs/presentation/FunRec_Interview_Story.pptx`。
- 生成器：`tools/build_interview_deck.py`；结构校验确认 14 页且每页具有标题、正文和页脚元素。
- 当前 VS Code 环境未发现可调用的 PowerPoint 或 LibreOffice，故尚未完成逐页渲染视觉检查；不得将结构校验描述为已渲染检查。
