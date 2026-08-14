# FunRec 实验注册表

## 状态定义

- `planned`：已提出、未运行。
- `running`：正在运行。
- `completed`：工件和指标完整。
- `rejected`：实验无效或口径不可信，并保留原因。

## 实验总表

| ID | 状态 | Git SHA | Seed | 主要改动 | Validation | Test | 结论 |
|---|---|---|---:|---|---|---|---|
| MIGRATION-ACCEPTANCE | completed | e69ccda | 42 | TensorFlow → PyTorch 工程迁移 | Recall@10=0.100662; DeepFM AUC=0.854670 | 不适用 | 证明迁移与全链路可运行，不作为独立 Test 结果 |
| EXP-000 | running | 126641b | 42 | 严格时序三段切分、负采样冲突修复与独立 Test 入口 | 协议审计通过；112 tests | 未运行 | 数据协议已冻结，待 smoke 与正式训练 |

## 单次实验必须保存

- 实验 ID 与假设。
- Git SHA 与工作区状态。
- 数据版本、切分统计和校验摘要。
- 完整命令与配置。
- 随机种子。
- 训练 history 与最佳 epoch。
- Validation 与 Test 指标。
- 训练时间、推理延迟、显存和模型大小。
- checkpoint 与部署工件路径。
- 成功、失败及下一步解释。

## EXP-000 预注册

### 目标

建立没有 Validation/Test 复用、没有 Test 参与模型选择的可复现 PyTorch Baseline V0。

### 主要变量

本实验不做模型结构优化，只修正实验协议并补充指标。

### 成功条件

- 三段切分测试通过。
- Test 在训练结束前不可访问。
- Seed 42 全量训练完成。
- 所有指标和环境信息自动保存。
- 在线链路成功加载冻结工件。

### 数据协议审计

- 原始交互：1,000,209；用户：6,040。
- Ranking Train/Validation/Test：1,703,230 / 225,347 / 448,147 个采样样本。
- Retrieval Train/Validation/Test：982,089 / 6,040 / 6,040 个样本。
- 六类跨 split 用户—电影冲突均为 0。
- Validation/Test 用户覆盖均为 6,040。
- 审计文件：`docs/results/baseline_v0_data_audit.json`。
