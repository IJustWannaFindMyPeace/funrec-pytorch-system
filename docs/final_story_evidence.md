# FunRec 最终叙事证据清单

## 冻结决策

模型搜索停止。部署和面试展示以 `pytorch-baseline-v1` 为可信锚点；后续候选均未满足预注册的五项 Validation 联合准入门槛。封存 Test 不再访问、反序列化或重评估。

## 证据序列

| 阶段 | 结论 | 证据 |
|---|---|---|
| Baseline V1 | 可部署、可追溯的锚点；唯一 Test 核验已封存 | `pytorch-baseline-v1`；`docs/results/baseline_v0_*.json` |
| History-20 masked mean | 1/5，拒绝 | `docs/results/history20_masked_mean_validation_results.json` |
| Personalized Attention-20 | 4/5；gap 超限，拒绝 | `docs/results/attention20_validation_results.json` |
| Dual-timescale Attention-20 | 4/5；同一 gap 超限，关闭 attention/pooling 家族 | `docs/results/dualtimescale_attention20_validation_results.json` |
| Activity-balanced 原运行 | 含 Test 的旧训练工件被反序列化；协议无效、隔离 | `docs/results/activity_balanced_history10_protocol_incident.json` |
| Activity-balanced 替代运行 | 有效但 1/5，拒绝 | `docs/results/activity_balanced_history10_rerun_validation_results.json` |

## 交付边界

- 最终 PPTX：`docs/presentation/FunRec_Interview_Story_Final.pptx`。
- 已使用 Aspose.Slides 将真实 PPTX 渲染为 `docs/presentation/FunRec_Interview_Story_Final_rendered/slide-01.png` 至 `slide-15.png`，并逐页检查：15/15 页存在标题、正文和页脚，未发现空白页或版式溢出。渲染器处于评估模式，预览 PNG 带中央水印；水印不写入 PPTX，但会遮挡预览正文。因此面试前如可获得 PowerPoint/LibreOffice，仍应作一次无水印的人工放映复核。
- 可复现生成与渲染：`tools/build_interview_deck.py`、`tools/render_interview_deck.py`。
- 任何新模型方向都需要新的假设、单一核心变量、预注册配置和用户对长时间训练的确认。
