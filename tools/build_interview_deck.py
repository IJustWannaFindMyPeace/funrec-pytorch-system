"""Build the factual FunRec interview deck from frozen project evidence."""
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

OUT = Path("docs/presentation/FunRec_Interview_Story.pptx")
SLIDES = [
    ("FunRec：从工程迁移到可信推荐迭代", ["MovieLens 1M · PyTorch YouTubeDNN + DeepFM", "主线：协议可信 → 证据驱动 → 受控迭代", "所有优化只用 Validation；封存 Test 不参与选型"]),
    ("端到端推荐链路", ["多路召回：YouTubeDNN / user preference", "候选排序：DeepFM pointwise BCE", "重排：类型与年代多样性；容器内同工件验证"]),
    ("数据与时序协议", ["1,000,209 交互 · 6,040 用户 · 3,883 电影", "每用户时序 Train / Validation / Test", "Train 982,089；Validation / Test 各 6,040"]),
    ("先审计，再相信指标", ["修复 36,396 个跨 split 用户—电影重叠与标签冲突", "训练期计算标签阈值；Validation 用于选择 checkpoint", "Baseline Test 仅运行一次，随后封存"]),
    ("Baseline：真实但有限", ["YouTubeDNN：History-10 masked mean，最佳 Epoch 14", "唯一 Test：Recall@10 0.137086；NDCG@10 0.070298", "DeepFM Test ROC-AUC 0.841382"]),
    ("Validation 诊断定位弱点", ["AQ3 Recall@10 0.105123；AQ0 0.198423", "AQ0−AQ3 gap 0.093300", "Tail PQ0 Recall@10 0.079396", "这些是相关性，不直接宣称兴趣漂移因果"]),
    ("失败对照：History-20 masked mean", ["Validation Recall@10 0.143543；NDCG@10 0.072490", "仅通过 5 项预注册门槛中的 1 项", "结论：validation_rejected；Test 未访问"]),
    ("遮罩诊断约束了解释", ["full-20 / recent-10 / older-10 Recall@10：0.143543 / 0.122682 / 0.071192", "仅满足 1/4 个“简单等权稀释”判据", "因此下一步提高聚合表达，而不是删除旧历史"]),
    ("候选设计：个性化位置注意力", ["唯一变量：History-20 + personalized attention", "静态用户特征产生 query；movie / genre 独立 attention", "左 padding、位置 embedding、masked softmax；其他训练条件冻结"]),
    ("Attention-20：预注册与训练", ["最多 30 epochs，patience=3，seed=42", "按 Validation loss 选择 checkpoint；Epoch 20 最佳", "Epoch 23 早停；Test accessed=false"]),
    ("Attention-20：总体指标显著提升", ["Recall@10 0.199007（门槛 ≥ 0.161954）", "NDCG@10 0.101495（门槛 ≥ 0.072737）", "AQ3 0.139721、Tail PQ0 0.120735：均通过"]),
    ("但不能只看总体：按规则拒绝", ["AQ0−AQ3 gap = 0.120463", "冻结上限 = 0.088300", "4/5 通过仍不足晋级：validation_rejected", "不修改门槛，不打开 Test"]),
    ("机制诊断与下一假设", ["AQ3 movie attention 更分散：有效历史 9.628 vs AQ0 8.226", "AQ3 最近 5 项权重更低：0.702 vs 0.756", "仅为 Validation 相关性：保留双时间尺度、学习近期—较早融合，需重新预注册"]),
    ("面试总结：可信闭环优先", ["问题：迁移后的指标能否信任？", "证据：时序协议、封存 Test、切片诊断、失败实验", "决策：总体变好但 gap 超限，拒绝 Attention-20", "下一步：新单变量假设 → 预注册 → Validation 决策 → 再讨论最终 Test"]),
]

def add_text(slide, text, left, top, width, height, size, color, bold=False):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    p = box.text_frame.paragraphs[0]
    p.text, p.font.size, p.font.bold = text, Pt(size), bold
    p.font.color.rgb, p.font.name = RGBColor(*color), "Aptos"
    return box

def build():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists(): raise FileExistsError(f"Refusing to overwrite: {OUT}")
    prs = Presentation(); prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    for i, (title, bullets) in enumerate(SLIDES, 1):
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        bg = slide.background.fill; bg.solid(); bg.fore_color.rgb = RGBColor(12, 20, 38)
        accent = slide.shapes.add_shape(1, Inches(.55), Inches(.55), Inches(.12), Inches(1.0))
        accent.fill.solid(); accent.fill.fore_color.rgb = RGBColor(41, 197, 164)
        add_text(slide, title, .9, .55, 11.8, .7, 27, (245,248,252), True)
        add_text(slide, f"FUNREC / {i:02d}", .9, 1.35, 5, .3, 10, (112, 137, 168), True)
        box = slide.shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(11.2), Inches(4.6)); tf=box.text_frame; tf.clear()
        for n, bullet in enumerate(bullets):
            p = tf.paragraphs[0] if n == 0 else tf.add_paragraph(); p.text = bullet; p.level=0; p.font.size=Pt(20); p.font.color.rgb=RGBColor(220,229,240); p.font.name="Aptos"; p.space_after=Pt(18)
        add_text(slide, "真实实验结果 / Validation 诊断 / 未验证假设明确区分", .9, 7.05, 11, .2, 9, (112,137,168))
    prs.save(OUT)
    print(OUT)
if __name__ == "__main__": build()
