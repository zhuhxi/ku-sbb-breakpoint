"""Final presentation deck (6 min talk, 9 slides). python reports/final/build_deck.py -> reports/final/BreakPoint_final.pptx"""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_LABEL_POSITION, XL_MARKER_STYLE

NAVY, TEAL, ORANGE, INK, MUTED, LIGHT, WHITE, TINT = (RGBColor(0x1F, 0x2A, 0x44), RGBColor(0x0F, 0x7C, 0x80), RGBColor(0xE0, 0x7A, 0x1F),
    RGBColor(0x1F, 0x29, 0x37), RGBColor(0x6B, 0x72, 0x80), RGBColor(0xF2, 0xF4, 0xF7), RGBColor(0xFF, 0xFF, 0xFF), RGBColor(0xE8, 0xF3, 0xF3))
FONT = "Calibri"
P = Presentation(); P.slide_width, P.slide_height = Inches(13.333), Inches(7.5); BL = P.slide_layouts[6]
N_MAIN = 8


def tb(s, x, y, w, h, text, size=16, bold=False, color=INK, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, italic=False):
    t = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); f = t.text_frame; f.word_wrap = True; f.vertical_anchor = anchor
    f.margin_left = f.margin_right = Inches(0.05); f.margin_top = f.margin_bottom = Inches(0.03)
    lines = text if isinstance(text, list) else [text]
    for i, ln in enumerate(lines):
        p = f.paragraphs[0] if i == 0 else f.add_paragraph(); p.alignment = align
        runs = ln if isinstance(ln, list) else [(ln, {})]
        for rt, o in runs:
            r = p.add_run(); r.text = rt; r.font.name = FONT; r.font.size = Pt(o.get("size", size)); r.font.bold = o.get("bold", bold)
            r.font.italic = o.get("italic", italic); r.font.color.rgb = o.get("color", color)
        p.space_after = Pt(o.get("after", 6)) if runs else Pt(6)
    return t


def box(s, x, y, w, h, text, fill=LIGHT, color=INK, size=14, bold=False, shape=MSO_SHAPE.ROUNDED_RECTANGLE, align=PP_ALIGN.CENTER):
    b = s.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h)); b.fill.solid(); b.fill.fore_color.rgb = fill; b.line.fill.background()
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE: b.adjustments[0] = 0.12
    b.shadow.inherit = False
    f = b.text_frame; f.word_wrap = True; f.vertical_anchor = MSO_ANCHOR.MIDDLE; f.margin_left = f.margin_right = Inches(0.08)
    lines = text if isinstance(text, list) else [text]
    for i, ln in enumerate(lines):
        p = f.paragraphs[0] if i == 0 else f.add_paragraph(); p.alignment = align
        runs = ln if isinstance(ln, list) else [(ln, {})]
        for rt, o in runs:
            r = p.add_run(); r.text = rt; r.font.name = FONT; r.font.size = Pt(o.get("size", size)); r.font.bold = o.get("bold", bold); r.font.color.rgb = o.get("color", color)
    return b


def arrow(s, x, y, w=0.45, h=0.32, color=MUTED):
    a = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x), Inches(y), Inches(w), Inches(h)); a.fill.solid(); a.fill.fore_color.rgb = color; a.line.fill.background(); return a


def content(title, n, notes):
    s = P.slides.add_slide(BL); s.background.fill.solid(); s.background.fill.fore_color.rgb = WHITE
    tb(s, 0.6, 0.35, 12.1, 1.0, title, size=30, bold=True, color=NAVY, anchor=MSO_ANCHOR.MIDDLE)
    tb(s, 0.6, 7.0, 8, 0.35, "Team BreakPoint  ·  KU-SBB final", size=10, color=MUTED)
    tb(s, 11.7, 7.0, 1.0, 0.35, f"{n}", size=10, color=MUTED, align=PP_ALIGN.RIGHT)
    s.notes_slide.notes_text_frame.text = notes
    return s


def style_chart(ch, legend=False, vmin=None, vmax=None, fmt='0.00', gridcolor="E5E7EB"):
    ch.has_title = False; ch.font.name = FONT; ch.font.size = Pt(12); ch.font.color.rgb = MUTED; ch.has_legend = legend
    if legend: ch.legend.position = XL_LEGEND_POSITION.BOTTOM; ch.legend.include_in_layout = False; ch.legend.font.size = Pt(12)
    va = ch.value_axis; va.has_major_gridlines = True; va.major_gridlines.format.line.color.rgb = RGBColor.from_string(gridcolor)
    va.format.line.fill.background(); va.tick_labels.number_format = fmt; va.tick_labels.number_format_is_linked = False
    if vmin is not None: va.minimum_scale = vmin
    if vmax is not None: va.maximum_scale = vmax
    ca = ch.category_axis; ca.format.line.color.rgb = RGBColor.from_string("D1D5DB"); ca.has_major_gridlines = False


# 1 ---------------------------------------------------------------- title
s = P.slides.add_slide(BL); s.background.fill.solid(); s.background.fill.fore_color.rgb = WHITE
tb(s, 0.6, 0.55, 12.1, 1.0, "Let the generator teach the model", size=40, bold=True, color=NAVY)
tb(s, 0.6, 1.45, 12.1, 0.6, "Structural-break detection: rebuild how the data was made, then learn from synthetic data", size=18, color=MUTED)
s.shapes.add_picture("reports/final/hero/cover.png", Inches(0.45), Inches(2.3), width=Inches(12.4))
tb(s, 0.6, 6.15, 12.1, 0.5, [[("0.8779", {"bold": True, "size": 20, "color": RGBColor(0xFD, 0x59, 0x56)}), ("  public AUC     ", {"size": 15, "color": INK}),
    ("0.8741", {"bold": True, "size": 20, "color": NAVY}), ("  out-of-fold AUC     ", {"size": 15, "color": INK}),
    ("#2", {"bold": True, "size": 20, "color": NAVY}), ("  of 6 finalists after qualification", {"size": 15, "color": INK})]])
tb(s, 0.6, 6.85, 12, 0.4, "Team BreakPoint  ·  KU Applied AI Challenge 2026 (KU-SBB)  ·  Final, 11 November", size=11, color=MUTED)
s.notes_slide.notes_text_frame.text = ("[~15 s] Hello, we are team BreakPoint. Our one idea: instead of only learning from 24,000 labels, we rebuilt the process that "
    "generated the data, and let synthetic data teach the model what the labels cannot. Our best submission scores 0.878 public AUC.\n"
    "中文提示：开场一句话讲核心思路：反推生成器，让假数据来教模型。报 0.878。")

# 2 ---------------------------------------------------------------- problem
s = content("Breaks range from obvious to invisible", 2,
    "[~45 s] Each series has a known boundary; we predict whether the process changed there. It can be the level, the variance, the "
    "dependence, or the noise itself. The first panels are easy. Now look at the last one: there is a level shift of a quarter of a standard "
    "deviation — can you see it? In our calibrated generator about one in three level or variance breaks is this small. 24,000 labels cannot "
    "teach every such change in every kind of series, so we needed another source of supervision.\n"
    "中文提示：先讲任务；指着最后一格问“看得出吗？”——0.27 个标准差的水平变化；生成器里约 1/3 的水平/方差断点这么小；所以需要标签以外的监督来源。")
s.shapes.add_picture("reports/final/figmirror/A_examples/figure.png", Inches(0.5), Inches(1.35), height=Inches(5.3))
tb(s, 0.5, 6.65, 9.6, 0.35, "Synthetic examples from our generator. Dashed line = known boundary n0; band = segment mean ± 1 sd.", size=11, color=MUTED, italic=True)
for i, (h, b) in enumerate((("The task", "Known boundary: did the process change there? ROC-AUC on 4,000 test series."),
                            ("What can change", "Level, variance, dependence or the noise distribution."),
                            ("Why it is hard", "About 1 in 3 level / variance breaks is as small as the last panel."))):
    box(s, 10.35, 1.5 + i * 1.8, 2.55, 1.65, [[(h, {"bold": True, "size": 15, "color": NAVY})], [(b, {"size": 12.5})]], fill=LIGHT, align=PP_ALIGN.LEFT)

# 3 ---------------------------------------------------------------- generator
s = content("We reverse-engineered the generator behind the data", 3,
    "[~50 s] Key choice. By fitting every real series we inferred the recipe: a structure, a noise type, and optionally one break. We kept "
    "adjusting the recipe until a classifier could hardly tell real from fake: its AUC fell from 0.66 to 0.58, where 0.5 means indistinguishable. "
    "With a realistic generator we get unlimited labelled data, with all hidden parameters known.\n"
    "中文提示：逐条拟合真实序列推出配方：结构→噪声→断点；真实度（真假判别 AUC）0.66 降到 0.58（0.5=分不出）；从此有无限多带标签、参数全知的数据。")
s.shapes.add_picture("reports/final/figmirror/B_recipe/figure_slide.png", Inches(0.5), Inches(1.5), height=Inches(4.55))
tb(s, 0.5, 6.15, 6.6, 0.6, "Recipe fitted by matching real series (ku_gen.py in the repo): structure → noise → optional break at n0", size=12, color=MUTED, italic=True)
s.shapes.add_picture("reports/final/figmirror/C2_realism/figure.png", Inches(7.55), Inches(1.45), height=Inches(4.6))
tb(s, 7.55, 6.15, 5.1, 0.6, "Mean of 3 seeds, noise ±0.003", size=12, color=MUTED, italic=True)

# 4 ---------------------------------------------------------------- pipeline
s = content("Label-free signals feed a TabPFN ensemble", 4,
    "[~50 s] The pipeline. Left: real series get classic statistics — two-sample tests and likelihood-ratio break tests on AR residuals — computed "
    "on the series and on its time-reversed copy. Middle: the generator trains three things without any real label: estimators of the series type "
    "and break size, a CNN and a fine-tuned Chronos-2 that detect breaks zero-shot at AUC 0.84 and 0.85, and a noise classifier per segment. "
    "Right: TabPFN, a pretrained tabular transformer, reads these features; a second TabPFN stacks the arms.\n"
    "中文提示：左=真实序列统计特征(+时间反转)；中=生成器训练的三样东西，全程不用真实标签；右=TabPFN 分支 + TabPFN 堆叠。")
s.shapes.add_picture("reports/final/figmirror/G_pipeline/figure.png", Inches(1.05), Inches(1.35), height=Inches(5.6))

# 5 ---------------------------------------------------------------- validation
s = content("Every gain had to survive fold-clean validation", 5,
    "[~45 s] Validation. All label-dependent steps — feature selection, TabPFN context, stacker — are fitted inside the training folds. Each change "
    "is screened fast, then confirmed at full settings with a paired bootstrap; quick gains usually halve, and we only kept what survived. The public "
    "leaderboard uses about 1,500 series, so its noise is about ±0.009 — larger than most of our improvements — so we decided by out-of-fold AUC. "
    "The chart shows both rising together across our submissions.\n"
    "中文提示：所有用标签的步骤都在折内；快速筛→正式确认+配对 bootstrap，快速增益通常减半；公榜只有约 1500 条、噪声 ±0.009，按 OOF 决策；图里 OOF 和公榜同步上升。")
rules = [("Fold-clean", "5-fold stratified CV; selection, TabPFN context and stacker fitted inside each training fold"),
         ("Two-stage test", "Fast screen, then full-setting confirm with a paired bootstrap CI; quick gains usually halve"),
         ("Decide by OOF", "Public LB = ~1,520 series, noise ±0.009 (larger than most gains); OOF on 24,000 series is ±0.002")]
for i, (h, b) in enumerate(rules):
    y = 1.55 + i * 1.7
    c = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(0.6), Inches(y + 0.1), Inches(0.6), Inches(0.6)); c.fill.solid(); c.fill.fore_color.rgb = TEAL; c.line.fill.background()
    c.text_frame.text = str(i + 1); pp = c.text_frame.paragraphs[0]; pp.alignment = PP_ALIGN.CENTER; pp.runs[0].font.size = Pt(18); pp.runs[0].font.bold = True; pp.runs[0].font.color.rgb = WHITE; pp.runs[0].font.name = FONT
    tb(s, 1.4, y, 4.3, 1.5, [[(h, {"bold": True, "size": 18, "color": NAVY})], [(b, {"size": 14})]])
s.shapes.add_picture("reports/final/figmirror/C_curves/figure.png", Inches(6.95), Inches(1.35), height=Inches(5.05))
tb(s, 6.2, 6.5, 6.5, 0.45, "D = CPU-auditable LightGBM, G2 = qualification, X = best (blendXFS)", size=12, color=MUTED, italic=True)

# 6 ---------------------------------------------------------------- oracle
s = content("An oracle experiment showed where the fixable error is", 6,
    "[~40 s] How we found our last gain. On synthetic data we know every hidden parameter. So we asked: if the model were told the true "
    "value, how much would AUC rise? The answer points to what is worth estimating better. The noise distribution before the boundary was worth "
    "+0.007 — more than all other parameters together.\n"
    "中文提示：假数据里所有隐藏参数已知→把真值当特征给模型→AUC 涨多少=把它估准值多少；噪声分布最值钱 +0.007，比其他加起来还多。")
s.shapes.add_picture("reports/final/figmirror/D_oracle/figure.png", Inches(0.5), Inches(1.4), width=Inches(6.9))
tb(s, 0.6, 6.5, 6.6, 0.45, "LightGBM on 90k synthetic series, tested on 30k held out; baseline 0.815", size=12, color=MUTED, italic=True)
tb(s, 7.7, 1.3, 5.0, 0.45, "How the oracle works", size=14, bold=True, color=INK)
for i, (h, b) in enumerate((("Synthetic data", "every hidden parameter of every series is known"), ("Tell the model", "add the true value as an extra feature, retrain"),
                            ("Measure", "AUC gain = what better estimation of it is worth"))):
    y = 1.85 + i * 1.25
    c = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(7.7), Inches(y + 0.15), Inches(0.6), Inches(0.6)); c.fill.solid(); c.fill.fore_color.rgb = TEAL; c.line.fill.background()
    c.text_frame.text = str(i + 1); pp = c.text_frame.paragraphs[0]; pp.alignment = PP_ALIGN.CENTER; pp.runs[0].font.size = Pt(18); pp.runs[0].font.bold = True; pp.runs[0].font.color.rgb = WHITE; pp.runs[0].font.name = FONT
    tb(s, 8.45, y, 4.25, 1.1, [[(h, {"bold": True, "size": 17, "color": NAVY})], [(b, {"size": 15})]])
box(s, 7.7, 5.65, 5.0, 0.95, [[("Biggest lever: ", {"bold": True, "size": 15, "color": ORANGE}), ("the noise distribution before the boundary", {"size": 15})]], fill=LIGHT, align=PP_ALIGN.LEFT)

# 7 ---------------------------------------------------------------- noise classifier
s = content("A noise classifier per segment turned this into score", 7,
    "[~45 s] So we trained a classifier on synthetic segments — after removing the AR structure — that predicts the noise type, tail weight "
    "and volatility clustering. We apply it before and after the boundary and compare. In this example the spread is the same but the tails are "
    "heavier after the boundary — a variance test misses it. It improved every level: best LightGBM, best TabPFN arm, the stacker, and the public "
    "score from 0.876 to 0.878.\n"
    "中文提示：在假数据上训练“看噪声”的分类器（去掉 AR 结构后）；断点前后各判一次再比较；例子：方差相同、尾部变重，方差检验看不出；四层都涨，公榜 0.876→0.878。")
tb(s, 0.6, 1.3, 6.3, 0.45, "Same spread, heavier tails after the break", size=14, bold=True, color=INK)
s.shapes.add_picture("reports/final/figmirror/E_noise/figure.png", Inches(0.5), Inches(2.3), width=Inches(6.4))
tb(s, 0.6, 6.45, 6.2, 0.45, "One synthetic series; AR(5) residuals, each segment standardized to sd 1", size=12, color=MUTED, italic=True)
tb(s, 7.1, 1.3, 5.6, 0.45, "Same pipeline, plus the noise classifier", size=14, bold=True, color=INK)
for i, (lab, a, b) in enumerate((("Best LightGBM", "0.8567", "0.8588"), ("Best TabPFN arm", "0.8716", "0.8729"), ("Stacker (OOF)", "0.8733", "0.8741"), ("Public AUC", "0.8764", "0.8779"))):
    y = 1.9 + i * 1.12
    box(s, 7.1, y, 5.6, 0.95, [[(lab, {"size": 15, "color": INK})]], fill=LIGHT, align=PP_ALIGN.LEFT)
    tb(s, 9.6, y + 0.18, 3.0, 0.6, [[(a + "  →  ", {"size": 18, "color": MUTED}), (b, {"size": 22, "bold": True, "color": TEAL})]], align=PP_ALIGN.RIGHT)
tb(s, 7.1, 6.45, 5.6, 0.45, "LightGBM gain confirmed: +0.0021, 95% CI +0.0011 to +0.0032", size=12, color=MUTED, italic=True)

# 8 ---------------------------------------------------------------- limitations
s = content("Limitations: hard series, compute, generator", 8,
    "[~40 s] Limitations. Random-walk series stay hard, AUC 0.71, and the generator is equally weak there, so this looks intrinsic. Short series are "
    "harder too. The best model is GPU-heavy: hundreds of arms, although the best single arm is within 0.0012 of the stacker — most arms are "
    "redundant. For the 2-core CPU audit we keep a bit-exact LightGBM route at 0.861. And the generator is still detectable: synthetic breaks are "
    "slightly smaller than real ones.\n"
    "中文提示：RW 序列 0.71（假数据上也一样难，本质难）；短序列难；GPU 重、分支冗余（单分支只差 0.0012）；CPU 审核有逐位可复现的 LightGBM 路线 0.861；生成器仍可被识别，假断点偏小。")
s.shapes.add_picture("reports/final/figmirror/F_limits/figure_slide.png", Inches(0.4), Inches(2.0), width=Inches(7.7))
tb(s, 0.4, 1.35, 7.7, 0.45, "Out-of-fold AUC: errors concentrate in random walks and short series", size=14, bold=True, color=INK)
lims = [("Hard series", "Random walks (0.71) are equally hard on synthetic data, so this looks intrinsic; short series are harder too."),
        ("Compute", "Best model needs a GPU and 296 arms; the best single arm is within 0.0012, so most arms are redundant."),
        ("Audit route", "A bit-exact LightGBM route (0.861 public) runs in 31 s on 2 CPU cores; the GPU route is documented step by step."),
        ("Generator", "Still detectable (0.58); synthetic breaks are slightly smaller than real ones.")]
for i, (h, b) in enumerate(lims):
    y = 1.4 + i * 1.32
    tb(s, 8.35, y, 4.6, 1.25, [[(h, {"bold": True, "size": 16, "color": NAVY})], [(b, {"size": 13})]])

# 9 ---------------------------------------------------------------- takeaway
s = P.slides.add_slide(BL); s.background.fill.solid(); s.background.fill.fore_color.rgb = WHITE
tb(s, 0.6, 0.35, 12.1, 1.0, "Model how the data was made, then learn from it", size=32, bold=True, color=NAVY, anchor=MSO_ANCHOR.MIDDLE)
s.shapes.add_picture("reports/final/hero/loop.png", Inches(0.45), Inches(1.45), width=Inches(12.4))
tb(s, 0.6, 6.25, 12.1, 0.6, "Thank you — questions welcome", size=24, bold=True, color=NAVY, align=PP_ALIGN.CENTER)
# added 11-11 (slide update): code + checkpoint link line, hyperlinked "link" run (see slides/README.md for the URL)
s.notes_slide.notes_text_frame.text = ("[~15 s] To sum up: model how the data was made, learn from it without labels, and measure what is fixable before "
    "building. Thank you, we are happy to take questions.\n中文提示：三句话总结，然后请提问。")

P.save("reports/final/BreakPoint_final.pptx"); print("saved", len(P.slides._sldIdLst), "slides")
