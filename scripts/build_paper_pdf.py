#!/usr/bin/env python3
"""Build the sealed ECHO-CAVER research paper with ReportLab."""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch, mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "paper/ECHO_CAVER_research_archive.pdf"
FULL = json.loads((ROOT / "results/full_mmad_analysis.json").read_text())
CLEAN = json.loads((ROOT / "results/clean_panel_analysis.json").read_text())
FREB = json.loads((ROOT / "results/freb_full_reference.json").read_text())
TRAIN = json.loads((ROOT / "artifacts/training/echo_training_summary.json").read_text())

NAVY = HexColor("#0B172A")
INK = HexColor("#172033")
MUTED = HexColor("#5D6B7E")
PALE = HexColor("#EEF4F8")
GRID = HexColor("#D7E0EA")
CYAN = HexColor("#0891B2")
CYAN_LIGHT = HexColor("#CFFAFE")
BLUE = HexColor("#2563EB")
AMBER = HexColor("#D97706")
ORANGE = HexColor("#F97316")
WHITE = colors.white


class PaperDoc(BaseDocTemplate):
    def __init__(self, path: Path):
        super().__init__(
            str(path),
            pagesize=A4,
            leftMargin=18 * mm,
            rightMargin=18 * mm,
            topMargin=19 * mm,
            bottomMargin=17 * mm,
            title="[Research] ECHO-CAVER: Evidence-Tangent Residual Readers for FREB-CAVER",
            author="JaeWoo Choi",
            subject="Private research archive for ECHO-CAVER v1",
        )
        frame = Frame(
            self.leftMargin,
            self.bottomMargin,
            self.width,
            self.height,
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
        )
        self.addPageTemplates(PageTemplate(id="paper", frames=[frame], onPage=page_chrome))


def page_chrome(canvas, doc) -> None:
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(GRID)
    canvas.setLineWidth(0.6)
    canvas.line(18 * mm, height - 13 * mm, width - 18 * mm, height - 13 * mm)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, height - 10.2 * mm, "[Research] ECHO-CAVER - Private research archive")
    canvas.drawRightString(width - 18 * mm, height - 10.2 * mm, "24 August 2026")
    canvas.line(18 * mm, 11.5 * mm, width - 18 * mm, 11.5 * mm)
    canvas.drawString(18 * mm, 7.8 * mm, "Full benchmark evidence and clean-panel claim boundaries are reported separately.")
    canvas.drawRightString(width - 18 * mm, 7.8 * mm, f"{doc.page}")
    canvas.restoreState()


styles = getSampleStyleSheet()
TITLE = ParagraphStyle(
    "Title",
    parent=styles["Title"],
    fontName="Helvetica-Bold",
    fontSize=25,
    leading=30,
    alignment=TA_LEFT,
    textColor=NAVY,
    spaceAfter=9,
)
SUBTITLE = ParagraphStyle(
    "Subtitle",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=12,
    leading=17,
    textColor=MUTED,
    spaceAfter=14,
)
H1 = ParagraphStyle(
    "H1",
    parent=styles["Heading1"],
    fontName="Helvetica-Bold",
    fontSize=16,
    leading=20,
    textColor=NAVY,
    spaceBefore=10,
    spaceAfter=7,
)
H2 = ParagraphStyle(
    "H2",
    parent=styles["Heading2"],
    fontName="Helvetica-Bold",
    fontSize=11.5,
    leading=15,
    textColor=CYAN,
    spaceBefore=7,
    spaceAfter=5,
)
BODY = ParagraphStyle(
    "Body",
    parent=styles["BodyText"],
    fontName="Helvetica",
    fontSize=9.2,
    leading=13.2,
    textColor=INK,
    spaceAfter=6,
)
SMALL = ParagraphStyle(
    "Small",
    parent=BODY,
    fontSize=7.7,
    leading=10.2,
    textColor=MUTED,
)
CAPTION = ParagraphStyle(
    "Caption",
    parent=SMALL,
    alignment=TA_CENTER,
    spaceBefore=4,
    spaceAfter=7,
)
CALLOUT = ParagraphStyle(
    "Callout",
    parent=BODY,
    fontName="Helvetica-Bold",
    fontSize=10.3,
    leading=14.5,
    textColor=NAVY,
)


def p(text: str, style=BODY) -> Paragraph:
    return Paragraph(text, style)


def metric_table(rows, widths=None, align_right_from=1) -> Table:
    table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.8),
        ("LEADING", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.45, GRID),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, PALE]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4.2),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]
    if align_right_from is not None:
        commands.append(("ALIGN", (align_right_from, 1), (-1, -1), "RIGHT"))
        commands.append(("ALIGN", (align_right_from, 0), (-1, 0), "RIGHT"))
    table.setStyle(TableStyle(commands))
    return table


def architecture() -> Drawing:
    d = Drawing(480, 165)
    y, h = 79, 58
    boxes = [
        (8, 90, "Input pair", "query + normal ref", BLUE),
        (112, 98, "Frozen JUDO", "vision + decoder", BLUE),
        (224, 108, "Frozen FREB", "writers 18..26", AMBER),
        (346, 126, "ECHO readers", "layers 19..27", CYAN),
    ]
    for x, w, title, subtitle, color in boxes:
        d.add(Rect(x, y, w, h, rx=8, ry=8, fillColor=colors.white, strokeColor=color, strokeWidth=1.4))
        d.add(String(x + w / 2, y + 35, title, textAnchor="middle", fontName="Helvetica-Bold", fontSize=9, fillColor=NAVY))
        d.add(String(x + w / 2, y + 19, subtitle, textAnchor="middle", fontName="Helvetica", fontSize=7.5, fillColor=MUTED))
    for x1, x2 in [(98, 112), (210, 224), (332, 346)]:
        d.add(Line(x1, y + h / 2, x2 - 4, y + h / 2, strokeColor=HexColor("#7C93B6"), strokeWidth=1.4))
        d.add(Polygon([x2 - 4, y + h / 2 + 3, x2, y + h / 2, x2 - 4, y + h / 2 - 3], fillColor=HexColor("#7C93B6"), strokeColor=None))
    d.add(Rect(345, 18, 127, 38, rx=7, ry=7, fillColor=CYAN_LIGHT, strokeColor=CYAN, strokeWidth=1))
    d.add(String(408.5, 40, "Evidence projection", textAnchor="middle", fontName="Helvetica-Bold", fontSize=8.5, fillColor=NAVY))
    d.add(String(408.5, 26, "B <- U32 U32^T B", textAnchor="middle", fontName="Helvetica", fontSize=8, fillColor=CYAN))
    d.add(Line(408.5, 56, 408.5, 79, strokeColor=CYAN, strokeWidth=1.3))
    d.add(String(238, 4, "One runtime - no router, threshold, ensemble, crop, or post-hoc correction", textAnchor="middle", fontName="Helvetica", fontSize=7.5, fillColor=MUTED))
    return d


def delta_chart() -> Drawing:
    e = FULL["metrics"]["per_task_macro"]
    f = FREB["strict_parser_metrics"]["per_task_source_macro_accuracy"]
    order = [
        ("AD", "Anomaly Detection"),
        ("DA", "Defect Analysis"),
        ("DC", "Defect Classification"),
        ("DD", "Defect Description"),
        ("DL", "Defect Localization"),
        ("OA", "Object Analysis"),
        ("OC", "Object Classification"),
    ]
    d = Drawing(480, 210)
    zero = 177
    scale = 142
    d.add(Line(zero, 16, zero, 194, strokeColor=HexColor("#64748B"), strokeWidth=1))
    for i, (short, task) in enumerate(order):
        y = 178 - i * 25
        delta = 100 * (e[task] - f[task])
        d.add(String(8, y + 3, short, fontName="Helvetica-Bold", fontSize=8, fillColor=NAVY))
        d.add(String(31, y + 3, task, fontName="Helvetica", fontSize=7.5, fillColor=MUTED))
        width = abs(delta) * scale
        x = zero if delta >= 0 else zero - width
        color = CYAN if delta >= 0 else ORANGE
        d.add(Rect(x, y - 4, width, 12, rx=3, ry=3, fillColor=color, strokeColor=None))
        label_x = x + width + 5 if delta >= 0 else x - 5
        anchor = "start" if delta >= 0 else "end"
        d.add(String(label_x, y - 1, f"{delta:+.3f}", textAnchor=anchor, fontName="Helvetica-Bold", fontSize=7.4, fillColor=color))
    for value in (-0.5, 0.0, 0.5, 1.0, 1.5):
        x = zero + value * scale
        d.add(Line(x, 11, x, 15, strokeColor=GRID, strokeWidth=0.7))
        d.add(String(x, 1, f"{value:+.1f}" if value else "0", textAnchor="middle", fontName="Helvetica", fontSize=6.5, fillColor=MUTED))
    return d


def story() -> list:
    m = FULL["metrics"]
    fm = FREB["strict_parser_metrics"]
    cm = CLEAN["treatment_metrics"]
    cb = CLEAN["control_metrics"]
    delta = CLEAN["paired_delta_treatment_minus_control"]
    ci = CLEAN["cluster_bootstrap"]["metrics"]["mmad_overall_macro"]["delta_ci95"]
    items = []

    items += [
        Spacer(1, 13 * mm),
        p("[Research]", ParagraphStyle("Tag", parent=CALLOUT, fontSize=11, textColor=CYAN, spaceAfter=5)),
        p("ECHO-CAVER: Evidence-Tangent Residual Readers for FREB-CAVER", TITLE),
        p("A private paper archive for model-native industrial anomaly reasoning", SUBTITLE),
        Spacer(1, 4 * mm),
        Table(
            [[p("FULL MMAD-COMPATIBLE OVERALL", SMALL), p("CLEAN PAIRED SIGNAL", SMALL), p("TRAINABLE FRACTION", SMALL)],
             [p(f"<b>{100*m['mmad_overall_macro']:.6f}%</b>", CALLOUT), p(f"<b>+{100*delta['mmad_overall_macro']:.4f} pt</b>", CALLOUT), p("<b>0.02853%</b>", CALLOUT)]],
            colWidths=[56 * mm, 56 * mm, 56 * mm],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), PALE),
                ("BOX", (0, 0), (-1, -1), 0.8, GRID),
                ("INNERGRID", (0, 0), (-1, -1), 0.6, GRID),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]),
        ),
        Spacer(1, 7 * mm),
        p("Abstract", H1),
        p(
            "FREB-CAVER inserts query-reference comparison evidence into a frozen JUDO multimodal reasoner, but the frozen downstream decoder may not optimally read the new state. ECHO-CAVER installs rank-16 residual readers immediately after five FREB writers and constrains their output directions to each writer's rank-32 left-singular span. Gradient and post-optimizer projection preserve the constraint; zero initialization gives exact FREB identity. The full 39,670-question census reaches <b>82.416847%</b>. A 5,600-row component-disjoint panel shows a positive but statistically inconclusive +0.2698-point signal. This archive therefore retains ECHO as a full-benchmark candidate without claiming confirmed clean generalization."
        ),
        Spacer(1, 3 * mm),
        p("JaeWoo Choi - 24 August 2026 - Version 0.1.0", SMALL),
        PageBreak(),
    ]

    items += [
        p("1. Motivation and contribution", H1),
        p("Industrial anomaly QA requires local visual comparison and domain-aware language reasoning. JUDO juxtaposes a query image with normal context; FREB-CAVER adds a causal visual-evidence carrier to selected decoder layers. ECHO addresses the remaining writer-reader mismatch: a frozen downstream decoder may not convert the new hidden-state evidence into the correct option without disrupting behavior it already performs well."),
        p("Contribution", H2),
        p("ECHO derives both placement and trainable geometry from the deployed FREB carrier. Readers sit immediately after the five evidence writers. Their LoRA-B outputs are restricted to the writers' dominant singular span, so adaptation can reinterpret established visual evidence without opening an unrestricted residual channel."),
        architecture(),
        p("Figure 1. Frozen components are blue/amber; trainable evidence-tangent readers are cyan.", CAPTION),
        p("Novelty boundary", H2),
        p("Ordinary LoRA can learn any low-rank output direction. O-LoRA-style continual learning separates tasks with orthogonal subspaces. ECHO makes the opposite geometric choice: it trains inside the existing evidence span, because the goal is to read a frozen multimodal carrier rather than isolate a new task."),
        p("The method is a single end-to-end runtime. It does not use an external router, score threshold, ensemble, crop policy, or post-hoc answer correction.", CALLOUT),
        PageBreak(),
    ]

    items += [
        p("2. Method", H1),
        p("Evidence projection", H2),
        p("For frozen FREB writer W_i = U_i Sigma_i V_i^T, retain the first 32 columns of U_i and form P_i = U_i U_i^T. At the corresponding downstream layer, ECHO applies rank-16 LoRA: y = W_0 x + (alpha/r) B_i A_i x with alpha=32. During backpropagation grad(B_i) is replaced by P_i grad(B_i); after every AdamW step, B_i is replaced by P_i B_i. The second projection prevents optimizer state from accumulating off-span weight components."),
        p("Placement and parameters", H2),
        metric_table([
            ["Property", "Frozen writer", "ECHO reader"],
            ["Decoder layers", "18, 20, 22, 24, 26", "19, 21, 23, 25, 27"],
            ["Modules", "FREB belief_up", "o_proj + down_proj"],
            ["Rank", "writer width 128", "LoRA rank 16"],
            ["Geometric span", "source of U32", "B constrained to U32"],
        ], widths=[43 * mm, 61 * mm, 66 * mm], align_right_from=None),
        Spacer(1, 3 * mm),
        p("Training objective", H2),
        p("Each example is paired with a semantic option permutation. The loss combines answer cross-entropy, 0.25-weight semantic option-orbit equivariance, 1.0-weight KL retention to frozen FREB, and 0.75-weight wrong-answer repair with margin 0.25. The repair term acts when frozen FREB ranks an incorrect option over the gold answer."),
        metric_table([
            ["Training contract", "Value"],
            ["Train / validation pairs", "963 / 184 (non-AD)"],
            ["Steps / batch", "100 / 2 pairs"],
            ["Optimizer", "AdamW, lr 2e-5, weight decay 0"],
            ["Trainable parameters", "2,375,680 of 8,327,619,987 (0.02853%)"],
            ["Training time", "183.07 s on one NVIDIA H200"],
            ["Peak allocated memory", "23.84 GiB"],
        ], widths=[71 * mm, 99 * mm], align_right_from=None),
        Spacer(1, 3 * mm),
        p("Mechanics verification", H2),
        p("The archived receipt records zero-init exact identity (maximum absolute difference 0.0), nonzero first-step gradients at all ten LoRA-B targets, and rank-32 projection for every trainable output matrix."),
        PageBreak(),
    ]

    task_order = ["Anomaly Detection", "Defect Analysis", "Defect Classification", "Defect Description", "Defect Localization", "Object Analysis", "Object Classification"]
    full_rows = [["Task", "Archived FREB", "ECHO", "Delta"]]
    for task in task_order:
        fv = 100 * fm["per_task_source_macro_accuracy"][task]
        ev = 100 * m["per_task_macro"][task]
        full_rows.append([task, f"{fv:.4f}", f"{ev:.4f}", f"{ev-fv:+.4f}"])
    full_rows.append(["Overall", f"{100*fm['normalized_28_cell_macro_accuracy']:.4f}", f"{100*m['mmad_overall_macro']:.4f}", f"{100*(m['mmad_overall_macro']-fm['normalized_28_cell_macro_accuracy']):+.4f}"])

    items += [
        p("3. Evaluation and full-census result", H1),
        p("Strict score contract", H2),
        p("Overall is the unweighted mean of 28 source-by-task cells. Each Anomaly Detection cell uses balanced accuracy over normal and anomaly labels; all other cells use accuracy. A completion must contain exactly one case-sensitive answer tag with an option label. Invalid completions are incorrect."),
        p("Coverage", H2),
        p("Eight independently inferred shards cover exactly 39,670 frozen IDs with no duplicate or missing row. Every result uses the same adapter identity and normalized inference contract. Deterministic re-analysis reproduced the same SHA-256."),
        metric_table(full_rows, widths=[74 * mm, 32 * mm, 32 * mm, 30 * mm]),
        Spacer(1, 3 * mm),
        delta_chart(),
        p("Figure 2. Full-census task deltas. These are descriptive because the archived FREB row-level predictions are unavailable.", CAPTION),
        PageBreak(),
    ]

    items += [
        p("4. Clean paired evidence", H1),
        p("The clean panel contains 200 rows in each of 28 source-by-task cells. It excludes every query-reference connected component overlapping CAPRI/PACE training assets and records a completed near-duplicate component audit. FREB and ECHO are evaluated on identical rows."),
        metric_table([
            ["Metric", "FREB", "ECHO", "Delta"],
            ["Overall", f"{100*cb['mmad_overall_macro']:.4f}", f"{100*cm['mmad_overall_macro']:.4f}", f"{100*delta['mmad_overall_macro']:+.4f}"],
            ["non-AD macro", f"{100*cb['non_ad_six_task_macro']:.4f}", f"{100*cm['non_ad_six_task_macro']:.4f}", f"{100*delta['non_ad_six_task_macro']:+.4f}"],
            ["AD balanced", f"{100*cb['ad_balanced_accuracy']:.4f}", f"{100*cm['ad_balanced_accuracy']:.4f}", f"{100*delta['ad_balanced_accuracy']:+.4f}"],
            ["ITT correct", "4,556", "4,572", "+16"],
        ], widths=[64 * mm, 35 * mm, 35 * mm, 34 * mm]),
        Spacer(1, 5 * mm),
        p("Paired uncertainty", H2),
        metric_table([
            ["Statistic", "Result"],
            ["ECHO-only / FREB-only correct", "58 / 42"],
            ["Exact McNemar p", "0.1332106"],
            ["Component bootstrap", "50,000 replicates, 2,280 clusters"],
            ["Overall delta 95% CI", f"[{100*ci[0]:+.4f}, {100*ci[1]:+.4f}] points"],
            ["P(delta > 0)", "0.92716"],
        ], widths=[72 * mm, 96 * mm], align_right_from=None),
        Spacer(1, 5 * mm),
        Table([[p("Interpretation", CALLOUT), p("The clean point estimate is directionally positive, but the interval includes zero and McNemar is not significant. The preregistered absolute Overall > 82% clean-panel gate is not met.", BODY)]], colWidths=[35 * mm, 133 * mm], style=TableStyle([("BACKGROUND", (0, 0), (-1, -1), CYAN_LIGHT), ("BOX", (0, 0), (-1, -1), 0.8, CYAN), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7), ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)])),
        PageBreak(),
    ]

    items += [
        p("5. Failure analysis and claim boundaries", H1),
        p("Reasoning drift", H2),
        p("ECHO changes the generated thinking trace on 1,349/5,600 clean-panel examples. Among discordant decisions, 35/58 corrections and 30/42 regressions also change the trace. Evidence-span restriction therefore does not guarantee that adaptation is localized to the final answer state."),
        p("Format and task trade-offs", H2),
        p("The full run contains 45 strict-format-invalid generations, versus nine in the archived FREB result. Full gains are concentrated in Defect Classification (+1.5331), Defect Localization (+1.2348), and Defect Description (+1.0712) points. Defect Analysis, Object Analysis, and Object Classification regress slightly. The clean-panel AD balanced score also decreases by 0.0662 points."),
        p("Supported claims", H2),
        metric_table([
            ["Supported", "Not supported"],
            ["82.416847% Full-MMAD-compatible census", "confirmed independent generalization gain"],
            ["single-runtime evidence-tangent LoRA", "state-of-the-art under a harmonized protocol"],
            ["positive clean-panel point estimate", "material solution to anomaly detection"],
            ["exact checkpoint and code identities", "paired significance for the archived full FREB delta"],
        ], widths=[84 * mm, 84 * mm], align_right_from=None),
        Spacer(1, 5 * mm),
        p("Prospective mechanism", H2),
        p("A successor should share FREB's causal answer-prefix decision mask with the ECHO readers. Segmentation and reasoning tokens would then remain exactly frozen FREB computation, while evidence-tangent adaptation operates only where the final option is formed. This proposal is future work, not part of the ECHO v1 result."),
        PageBreak(),
    ]

    items += [
        p("6. Reproducibility record", H1),
        p("The private repository includes the exact training and inference programs, strict sharded and paired analyzers, the 9.1MB adapter, training mechanics receipt, clean and full analysis JSON, launch recipes, and cross-pod ledger. MMAD assets, row-level questions, raw generations, JUDO weights, and FREB weights are not redistributed."),
        metric_table([
            ["Artifact", "SHA-256"],
            ["ECHO weights", "21623bfafc3d329128e2753a5600a9197088ef527deeb44687ea0eacd89f4bb9"],
            ["Exact inference runner", "f160385a1004749485f2dff060e121ac1bf280770d92914171d7613289b8d6a9"],
            ["FREB weights", "ac3233dbee2de9dd1aaea4acede275ddc2d7a322427795717d9232291e03b0ed"],
            ["Full manifest", "61a266302585230610ad58876e76e073ed04f7dccb31c9c61daeaf44a81d4693"],
            ["Full analysis", "68195e2e0fe5e49108b0a48b8b602a1b6e1617611bad2c0b11330ebb6eb8e547"],
        ], widths=[45 * mm, 123 * mm], align_right_from=None),
        Spacer(1, 5 * mm),
        p("Compute record", H2),
        p("Full inference used at most four NVIDIA H200 GPUs over two pods and eight independent shards. The measured inference window was 3 h 14 min 59 s. The conservative two-pod lifetime meter was USD 71.84 and includes earlier ECHO experiments on the primary pod."),
        p("Archive verification", H2),
        p("Run <font name='Courier'>python3 scripts/verify_archive.py</font> to re-hash the checkpoint and runner, recompute Overall from task values, validate the exact shard cover, and cross-check the clean-panel paired statistics. The verifier requires no GPU, dataset, or network access."),
        p("Data and safety", H2),
        p("This is research software, not a production inspection system. Access to this private archive does not grant rights to JUDO, Qwen2.5-VL, FREB-CAVER, MMAD, or constituent datasets. Users must obtain and review every upstream artifact separately."),
        PageBreak(),
    ]

    items += [
        p("7. Conclusion", H1),
        p("ECHO-CAVER is a compact writer-reader alignment mechanism for an existing multimodal evidence carrier. Its distinctive choice is evidence-tangent adaptation: the downstream reader is trainable, but its output directions remain inside the dominant span of the frozen visual writer. Full-benchmark performance motivates retention of the method; clean paired evidence remains inconclusive. The archive records both facts explicitly."),
        p("References", H1),
        p("[1] H. Kang, W. Lee, J. Kim, and H. Park. JUDO: A Juxtaposed Domain-Oriented Multimodal Reasoner for Industrial Anomaly QA. ICLR 2026; arXiv:2605.20284."),
        p("[2] X. Jiang et al. MMAD: The First-Ever Comprehensive Benchmark for Multimodal Large Language Models in Industrial Anomaly Detection. arXiv:2410.09453, 2024."),
        p("[3] E. J. Hu et al. LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022; arXiv:2106.09685."),
        p("[4] X. Wang et al. Orthogonal Subspace Learning for Language Model Continual Learning. arXiv:2310.14152, 2023."),
        p("[5] FREB-CAVER research contributors. FREB-CAVER: Model-native causal visual-evidence binding for industrial anomaly reasoning. Software artifact v0.1.0, 2026."),
        Spacer(1, 10 * mm),
        Table([[p("Archive status", CALLOUT), p("ECHO-CAVER v1 is retained as a full-benchmark research candidate. Further asset-disjoint replication is required before a generalization claim.", BODY)]], colWidths=[40 * mm, 128 * mm], style=TableStyle([("BACKGROUND", (0, 0), (-1, -1), PALE), ("BOX", (0, 0), (-1, -1), 0.8, GRID), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7), ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)])),
    ]
    return items


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    PaperDoc(OUTPUT).build(story())
    print(OUTPUT)


if __name__ == "__main__":
    main()

