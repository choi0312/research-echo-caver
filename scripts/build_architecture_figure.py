#!/usr/bin/env python3
"""Render the publication-style ECHO-CAVER architecture figure as a PNG."""

from __future__ import annotations

from pathlib import Path
from textwrap import wrap

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets/echo_caver_architecture.png"
W, H = 2400, 1350

BG = "#F7F9FC"
WHITE = "#FFFFFF"
NAVY = "#13233A"
MUTED = "#64748B"
GRID = "#D8E1EC"
BLUE = "#2563EB"
BLUE_PALE = "#EAF1FF"
AMBER = "#D97706"
AMBER_PALE = "#FFF5E6"
CYAN = "#0891B2"
CYAN_PALE = "#E6FAFD"
VIOLET = "#7C3AED"
VIOLET_PALE = "#F1EBFF"
GREEN = "#15803D"
GREEN_PALE = "#EAF8EF"
RED = "#DC2626"

FONT_REGULAR = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REGULAR, size)


def rounded(draw: ImageDraw.ImageDraw, xy, radius=28, fill=WHITE, outline=GRID, width=3):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def center_text(draw, xy, text, fnt, fill=NAVY):
    box = draw.textbbox((0, 0), text, font=fnt)
    x = xy[0] - (box[2] - box[0]) / 2
    y = xy[1] - (box[3] - box[1]) / 2
    draw.text((x, y), text, font=fnt, fill=fill)


def multiline_center(draw, center, text, fnt, fill=NAVY, spacing=7):
    lines = text.split("\n")
    heights = [draw.textbbox((0, 0), line, font=fnt)[3] for line in lines]
    total = sum(heights) + spacing * (len(lines) - 1)
    y = center[1] - total / 2
    for line, h in zip(lines, heights):
        center_text(draw, (center[0], y + h / 2), line, fnt, fill)
        y += h + spacing


def arrow(draw, start, end, color="#7890AE", width=8, head=22):
    draw.line((start, end), fill=color, width=width)
    x2, y2 = end
    x1, y1 = start
    dx, dy = x2 - x1, y2 - y1
    length = max((dx * dx + dy * dy) ** 0.5, 1)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    base_x, base_y = x2 - ux * head, y2 - uy * head
    points = [
        (x2, y2),
        (base_x + px * head * 0.55, base_y + py * head * 0.55),
        (base_x - px * head * 0.55, base_y - py * head * 0.55),
    ]
    draw.polygon(points, fill=color)


def pill(draw, xy, text, fill, text_fill, outline=None, fnt=None):
    fnt = fnt or font(24, True)
    rounded(draw, xy, radius=18, fill=fill, outline=outline or fill, width=2)
    center_text(draw, ((xy[0] + xy[2]) / 2, (xy[1] + xy[3]) / 2 - 1), text, fnt, text_fill)


def small_image_card(draw, xy, title, abnormal=False):
    x1, y1, x2, y2 = xy
    rounded(draw, xy, radius=18, fill=WHITE, outline=GRID, width=2)
    draw.text((x1 + 16, y1 + 12), title, font=font(20, True), fill=NAVY)
    gx1, gy1, gx2, gy2 = x1 + 16, y1 + 46, x2 - 16, y2 - 16
    rounded(draw, (gx1, gy1, gx2, gy2), radius=12, fill="#E8EEF5", outline="#CBD5E1", width=2)
    step = 26
    for x in range(int(gx1 + 9), int(gx2), step):
        draw.line((x, gy1 + 8, x, gy2 - 8), fill="#CBD5E1", width=2)
    for y in range(int(gy1 + 9), int(gy2), step):
        draw.line((gx1 + 8, y, gx2 - 8, y), fill="#CBD5E1", width=2)
    if abnormal:
        cx, cy = gx1 + 0.68 * (gx2 - gx1), gy1 + 0.55 * (gy2 - gy1)
        draw.ellipse((cx - 22, cy - 18, cx + 22, cy + 18), fill="#FCA5A5", outline=RED, width=4)


def build() -> None:
    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)

    draw.text((76, 54), "ECHO-CAVER", font=font(52, True), fill=NAVY)
    draw.text((475, 65), "Evidence-Tangent Writer-Reader Alignment", font=font(36, True), fill=CYAN)
    draw.text(
        (78, 121),
        "A single end-to-end JUDO runtime that learns to read frozen comparison evidence",
        font=font(25),
        fill=MUTED,
    )

    # Main inference pathway.
    cards = [
        (64, 218, 414, 552, BLUE_PALE, BLUE, "01", "Juxtaposed input"),
        (500, 218, 882, 552, BLUE_PALE, BLUE, "02", "Frozen JUDO"),
        (968, 190, 1356, 580, AMBER_PALE, AMBER, "03", "FREB evidence writers"),
        (1450, 190, 1968, 580, VIOLET_PALE, VIOLET, "04", "ECHO residual readers"),
        (2058, 218, 2334, 552, GREEN_PALE, GREEN, "05", "Structured output"),
    ]
    for x1, y1, x2, y2, fill, edge, number, title in cards:
        rounded(draw, (x1, y1, x2, y2), radius=30, fill=fill, outline=edge, width=4)
        pill(draw, (x1 + 20, y1 + 20, x1 + 74, y1 + 62), number, edge, WHITE, fnt=font(21, True))
        title_font = font(22 if number == "05" else 27, True)
        draw.text((x1 + 82 if number == "05" else x1 + 88, y1 + 25), title, font=title_font, fill=NAVY)

    small_image_card(draw, (86, 300, 238, 514), "Query", abnormal=True)
    small_image_card(draw, (250, 300, 392, 514), "Normal ref.", abnormal=False)

    pill(draw, (532, 306, 850, 366), "Vision encoder", WHITE, BLUE, outline="#AFC7FF")
    arrow(draw, (691, 374), (691, 403), color=BLUE, width=6, head=18)
    pill(draw, (532, 411, 850, 471), "Qwen2.5-VL decoder", WHITE, BLUE, outline="#AFC7FF")
    draw.text((556, 493), "all backbone weights frozen", font=font(21), fill=MUTED)

    draw.text((1002, 281), "Write comparison belief", font=font(23, True), fill=AMBER)
    layer_y = [330, 378, 426, 474, 522]
    for idx, y in enumerate(layer_y):
        pill(draw, (1004, y, 1320, y + 38), f"belief_up  ·  L{18 + idx * 2}", WHITE, AMBER, outline="#F3C889", fnt=font(19, True))

    draw.text((1484, 281), "Read evidence one block later", font=font(23, True), fill=VIOLET)
    for idx, y in enumerate(layer_y):
        pill(draw, (1484, y, 1930, y + 38), f"LoRA(o_proj, down_proj)  ·  L{19 + idx * 2}", WHITE, VIOLET, outline="#CFBDFB", fnt=font(18, True))
        arrow(draw, (1322, y + 19), (1473, y + 19), color=CYAN, width=4, head=14)

    multiline_center(draw, (2196, 350), "<seg>\n<think>\n<answer>X</answer>", font(24, True), GREEN, spacing=18)
    draw.text((2092, 491), "one checkpoint", font=font(21), fill=MUTED)

    for start, end in [((414, 385), (488, 385)), ((882, 385), (956, 385)), ((1968, 385), (2046, 385))]:
        arrow(draw, start, end)

    # Lower mechanism panel.
    rounded(draw, (64, 645, 2334, 1250), radius=34, fill=WHITE, outline=GRID, width=3)
    draw.text((96, 682), "Mechanism", font=font(31, True), fill=NAVY)
    pill(draw, (265, 678, 482, 724), "TRAINABLE ONLY", CYAN_PALE, CYAN, outline=CYAN, fnt=font(18, True))

    rounded(draw, (96, 754, 1410, 1206), radius=24, fill="#FBFCFE", outline="#DCE6F1", width=2)
    draw.text((128, 786), "A. Evidence-tangent geometry", font=font(27, True), fill=NAVY)
    draw.text((128, 832), "The reader may adapt only inside the dominant output span of its paired writer.", font=font(22), fill=MUTED)

    geo_boxes = [
        (130, 918, 356, 1078, "Frozen writer", "W_i", AMBER_PALE, AMBER),
        (425, 918, 690, 1078, "Truncated SVD", "U_i,32", BLUE_PALE, BLUE),
        (760, 918, 1022, 1078, "Projector", "P_i = U U^T", CYAN_PALE, CYAN),
        (1092, 918, 1374, 1078, "LoRA output", "B_i <- P_i B_i", VIOLET_PALE, VIOLET),
    ]
    for x1, y1, x2, y2, label, equation, fill, edge in geo_boxes:
        rounded(draw, (x1, y1, x2, y2), radius=20, fill=fill, outline=edge, width=3)
        center_text(draw, ((x1 + x2) / 2, y1 + 49), label, font(20, True), NAVY)
        center_text(draw, ((x1 + x2) / 2, y1 + 107), equation, font(25, True), edge)
    for start, end in [((356, 998), (413, 998)), ((690, 998), (748, 998)), ((1022, 998), (1080, 998))]:
        arrow(draw, start, end, width=6, head=18)
    draw.text((130, 1124), "Projection is applied to gradients and again after every AdamW step.", font=font(21, True), fill=CYAN)

    rounded(draw, (1450, 754, 2302, 1206), radius=24, fill="#FBFCFE", outline="#DCE6F1", width=2)
    draw.text((1482, 786), "B. Behavior-preserving training", font=font(27, True), fill=NAVY)
    objectives = [
        ("CE", "answer supervision", BLUE_PALE, BLUE),
        ("EQ", "option-orbit equivariance", CYAN_PALE, CYAN),
        ("KL", "frozen-FREB retention", AMBER_PALE, AMBER),
        ("R", "wrong-answer repair", VIOLET_PALE, VIOLET),
    ]
    y = 860
    for tag, label, fill, edge in objectives:
        pill(draw, (1484, y, 1550, y + 48), tag, fill, edge, outline=edge, fnt=font(19, True))
        draw.text((1574, y + 10), label, font=font(22, True), fill=NAVY)
        y += 65
    rounded(draw, (1484, 1124, 2268, 1178), radius=16, fill=GREEN_PALE, outline="#92D4A8", width=2)
    center_text(draw, (1876, 1151), "Zero-init: ECHO step 0 == frozen FREB", font(21, True), GREEN)

    # Footer legend.
    legend_y = 1291
    legend = [(BLUE, "frozen backbone"), (AMBER, "frozen evidence writer"), (VIOLET, "trainable reader"), (GREEN, "output")]
    x = 78
    for color, label in legend:
        draw.rounded_rectangle((x, legend_y - 11, x + 24, legend_y + 13), radius=6, fill=color)
        draw.text((x + 36, legend_y - 13), label, font=font(19), fill=MUTED)
        x += 300
    draw.text((1745, legend_y - 13), "No router · no ensemble · no post-hoc correction", font=font(20, True), fill=NAVY)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT, "PNG", optimize=True, dpi=(300, 300))
    print(OUTPUT)


if __name__ == "__main__":
    build()
