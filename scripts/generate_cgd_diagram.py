#!/usr/bin/env python3
"""
Generate a publication-quality CGD pipeline diagram using matplotlib.
Consistency-Guided Decoding (CGD) Pipeline for ICLR conference paper.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path
import numpy as np

# ─── Color Palette ─────────────────────────────────────────────────────
BLUE       = '#4A90D9'
BLUE_LIGHT = '#D6E8F7'
ORANGE     = '#E67E22'
ORANGE_LT  = '#FDEBD0'
RED        = '#E74C3C'
GREEN      = '#27AE60'
GREEN_LT   = '#D5F5E3'
YELLOW     = '#D4AC0D'
YELLOW_LT  = '#FEF9E7'
GRAY       = '#95A5A6'
GRAY_LIGHT = '#ECF0F1'
DARK       = '#2C3E50'
WHITE      = '#FFFFFF'
PREMISE_BG = '#EBF5FB'

# ─── Figure Setup ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(15, 6.0))
ax.set_xlim(-0.5, 15.5)
ax.set_ylim(-1.5, 6.2)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor(WHITE)
ax.set_facecolor(WHITE)

# ─── Helper Functions ──────────────────────────────────────────────────

def draw_rounded_box(ax, x, y, w, h, facecolor, edgecolor, label_lines,
                     fontsize=9, linewidth=1.5, radius=0.15, zorder=3,
                     bold_first=True):
    """Draw a rounded rectangle with centered multi-line text."""
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=facecolor, edgecolor=edgecolor,
        linewidth=linewidth, zorder=zorder
    )
    ax.add_patch(box)

    # Calculate line spacing in data coordinates
    # The figure is 15 units wide displayed in ~15 inches, so ~1 unit = 1 inch
    # At 72 points/inch, fontsize 10 ≈ 10/72 inches ≈ 0.14 data units
    spacing = fontsize / 72.0 * 1.8  # empirical multiplier for data coords
    n = len(label_lines)
    total = (n - 1) * spacing
    for i, line in enumerate(label_lines):
        yy = y + total / 2 - i * spacing
        weight = 'bold' if (i == 0 and bold_first) else 'normal'
        fs = fontsize if i == 0 else fontsize - 1.0
        ax.text(x, yy, line, ha='center', va='center',
                fontsize=fs, fontweight=weight, color=DARK,
                fontfamily='sans-serif', zorder=zorder + 1)
    return box


def draw_diamond(ax, cx, cy, w, h, facecolor, edgecolor, label,
                 fontsize=8.5, linewidth=1.8, zorder=3):
    """Draw a decision diamond."""
    verts = [
        (cx, cy + h/2),
        (cx + w/2, cy),
        (cx, cy - h/2),
        (cx - w/2, cy),
        (cx, cy + h/2),
    ]
    codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
    path = Path(verts, codes)
    patch = mpatches.PathPatch(path, facecolor=facecolor, edgecolor=edgecolor,
                                linewidth=linewidth, zorder=zorder)
    ax.add_patch(patch)
    ax.text(cx, cy, label, ha='center', va='center', fontsize=fontsize,
            fontweight='bold', color=DARK, fontfamily='sans-serif',
            zorder=zorder + 1, linespacing=1.2)


def draw_cylinder(ax, cx, cy, w, h, facecolor, edgecolor, label_lines,
                  fontsize=8.5, linewidth=1.3, zorder=3):
    """Draw a cylinder (database) icon."""
    ell_h = h * 0.15
    rect = mpatches.Rectangle((cx - w/2, cy - h/2), w, h,
                               facecolor=facecolor, edgecolor=edgecolor,
                               linewidth=linewidth, zorder=zorder)
    ax.add_patch(rect)
    top_ell = mpatches.Ellipse((cx, cy + h/2), w, ell_h * 2,
                                facecolor=facecolor, edgecolor=edgecolor,
                                linewidth=linewidth, zorder=zorder + 1)
    ax.add_patch(top_ell)
    bot_ell = mpatches.Arc((cx, cy - h/2), w, ell_h * 2, angle=0,
                            theta1=180, theta2=360,
                            edgecolor=edgecolor, linewidth=linewidth, zorder=zorder + 1)
    ax.add_patch(bot_ell)
    cover = mpatches.Rectangle((cx - w/2 + 0.01, cy + h/2 - ell_h * 0.8),
                                w - 0.02, ell_h * 0.8,
                                facecolor=facecolor, edgecolor='none', zorder=zorder + 0.5)
    ax.add_patch(cover)

    spacing = fontsize / 72.0 * 1.8
    n = len(label_lines)
    total = (n - 1) * spacing
    for i, line in enumerate(label_lines):
        yy = cy + total / 2 - i * spacing
        weight = 'bold' if i == 0 else 'normal'
        fs = fontsize if i == 0 else fontsize - 1
        ax.text(cx, yy, line, ha='center', va='center',
                fontsize=fs, fontweight=weight, color=DARK,
                fontfamily='sans-serif', zorder=zorder + 2)


def draw_arrow(ax, x1, y1, x2, y2, color=DARK, lw=1.3,
               connectionstyle='arc3,rad=0', shrinkA=4, shrinkB=4, zorder=2):
    """Draw an arrow."""
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='-|>', color=color, linewidth=lw,
        connectionstyle=connectionstyle,
        shrinkA=shrinkA, shrinkB=shrinkB,
        zorder=zorder, mutation_scale=14
    )
    ax.add_patch(arr)
    return arr


def draw_badge(ax, cx, cy, num, color):
    """Draw a numbered step badge."""
    badge = mpatches.Circle((cx, cy), 0.22, facecolor=color, edgecolor=WHITE,
                             linewidth=1.5, zorder=6)
    ax.add_patch(badge)
    ax.text(cx, cy, str(num), ha='center', va='center', fontsize=8.5,
            fontweight='bold', color=WHITE, fontfamily='sans-serif', zorder=7)


# ─── Layout Coordinates ───────────────────────────────────────────────
y_main    = 2.8     # main pipeline row
y_premise = 5.2     # premise box
y_input   = 3.9     # question set
y_hist    = 0.3     # history cylinder
y_repair  = 0.3     # repair box

x_input   = 1.3
x_gen     = 4.6
x_check   = 7.5
x_diamond = 10.2
x_output  = 13.3
x_repair  = 10.2
x_hist    = 7.2

# ─── Title ─────────────────────────────────────────────────────────────
ax.text(7.5, 5.95, 'Consistency-Guided Decoding (CGD) Pipeline',
        ha='center', va='center', fontsize=14, fontweight='bold',
        color=DARK, fontfamily='sans-serif', zorder=10)

# ─── 1. Shared Premise Box ────────────────────────────────────────────
draw_rounded_box(ax, x_input, y_premise, 2.6, 0.55, PREMISE_BG, BLUE,
                 ['Shared Premise P'], fontsize=10, radius=0.12)

draw_arrow(ax, x_input, y_premise - 0.30, x_input, y_input + 0.38,
           color=BLUE, lw=1.5, shrinkA=2, shrinkB=2)

# ─── 2. Question Set Input ────────────────────────────────────────────
draw_rounded_box(ax, x_input, y_input, 2.3, 0.65, BLUE_LIGHT, BLUE,
                 ['Question Set', 'Q = {q\u2081, q\u2082, ..., q\u2099}'],
                 fontsize=10, radius=0.12)

# Arrow from input down then right to Generate
# Vertical segment
ax.plot([x_input + 1.15, x_input + 1.15], [y_input - 0.35, y_main],
        color=DARK, lw=1.3, solid_capstyle='round', zorder=2)
# Horizontal to Generate
draw_arrow(ax, x_input + 1.15, y_main, x_gen - 1.05, y_main,
           color=DARK, lw=1.3, shrinkA=0, shrinkB=4)

# ─── Dashed loop boundary ─────────────────────────────────────────────
loop_left  = 3.1
loop_right = 11.7
loop_top   = y_main + 1.1
loop_bot   = y_repair - 0.75
loop_rect = mpatches.FancyBboxPatch(
    (loop_left, loop_bot), loop_right - loop_left, loop_top - loop_bot,
    boxstyle="round,pad=0,rounding_size=0.25",
    facecolor='none', edgecolor=GRAY, linewidth=1.2,
    linestyle=(0, (6, 4)), zorder=1
)
ax.add_patch(loop_rect)
ax.text(loop_left + 0.2, loop_top - 0.08, 'for each q\u1D62',
        ha='left', va='top', fontsize=9, color=GRAY,
        fontfamily='sans-serif', fontstyle='italic', fontweight='bold', zorder=5)

# ─── 3. Generate Box (BLUE) ───────────────────────────────────────────
bw, bh = 2.1, 0.9
draw_rounded_box(ax, x_gen, y_main, bw, bh, BLUE_LIGHT, BLUE,
                 ['Generate', 'LLM  M'], fontsize=10, radius=0.13)
draw_badge(ax, x_gen - bw/2 + 0.22, y_main + bh/2 - 0.22, 1, BLUE)

# Arrow: Generate → Check
draw_arrow(ax, x_gen + bw/2 + 0.02, y_main, x_check - bw/2 - 0.02, y_main,
           color=DARK, lw=1.5)
ax.text((x_gen + x_check) / 2, y_main + 0.15, 'a\u1D62',
        ha='center', va='bottom', fontsize=9, color=DARK,
        fontfamily='sans-serif', fontstyle='italic', zorder=10)

# ─── 4. Check Box (ORANGE) ────────────────────────────────────────────
draw_rounded_box(ax, x_check, y_main, bw, bh, ORANGE_LT, ORANGE,
                 ['Check', 'NLI Checker  C'], fontsize=10, radius=0.13)
draw_badge(ax, x_check - bw/2 + 0.22, y_main + bh/2 - 0.22, 2, ORANGE)

# Arrow: Check → Diamond
draw_arrow(ax, x_check + bw/2 + 0.02, y_main, x_diamond - 0.68, y_main,
           color=DARK, lw=1.5)

# ─── 5. Answer History Cylinder ───────────────────────────────────────
cyl_w, cyl_h = 2.0, 0.85
draw_cylinder(ax, x_hist, y_hist, cyl_w, cyl_h, GRAY_LIGHT, GRAY,
              ['Answer History', '{(q\u2081,a\u2081), (q\u2082,a\u2082), ...}'],
              fontsize=8.5, linewidth=1.3)

# Read arrow: History → Check
draw_arrow(ax, x_hist - 0.25, y_hist + cyl_h/2 + 0.08, x_check - 0.25, y_main - bh/2 - 0.02,
           color=GRAY, lw=1.2)
ax.text(x_hist - 0.7, (y_hist + y_main) / 2, 'read',
        ha='center', va='center', fontsize=7.5, color=GRAY,
        fontfamily='sans-serif', fontstyle='italic', zorder=10)

# Store arrow: (accepted answers go to history)
draw_arrow(ax, x_hist + 0.25, y_main - bh/2 - 0.02, x_hist + 0.25, y_hist + cyl_h/2 + 0.08,
           color=GREEN, lw=1.2)
ax.text(x_hist + 0.7, (y_hist + y_main) / 2, 'store',
        ha='center', va='center', fontsize=7.5, color=GREEN,
        fontfamily='sans-serif', fontstyle='italic', zorder=10)

# ─── 6. Contradiction Decision Diamond ────────────────────────────────
dw, dh = 1.3, 1.3
draw_diamond(ax, x_diamond, y_main, dw, dh, YELLOW_LT, YELLOW,
             'Contra-\ndiction?', fontsize=8.5, linewidth=1.8)

# ─── 7. "No" path → Output ────────────────────────────────────────────
draw_arrow(ax, x_diamond + dw/2 + 0.02, y_main, x_output - 1.18, y_main,
           color=GREEN, lw=2.2)
ax.text(x_diamond + dw/2 + 0.25, y_main + 0.18, 'No',
        ha='center', va='bottom', fontsize=10, fontweight='bold',
        color=GREEN, fontfamily='sans-serif', zorder=10)

# ─── 8. "Yes" path → Repair ───────────────────────────────────────────
draw_arrow(ax, x_diamond, y_main - dh/2 - 0.02, x_diamond, y_repair + bh/2 * 0.9 + 0.05,
           color=RED, lw=2.2)
ax.text(x_diamond + 0.22, y_main - dh/2 - 0.25, 'Yes',
        ha='left', va='center', fontsize=10, fontweight='bold',
        color=RED, fontfamily='sans-serif', zorder=10)

# ─── 9. Repair Box (GREEN) ────────────────────────────────────────────
rw, rh = 2.3, 0.9
draw_rounded_box(ax, x_repair, y_repair, rw, rh, GREEN_LT, GREEN,
                 ['Repair', 'Revision Prompt \u2192 M'],
                 fontsize=9.5, radius=0.13)
draw_badge(ax, x_repair - rw/2 + 0.22, y_repair + rh/2 - 0.22, 3, GREEN)

# Re-check loop: Repair → Check (curved arrow)
repair_arrow = FancyArrowPatch(
    (x_repair - rw/2 - 0.02, y_repair + 0.1),
    (x_check - 0.5, y_main - bh/2 - 0.02),
    arrowstyle='-|>', color=GREEN, linewidth=1.5,
    connectionstyle='arc3,rad=-0.35',
    shrinkA=4, shrinkB=4, zorder=2, mutation_scale=14
)
ax.add_patch(repair_arrow)
ax.text(x_check - 1.75, y_repair + 0.65, 're-check',
        ha='center', va='center', fontsize=7.5, color=GREEN,
        fontfamily='sans-serif', fontstyle='italic', zorder=10)

# ─── 10. Output Box (GREEN) ───────────────────────────────────────────
ow, oh = 2.3, 0.9
draw_rounded_box(ax, x_output, y_main, ow, oh, GREEN_LT, GREEN,
                 ['Consistent Answers', '{a\u2081, a\u2082, ..., a\u2099}'],
                 fontsize=10.5, linewidth=2.2, radius=0.15)

# Checkmark
ax.text(x_output + ow/2 - 0.25, y_main + oh/2 - 0.2, '\u2713',
        ha='center', va='center', fontsize=13, zorder=6,
        color=GREEN, alpha=0.4, fontweight='bold')

# ─── Save ──────────────────────────────────────────────────────────────
out_dir = "/Users/aayambansal/Desktop/VStudio/#research-repos/iclr-2/figures"

png_path = f"{out_dir}/cgd_pipeline.png"
fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor=WHITE,
            edgecolor='none', pad_inches=0.15)
print(f"Saved PNG: {png_path}")

pdf_path = f"{out_dir}/cgd_pipeline.pdf"
fig.savefig(pdf_path, bbox_inches='tight', facecolor=WHITE,
            edgecolor='none', pad_inches=0.15)
print(f"Saved PDF: {pdf_path}")

plt.close(fig)
print("Done!")
