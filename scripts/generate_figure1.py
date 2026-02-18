#!/usr/bin/env python3
"""
Generate Figure 1: ConsistencyBench Overview — The Consistency Problem & Gap.

Publication-quality schematic for a full-width conference figure (ICLR style).
Outputs PDF (vector) and PNG (300 DPI).
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

# ── Palette ───────────────────────────────────────────────────────────────────
BLUE      = "#4A90D9"
RED       = "#E74C3C"
GREEN     = "#27AE60"
GRAY      = "#95A5A6"
DARK      = "#2C3E50"
LIGHT_BG  = "#F8F9FA"
LIGHT_BLUE = "#EBF2FA"
LIGHT_RED  = "#FDEDEC"
LIGHT_GREEN = "#EAFAF1"
WHITE     = "#FFFFFF"

# DejaVu Sans has checkmark / ballot-x glyphs
_SYM = fm.FontProperties(family="DejaVu Sans", weight="bold")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "text.usetex": False,
})


# ── Drawing helpers (all coordinates in Axes fraction) ────────────────────────
def rbox(ax, x, y, w, h, fc=WHITE, ec=DARK, lw=0.8, radius=0.015, zorder=2):
    """Rounded-corner rectangle *without* inflation.  Returns the patch."""
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw,
        transform=ax.transData, zorder=zorder, clip_on=False,
    )
    ax.add_patch(p)
    return p


def arrow(ax, x0, y0, x1, y1, color=DARK, lw=1.0, style="->", zorder=4):
    a = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle=style, color=color, linewidth=lw,
        mutation_scale=10, transform=ax.transData, zorder=zorder,
        clip_on=False,
    )
    ax.add_patch(a)
    return a


# ── Panel (a)  ────────────────────────────────────────────────────────────────
def draw_panel_a(ax):
    """The Consistency Problem — concrete example of cross-query inconsistency."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Panel label
    ax.text(0.0, 10.3, "(a)", fontsize=11, fontweight="bold", color=DARK)
    ax.text(0.55, 10.3, "The Consistency Problem", fontsize=10, fontweight="bold", color=DARK)

    # ── Shared Premise ────────────────────────────────────────────────────
    rbox(ax, 0.2, 8.7, 9.6, 0.85, fc=LIGHT_BLUE, ec=BLUE, lw=1.2, radius=0.06)
    ax.text(0.55, 9.12, "Shared Premise:", fontsize=8.5, fontweight="bold",
            color=BLUE, va="center")
    ax.text(2.4, 9.12, '"If it rains, Alice goes to the gym."',
            fontsize=8.5, fontstyle="italic", color=DARK, va="center")

    # Down-arrow
    arrow(ax, 5.0, 8.6, 5.0, 8.2, color=GRAY, lw=0.8)

    # ── Question rows ────────────────────────────────────────────────────
    questions = [
        dict(label="Q1",
             text='"If it rains, does Alice go to the gym?"',
             tag="Direct implication",
             answer="Yes", correct=True),
        dict(label="Q2",
             text='"If Alice did NOT go to the gym,\n can we conclude it didn\'t rain?"',
             tag="Contrapositive (logically equiv. to Q1)",
             answer="No", correct=False),
        dict(label="Q3",
             text='"If Alice went to the gym,\n can we conclude it rained?"',
             tag="Affirming the consequent (fallacy)",
             answer="Yes", correct=False),
    ]

    row_h = 1.45
    row_gap = 0.22
    top_y = 6.55

    for i, q in enumerate(questions):
        y = top_y - i * (row_h + row_gap)
        bg = LIGHT_GREEN if q["correct"] else LIGHT_RED
        ec = GREEN if q["correct"] else RED

        # Row box
        rbox(ax, 0.2, y, 9.6, row_h, fc=bg, ec=ec, lw=1.0, radius=0.06)

        # Label badge
        badge_w, badge_h = 0.55, 0.55
        bx = 0.5
        by = y + row_h / 2 - badge_h / 2
        rbox(ax, bx, by, badge_w, badge_h, fc=ec, ec=ec, lw=0, radius=0.06)
        ax.text(bx + badge_w / 2, by + badge_h / 2, q["label"],
                fontsize=8, fontweight="bold", color=WHITE,
                ha="center", va="center")

        # Question text
        ax.text(1.3, y + row_h * 0.62, q["text"],
                fontsize=8, color=DARK, va="center", linespacing=1.25)

        # Relation tag
        ax.text(1.3, y + row_h * 0.18, q["tag"],
                fontsize=6.5, color=GRAY, va="center", fontstyle="italic")

        # ── Answer badge (right side) ────────────────────────────────────
        ans_x = 7.0
        ans_w = 2.55
        ans_h = 0.55
        ans_y = y + row_h / 2 - ans_h / 2
        rbox(ax, ans_x, ans_y, ans_w, ans_h, fc=WHITE, ec=ec, lw=0.8, radius=0.04)

        ax.text(ans_x + 0.3, ans_y + ans_h / 2,
                f'"{q["answer"]}"', fontsize=8, fontweight="bold",
                color=DARK, va="center")

        icon = "\u2713" if q["correct"] else "\u2717"
        icol = GREEN if q["correct"] else RED
        status = "CORRECT" if q["correct"] else "WRONG"
        ax.text(ans_x + 1.0, ans_y + ans_h / 2, icon,
                fontsize=9.5, color=icol, va="center",
                fontproperties=_SYM)
        ax.text(ans_x + 1.35, ans_y + ans_h / 2, status,
                fontsize=7.5, fontweight="bold", color=icol, va="center")

    # ── Bracket linking Q1 & Q2 ──────────────────────────────────────────
    bx = 9.95
    y1_mid = top_y + row_h / 2
    y2_mid = top_y - (row_h + row_gap) + row_h / 2
    ax.plot([bx, bx + 0.15, bx + 0.15, bx],
            [y1_mid, y1_mid, y2_mid, y2_mid],
            color=RED, lw=1.0, solid_capstyle="round", clip_on=False)

    # ── Annotation ───────────────────────────────────────────────────────
    ann_y = 2.7
    ax.text(5.0, ann_y,
            "Q1 & Q2 are logically equivalent (contrapositive), "
            "but the model contradicts itself",
            fontsize=7.5, color=RED, ha="center", fontstyle="italic",
            fontweight="medium")

    # ── Summary metrics box ──────────────────────────────────────────────
    rbox(ax, 0.2, 0.6, 9.6, 1.8, fc=LIGHT_BG, ec=GRAY, lw=0.8, radius=0.06)

    # Individual Accuracy
    ax.text(2.0, 2.0, "Individual Accuracy",
            fontsize=8.5, fontweight="bold", color=BLUE, ha="center")
    ax.text(2.0, 1.3, "1 / 3 = 33%",
            fontsize=10, fontweight="bold", color=BLUE, ha="center")

    # Gap arrow
    arrow(ax, 3.5, 1.55, 6.0, 1.55, color=RED, lw=1.8, style="-|>")
    ax.text(4.75, 2.05, "CONSISTENCY", fontsize=7.0, fontweight="bold",
            color=RED, ha="center")
    ax.text(4.75, 1.73, "GAP", fontsize=7.0, fontweight="bold",
            color=RED, ha="center")

    # Set Consistency Rate
    ax.text(7.8, 2.0, "Set Consistency Rate",
            fontsize=8.5, fontweight="bold", color=RED, ha="center")
    ax.text(7.8, 1.3, "0 / 1 = 0%",
            fontsize=10, fontweight="bold", color=RED, ha="center")


# ── Panel (b)  ────────────────────────────────────────────────────────────────
def draw_panel_b(ax):
    """The Consistency Gap — grouped bar chart of IA vs SCR."""
    models   = ["GPT-4.1", "Claude\nOpus 4.6", "GPT-4o", "Gemini\n2.5 Pro"]
    ia_vals  = [83, 75, 71, 65]
    scr_vals = [47, 23, 20, 12]

    x = np.arange(len(models))
    bw = 0.32

    bars_ia  = ax.bar(x - bw/2, ia_vals,  bw, label="IA (Individual Accuracy)",
                      color=BLUE, edgecolor="white", linewidth=0.5, zorder=3)
    bars_scr = ax.bar(x + bw/2, scr_vals, bw, label="SCR (Set Consistency Rate)",
                      color=RED, edgecolor="white", linewidth=0.5, zorder=3)

    # Value labels
    for bar in bars_ia:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1.5,
                f"{int(h)}%", ha="center", va="bottom",
                fontsize=7.5, fontweight="bold", color=BLUE)
    for bar in bars_scr:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1.5,
                f"{int(h)}%", ha="center", va="bottom",
                fontsize=7.5, fontweight="bold", color=RED)

    # Gap brackets
    for i in range(len(models)):
        ia_h, scr_h = ia_vals[i], scr_vals[i]
        gap = ia_h - scr_h
        bx = x[i] + bw/2 + 0.22
        mid_y = (ia_h + scr_h) / 2
        ax.plot([bx, bx], [scr_h + 1, ia_h - 1],
                color=DARK, lw=0.6, zorder=4, clip_on=False)
        ax.plot([bx - 0.08, bx], [ia_h - 1, ia_h - 1],
                color=DARK, lw=0.6, zorder=4, clip_on=False)
        ax.plot([bx - 0.08, bx], [scr_h + 1, scr_h + 1],
                color=DARK, lw=0.6, zorder=4, clip_on=False)
        ax.text(bx + 0.08, mid_y, f"\u0394{gap}",
                fontsize=6.5, fontweight="bold", color=DARK,
                va="center", ha="left")

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=7.5)
    ax.set_ylabel("Score (%)", fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.yaxis.set_tick_params(labelsize=7.5)
    ax.yaxis.grid(True, alpha=0.25, lw=0.5, zorder=0)
    ax.set_axisbelow(True)

    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(0.6)

    leg = ax.legend(fontsize=7, loc="upper right", frameon=True,
                    framealpha=0.95, edgecolor=GRAY, borderpad=0.4,
                    handlelength=1.2, handletextpad=0.4)
    leg.get_frame().set_linewidth(0.5)

    # Panel label
    ax.text(-0.10, 1.06, "(b)", fontsize=11, fontweight="bold", color=DARK,
            transform=ax.transAxes)
    ax.text(0.02, 1.06, "The Consistency Gap", fontsize=10, fontweight="bold",
            color=DARK, transform=ax.transAxes)


# ── Compose and save ─────────────────────────────────────────────────────────
def main():
    fig = plt.figure(figsize=(10, 4), facecolor=WHITE)
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.5, 1],
                  wspace=0.32, left=0.03, right=0.97, top=0.91, bottom=0.08)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    draw_panel_a(ax_a)
    draw_panel_b(ax_b)

    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")
    os.makedirs(out_dir, exist_ok=True)

    pdf = os.path.join(out_dir, "figure1_overview.pdf")
    png = os.path.join(out_dir, "figure1_overview.png")

    fig.savefig(pdf, format="pdf", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(png, format="png", dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    print(f"Saved PDF: {pdf}")
    print(f"Saved PNG: {png}")


if __name__ == "__main__":
    main()
