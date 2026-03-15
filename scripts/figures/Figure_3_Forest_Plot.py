"""
Figure 3: Forest Plot — Detection & MCL Exceedance Models
Two-panel side-by-side layout matching original template.

Shows 8 key predictors (5 continuous/binary + 3 EPA regions) per panel.
Significant predictors in blue, non-significant in gray.
Light-blue band behind EPA Region section.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path

OUTPUT_PATH = Path("/sessions/kind-modest-dijkstra/mnt/5_PFAS_Research_Finalization/05_Figures")

# ═══════════════════════════════════════════════════════════════════
# PANEL A DATA — Detection Model (from regression CSV, verified)
# Only key predictors shown (matching original template selection)
# ═══════════════════════════════════════════════════════════════════
det_labels = [
    'Surface water source',
    'log(Pop. served)',
    'Private ownership',
    'log(Dist. Superfund)',
    'log(Dist. TRI spills)',
    '',  # spacer row between sections
    'EPA Region 1\n(New England)',
    'EPA Region 3\n(Mid-Atlantic)',
    'EPA Region 4\n(Southeast)',
]
det_or  = [2.037, 1.695, 1.393, 0.580, 0.683, None, 1.558, 1.400, 1.899]
det_lo  = [1.858, 1.556, 1.219, 0.521, 0.610, None, 1.248, 1.160, 1.633]
det_hi  = [2.233, 1.846, 1.593, 0.646, 0.765, None, 1.946, 1.688, 2.208]
det_sig = [True,  True,  True,  True,  True,  None, True,  True,  True]

# ═══════════════════════════════════════════════════════════════════
# PANEL B DATA — MCL Exceedance Model (from regression CSV, verified)
# ═══════════════════════════════════════════════════════════════════
exc_labels = [
    'Surface water source',
    'log(Pop. served)',
    'Private ownership',
    'log(Dist. Superfund)',
    'log(Dist. TRI spills)',
    '',  # spacer
    'EPA Region 1\n(New England)',
    'EPA Region 3\n(Mid-Atlantic)',
    'EPA Region 4\n(Southeast)',
]
exc_or  = [1.087, 1.866, 1.688, 0.556, 0.684, None, 4.351, 3.180, 4.880]
exc_lo  = [0.965, 1.679, 1.439, 0.485, 0.592, None, 3.348, 2.492, 3.956]
exc_hi  = [1.224, 2.074, 1.980, 0.636, 0.790, None, 5.654, 4.059, 6.021]
exc_sig = [False, True,  True,  True,  True,  None, True,  True,  True]


def make_panel(ax, labels, ors, los, his, sigs, title, xlim_right):
    """Draw one forest plot panel matching the original template."""
    n = len(labels)
    y_pos = list(range(n))[::-1]  # top-down ordering

    # ─── Light blue background band for EPA Region section ────────
    spacer_idx = labels.index('')
    spacer_y = y_pos[spacer_idx]
    ax.axhspan(-0.8, spacer_y - 0.2, facecolor='#EBF0F7', edgecolor='none',
               zorder=-1)

    # ─── Reference line at OR = 1 (null effect) ──────────────────
    ax.axvline(x=1.0, color='#555555', linewidth=0.9, linestyle='--',
               zorder=1, alpha=0.7)

    # ─── Plot each predictor ─────────────────────────────────────
    for i in range(n):
        if ors[i] is None:
            continue

        color = '#2166AC' if sigs[i] else '#777777'
        y = y_pos[i]

        # CI whisker line
        ax.plot([los[i], his[i]], [y, y], color=color, linewidth=2.0,
                solid_capstyle='round', zorder=2)
        # Point estimate dot
        ax.plot(ors[i], y, 'o', color=color, markersize=7, zorder=3,
                markeredgecolor='white', markeredgewidth=0.6)

        # OR (CI) text label — positioned right of CI upper bound
        ci_text = f'{ors[i]:.2f} ({los[i]:.2f}\u2013{his[i]:.2f})'
        # Ensure text starts at a minimum x to avoid crowding the dot
        min_label_x = 1.15
        label_x = max(his[i] * 1.10, min_label_x)
        ax.text(label_x, y, ci_text, va='center', ha='left', fontsize=7.5,
                color=color, fontweight='medium', zorder=5)

    # ─── Axis styling ────────────────────────────────────────────
    ax.set_xscale('log')
    ax.set_xlim(0.20, xlim_right)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f'{x:g}'))
    ax.set_xticks([0.25, 0.5, 1, 2, 4, 8])
    ax.set_xlabel('Odds ratio (95% CI)', fontsize=10, fontweight='medium')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_ylim(-0.8, n - 0.3)

    ax.set_title(title, fontsize=12.5, fontweight='bold', pad=14)

    # Clean spines — no left spine, no top/right
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.tick_params(axis='x', direction='out', length=4)


# ═══════════════════════════════════════════════════════════════════
# CREATE FIGURE — side-by-side panels
# ═══════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 5.5),
                                gridspec_kw={'wspace': 0.50})

# Panel A: Detection — CIs max at ~2.2, so xlim 8 is plenty
make_panel(ax1, det_labels, det_or, det_lo, det_hi, det_sig,
           'A    Detection Model (N = 10,137)', xlim_right=9)

# Panel B: Exceedance — CIs go up to 6.02, need more room for labels
make_panel(ax2, exc_labels, exc_or, exc_lo, exc_hi, exc_sig,
           'B    MCL Exceedance Model (N = 10,137)', xlim_right=12)

# ═══════════════════════════════════════════════════════════════════
# SAVE — 600 DPI PNG + vector PDF
# ═══════════════════════════════════════════════════════════════════
fig.savefig(OUTPUT_PATH / 'Figure_3_Forest_Plot_FINAL.png',
            dpi=600, bbox_inches='tight', facecolor='white')
fig.savefig(OUTPUT_PATH / 'Figure_3_Forest_Plot_FINAL.pdf',
            bbox_inches='tight', facecolor='white')
print("Saved Figure_3_Forest_Plot_FINAL.png (600 DPI) and .pdf")
plt.close()
