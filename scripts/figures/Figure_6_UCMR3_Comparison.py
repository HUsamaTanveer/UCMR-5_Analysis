"""
Figure 6: UCMR 3 vs UCMR 5 Historical Comparison and Regulatory Progress
Three-panel figure matching original template.

Panel A: Detection rate comparison (UCMR 3, UCMR 5 raw, UCMR 5 MRL-adjusted)
Panel B: Median concentration decline (PFOS, PFOA, PFHxS)
Panel C: Regulatory timeline

Data sources:
  - ucmr3_vs_ucmr5_mrl_adjusted.csv (Panel A)
  - VERIFIED_median_concentrations_full.csv (Panel B)
FIX: PFHpA raw 5.25 → 5.37 (from CSV)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from pathlib import Path

OUTPUT_PATH = Path("/sessions/kind-modest-dijkstra/mnt/5_PFAS_Research_Finalization/05_Figures")

# ═══════════════════════════════════════════════════════════════════
# COLOR SCHEME — matching original template
# ═══════════════════════════════════════════════════════════════════
DARK_BLUE = '#2C5F8A'
GOLDEN    = '#DAA520'
TEAL      = '#2E8B7D'
DARK_RED_LABEL = '#B71C1C'

CLR_NPDWR       = '#8B0000'
CLR_UCMR        = '#4682B4'
CLR_HA          = '#DAA520'
CLR_STEWARDSHIP = '#4CAF50'

# ═══════════════════════════════════════════════════════════════════
# PANEL A DATA — from ucmr3_vs_ucmr5_mrl_adjusted.csv
# FIX: PFHpA raw was 5.25 in v2, CSV says 5.37
# ═══════════════════════════════════════════════════════════════════
compounds_a = ['PFOS', 'PFOA', 'PFHxS', 'PFNA', 'PFBS', 'PFHpA']

ucmr3_detection     = [1.93, 2.38, 1.12, 0.28, 0.16, 1.75]
ucmr5_raw_detection = [12.98, 12.52, 9.92, 0.81, 16.27, 5.37]  # PFHpA FIXED
ucmr5_mrl_adjusted  = [0.37, 0.59, 0.29, 0.09, 0.14, 0.44]

deltas = [round(m - u, 2) for m, u in zip(ucmr5_mrl_adjusted, ucmr3_detection)]
# Results: [-1.56, -1.79, -0.83, -0.19, -0.02, -1.31]

# ═══════════════════════════════════════════════════════════════════
# PANEL B DATA — Median concentrations (ng/L)
# From VERIFIED_median_concentrations_full.csv
# ═══════════════════════════════════════════════════════════════════
compounds_b = ['PFOS', 'PFOA', 'PFHxS']
ucmr3_median = [60, 30, 80]       # ng/L
ucmr5_median = [6.8, 6.1, 5.0]    # ng/L
pct_decline  = ['-89%', '-80%', '-94%']

# ═══════════════════════════════════════════════════════════════════
# PANEL C DATA — Regulatory timeline
# ═══════════════════════════════════════════════════════════════════
events = [
    ('2024: NPDWR finalized',            CLR_NPDWR,       False),
    ('2023–25: UCMR 5',                  CLR_UCMR,        True),
    ('2022: Updated HA (0.004 ng/L)',     CLR_HA,          False),
    ('2016: Health Advisory (70 ng/L)',   CLR_HA,          False),
    ('2013–15: UCMR 3',                  CLR_UCMR,        True),
    ('2006: EPA PFOA Stewardship',        CLR_STEWARDSHIP, False),
    ('2000: 3M PFOS phase-out',           CLR_UCMR,       True),
]

# ═══════════════════════════════════════════════════════════════════
# CREATE FIGURE — 1×3 layout matching original
# ═══════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 9))
gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 0.7, 0.7],
                      wspace=0.28,
                      left=0.05, right=0.97, top=0.94, bottom=0.10)

# ─── PANEL A: Detection rates — horizontal grouped bars ───────────
ax_a = fig.add_subplot(gs[0, 0])

y_a = np.arange(len(compounds_a))
bar_h = 0.22

ax_a.barh(y_a + bar_h, ucmr3_detection, bar_h,
          color=DARK_BLUE, label='UCMR 3', zorder=2)
ax_a.barh(y_a, ucmr5_raw_detection, bar_h,
          color=GOLDEN, label='UCMR 5 (raw)', zorder=2)
ax_a.barh(y_a - bar_h, ucmr5_mrl_adjusted, bar_h,
          color=TEAL, label='UCMR 5 (MRL-adjusted)', zorder=2)

ax_a.set_yticks(y_a)
ax_a.set_yticklabels(compounds_a, fontsize=11, fontweight='medium')
ax_a.set_xlabel('PWS detection rate (%)', fontsize=11, fontweight='medium')
ax_a.invert_yaxis()
ax_a.grid(axis='x', alpha=0.3, linestyle='-', color='#CCCCCC')
ax_a.set_axisbelow(True)
ax_a.spines['top'].set_visible(False)
ax_a.spines['right'].set_visible(False)

# Delta annotations at end of widest bar
for i, (d, adj_val, raw_val) in enumerate(zip(deltas, ucmr5_mrl_adjusted, ucmr5_raw_detection)):
    max_val = max(ucmr3_detection[i], raw_val, adj_val)
    ax_a.text(max_val + 0.3, i, f'Δ{d}pp',
              va='center', fontsize=9, fontstyle='italic',
              color=GOLDEN, fontweight='bold')

ax_a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),
            ncol=3, fontsize=9.5, frameon=True, framealpha=0.95,
            edgecolor='#CCCCCC', fancybox=True)

ax_a.text(-0.06, 1.03, 'A', transform=ax_a.transAxes,
          fontsize=18, fontweight='bold', va='top')

# ─── PANEL B: Median concentration decline ────────────────────────
ax_b = fig.add_subplot(gs[0, 1])

x_b = np.arange(len(compounds_b))
bar_w = 0.35

ax_b.bar(x_b - bar_w/2, ucmr3_median, bar_w,
         color=DARK_BLUE, label='UCMR 3', zorder=2)
ax_b.bar(x_b + bar_w/2, ucmr5_median, bar_w,
         color=GOLDEN, label='UCMR 5', zorder=2)

for i, (u3, u5, pct) in enumerate(zip(ucmr3_median, ucmr5_median, pct_decline)):
    ax_b.text(i, max(u3, u5) + 2, pct, ha='center', va='bottom',
              fontsize=12, fontweight='bold', color=DARK_RED_LABEL)

ax_b.set_xticks(x_b)
ax_b.set_xticklabels(compounds_b, fontsize=11)
ax_b.set_ylabel('Median concentration (ng/L)', fontsize=11, fontweight='medium')
ax_b.set_ylim(0, 90)
ax_b.grid(axis='y', alpha=0.3, linestyle='-', color='#CCCCCC')
ax_b.set_axisbelow(True)
ax_b.spines['top'].set_visible(False)
ax_b.spines['right'].set_visible(False)

ax_b.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),
            ncol=2, fontsize=9.5, frameon=True, framealpha=0.95,
            edgecolor='#CCCCCC', fancybox=True)

ax_b.text(-0.10, 1.03, 'B', transform=ax_b.transAxes,
          fontsize=18, fontweight='bold', va='top')

# ─── PANEL C: Regulatory Timeline ────────────────────────────────
ax_c = fig.add_subplot(gs[0, 2])
ax_c.set_xlim(-1, 12)
ax_c.set_ylim(-1.2, 7.5)
ax_c.axis('off')

n_events = len(events)
positions = list(range(n_events - 1, -1, -1))

timeline_x = 2.0

# Highlight bands
for i, (pos, (label, color, highlight)) in enumerate(zip(positions, events)):
    if highlight:
        band = FancyBboxPatch((-0.5, pos - 0.38), 12, 0.76,
                              boxstyle='round,pad=0.1',
                              facecolor='#DDEAF6', edgecolor='none',
                              alpha=0.7, zorder=0)
        ax_c.add_patch(band)

# Vertical timeline spine
ax_c.plot([timeline_x, timeline_x], [-0.5, 6.5],
          color='#C0C0C0', linewidth=5, solid_capstyle='round', zorder=1)

# Event markers and labels
for i, (pos, (label, color, highlight)) in enumerate(zip(positions, events)):
    msize = 14 if highlight else 11
    ax_c.plot(timeline_x, pos, 'o', markersize=msize, color=color,
              markeredgecolor='white', markeredgewidth=2, zorder=3)
    ax_c.text(timeline_x + 1.0, pos, label, fontsize=10.5, va='center',
              fontweight='medium', color='#333333')

# MRL note at bottom
ax_c.text(timeline_x + 1.0, -0.85,
          'MRLs: 20–90 ng/L → 2–4 ng/L',
          fontsize=9, fontstyle='italic', color='#777777')

ax_c.text(-0.05, 1.03, 'C', transform=ax_c.transAxes,
          fontsize=18, fontweight='bold', va='top')

# ═══════════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════════
fig.savefig(OUTPUT_PATH / 'Figure_6_UCMR3_Comparison_FINAL.png',
            dpi=600, bbox_inches='tight', facecolor='white')
fig.savefig(OUTPUT_PATH / 'Figure_6_UCMR3_Comparison_FINAL.pdf',
            bbox_inches='tight', facecolor='white')
print("Saved Figure_6_UCMR3_Comparison_FINAL.png (600 DPI) and .pdf")
plt.close()
