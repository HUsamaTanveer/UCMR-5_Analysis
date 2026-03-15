"""
Figure 5: Treatment Gap & Running Annual Average (RAA) Analysis
Two-panel figure matching original template.

Panel A: Treatment status by system size (stacked bars)
Panel B: Screening vs RAA proxy per regulated compound (grouped vertical bars)

CORRECTED data: PFOA RAA proxy 438→436, conversion 34.8%→34.6%
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUTPUT_PATH = Path("/sessions/kind-modest-dijkstra/mnt/5_PFAS_Research_Finalization/05_Figures")

# ═══════════════════════════════════════════════════════════════════
# COLOR SCHEME — matching original template
# ═══════════════════════════════════════════════════════════════════
TEAL = '#009688'
BURNT_ORANGE = '#E65100'
DARK_GOLD = '#C8A415'
NAVY_BLUE = '#1565C0'
RED_LABEL = '#B71C1C'

# ═══════════════════════════════════════════════════════════════════
# PANEL A DATA — Treatment status among MCL-exceeding systems
# By system size category (n=1,717 total)
# ═══════════════════════════════════════════════════════════════════
size_labels = ['Very Small\n(≤3,300)', 'Small\n(3,301–10K)',
               'Medium\n(10K–100K)', 'Large\n(>100K)']
total_exceed = [93, 575, 884, 165]
has_treatment = [3, 32, 109, 31]
no_treatment = [90, 543, 775, 134]
treat_pct = [3, 6, 12, 19]

# ═══════════════════════════════════════════════════════════════════
# PANEL B DATA — Screening exceedance vs RAA proxy
# Per regulated compound (corrected PFOA values)
# ═══════════════════════════════════════════════════════════════════
compounds = ['PFOS', 'PFOA', 'PFHxS', 'PFNA', 'HFPO-DA']
screening = [1309, 1259, 171, 21, 16]
raa_proxy = [514, 436, 35, 6, 9]
conv_pct = [39.3, 34.6, 20.5, 28.6, 56.2]

# ═══════════════════════════════════════════════════════════════════
# CREATE FIGURE — 1×2 layout matching original
# ═══════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5),
                                gridspec_kw={'wspace': 0.35})

# ─── PANEL A: Treatment Status — stacked vertical bars ────────────
x_a = np.arange(len(size_labels))
bar_w = 0.6

ax1.bar(x_a, has_treatment, bar_w,
        color=TEAL, label='Has PFAS treatment', zorder=2)
ax1.bar(x_a, no_treatment, bar_w, bottom=has_treatment,
        color=BURNT_ORANGE, label='No PFAS treatment', zorder=2)

# Treatment % labels above each bar
for i, (t, pct) in enumerate(zip(total_exceed, treat_pct)):
    ax1.text(i, t + 18, f'{pct}% treated', ha='center', va='bottom',
             fontsize=10, fontweight='bold', color=BURNT_ORANGE)

ax1.set_xticks(x_a)
ax1.set_xticklabels(size_labels, fontsize=9)
ax1.set_xlabel('System size category', fontsize=10, fontweight='medium')
ax1.set_ylabel('Systems with MCL exceedance', fontsize=10, fontweight='medium')
ax1.set_title('Treatment Status Among MCL-Exceeding Systems\n(n=1,717 total)',
              fontsize=12, fontweight='bold')
ax1.set_ylim(0, 1050)

ax1.yaxis.grid(True, linestyle='-', alpha=0.3, color='#888888', zorder=0)
ax1.set_axisbelow(True)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Legend below panel
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
           ncol=2, frameon=True, fontsize=9, framealpha=1,
           edgecolor='#999999')

# Panel label
ax1.text(-0.08, 1.05, 'A', transform=ax1.transAxes,
         fontsize=14, fontweight='bold', va='top')

# ─── PANEL B: Screening vs RAA — grouped vertical bars ───────────
x_b = np.arange(len(compounds))
bar_w2 = 0.35

ax2.bar(x_b - bar_w2/2, screening, bar_w2,
        color=DARK_GOLD,
        label='Screening exceedance (single sample)',
        zorder=2)
ax2.bar(x_b + bar_w2/2, raa_proxy, bar_w2,
        color=NAVY_BLUE,
        label='RAA proxy (running annual average)',
        zorder=2)

# Conversion % labels above each pair
for i, (s, r, pct) in enumerate(zip(screening, raa_proxy, conv_pct)):
    label_y = s + 25
    ax2.text(i, label_y, f'{pct:.1f}%', ha='center', va='bottom',
             fontsize=9.5, fontweight='bold', color=RED_LABEL)

ax2.set_xticks(x_b)
ax2.set_xticklabels(compounds, fontsize=10)
ax2.set_xlabel('Regulated compound', fontsize=10, fontweight='medium')
ax2.set_ylabel('Systems exceeding MCL', fontsize=10, fontweight='medium')
ax2.set_title('Screening vs Running Annual Average (RAA)\n(Regulatory compliance assessment)',
              fontsize=12, fontweight='bold')
ax2.set_ylim(0, 1550)

ax2.yaxis.grid(True, linestyle='-', alpha=0.3, color='#888888', zorder=0)
ax2.set_axisbelow(True)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Callout box — top right with orange border
callout_text = ('Overall conversion: ~40%\n'
                '(60% of screening exceedances\n'
                'would not violate annual avg)')
ax2.annotate(callout_text, xy=(0.97, 0.97), xycoords='axes fraction',
             ha='right', va='top', fontsize=8.5,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF8E1',
                       edgecolor='#E65100', linewidth=1.2))

# Legend below panel
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
           ncol=2, frameon=True, fontsize=9, framealpha=1,
           edgecolor='#999999')

# Panel label
ax2.text(-0.08, 1.05, 'B', transform=ax2.transAxes,
         fontsize=14, fontweight='bold', va='top')

# ═══════════════════════════════════════════════════════════════════
# SAVE — 600 DPI PNG + vector PDF
# ═══════════════════════════════════════════════════════════════════
fig.savefig(OUTPUT_PATH / 'Figure_5_Treatment_RAA_FINAL.png',
            dpi=600, bbox_inches='tight', facecolor='white')
fig.savefig(OUTPUT_PATH / 'Figure_5_Treatment_RAA_FINAL.pdf',
            bbox_inches='tight', facecolor='white')
print("Saved Figure_5_Treatment_RAA_FINAL.png (600 DPI) and .pdf")
plt.close()
