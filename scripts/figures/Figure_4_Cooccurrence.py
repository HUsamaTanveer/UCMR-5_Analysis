"""
Figure 4: PFAS Co-occurrence Patterns and MCL Exceedance Association
Three-panel figure matching original template.

Panel A: Compound count distribution (detecting systems only, 1–13)
Panel B: MCL exceedance rate by compound count (grouped bins) with Spearman ρ
Panel C: Regulated compound co-occurrence (PFOS/PFOA breakdown)

Data sourced from canonical Jan 2026 CSVs.
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
STEEL_BLUE = "#4682B4"
DARK_RED = "#8B0000"

# ═══════════════════════════════════════════════════════════════════
# PANEL A DATA — from cooccurrence_compound_count_distribution.csv
# Detecting systems only (compound count 1–13), N = 3,539
# ═══════════════════════════════════════════════════════════════════
compound_counts = np.arange(1, 14)  # 1 through 13
system_counts = [1134, 512, 424, 403, 313, 262, 232, 177, 57, 18, 5, 1, 1]
# Verify sum
assert sum(system_counts) == 3539, f"Sum is {sum(system_counts)}, expected 3539"

# ═══════════════════════════════════════════════════════════════════
# PANEL B DATA — from cooccurrence_compound_count_vs_severity.csv
# Grouped bins with MCL exceedance rates
# ═══════════════════════════════════════════════════════════════════
bins = ['1', '2', '3', '4-5', '6-8', '9+']
mcl_exceedance_rates = [13.9, 37.3, 41.3, 63.3, 98.1, 100.0]
# Color gradient: blue → yellow → red (RdYlBu_r) matching original
colors_b = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(bins)))

# Spearman correlation (from analysis output)
spearman_rho = 0.624

# ═══════════════════════════════════════════════════════════════════
# PANEL C DATA — Regulated compound co-occurrence patterns
# Among systems with ANY regulated PFAS exceedance
# ═══════════════════════════════════════════════════════════════════
cooccurrence_labels = ['Both PFOS\n& PFOA', 'PFOS only', 'PFOA only', 'Other reg.\nonly']
cooccurrence_counts = [875, 434, 384, 24]

# ═══════════════════════════════════════════════════════════════════
# CREATE FIGURE — 1×3 horizontal panels (NO suptitle, matching original)
# ═══════════════════════════════════════════════════════════════════
fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(16, 5),
                                         gridspec_kw={'width_ratios': [1.0, 0.9, 0.9],
                                                      'wspace': 0.35})

# ─── PANEL A: Compound Count Distribution ─────────────────────────
ax_a.bar(compound_counts, system_counts, color=STEEL_BLUE,
         edgecolor='black', linewidth=0.5, width=0.8)
ax_a.set_xlabel('Number of PFAS compounds detected', fontsize=10)
ax_a.set_ylabel('Number of systems', fontsize=10)
ax_a.set_xticks(compound_counts)
ax_a.set_xticklabels(compound_counts, fontsize=9)
ax_a.set_ylim(0, max(system_counts) * 1.15)
ax_a.grid(axis='y', alpha=0.3, linestyle='--')
ax_a.set_axisbelow(True)
ax_a.spines['top'].set_visible(False)
ax_a.spines['right'].set_visible(False)

# Panel A label
ax_a.text(-0.08, 1.04, 'A', transform=ax_a.transAxes,
          fontsize=16, fontweight='bold', va='top')

# ─── PANEL B: Exceedance Rate by Compound Count ──────────────────
x_b = np.arange(len(bins))
ax_b.bar(x_b, mcl_exceedance_rates, color=colors_b,
         edgecolor='black', linewidth=0.5, width=0.7)
ax_b.set_xlabel('Number of compounds detected', fontsize=10)
ax_b.set_ylabel('MCL exceedance rate (%)', fontsize=10)
ax_b.set_xticks(x_b)
ax_b.set_xticklabels(bins, fontsize=9)
ax_b.set_ylim(0, 115)
ax_b.grid(axis='y', alpha=0.3, linestyle='--')
ax_b.set_axisbelow(True)
ax_b.spines['top'].set_visible(False)
ax_b.spines['right'].set_visible(False)

# Percentage labels above each bar
for i, val in enumerate(mcl_exceedance_rates):
    ax_b.text(i, val + 2.5, f'{val}%', ha='center', va='bottom',
              fontsize=9.5, fontweight='bold')

# Spearman stats box — top-left inside panel (peer review requirement)
stats_text = f'Spearman ρ = {spearman_rho}\np < 10⁻²⁹⁰'
ax_b.text(0.02, 0.97, stats_text,
          transform=ax_b.transAxes, fontsize=9, fontweight='bold',
          verticalalignment='top', horizontalalignment='left',
          bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                    edgecolor='black', linewidth=0.8, alpha=0.95))

# Panel B label
ax_b.text(-0.08, 1.04, 'B', transform=ax_b.transAxes,
          fontsize=16, fontweight='bold', va='top')

# ─── PANEL C: Regulated Compound Co-occurrence ───────────────────
y_c = np.arange(len(cooccurrence_labels))
ax_c.barh(y_c, cooccurrence_counts, color=DARK_RED,
          edgecolor='black', linewidth=0.5, height=0.6)
ax_c.set_yticks(y_c)
ax_c.set_yticklabels(cooccurrence_labels, fontsize=10)
ax_c.set_xlabel('Number of systems', fontsize=10)
ax_c.invert_yaxis()
ax_c.set_xlim(0, max(cooccurrence_counts) * 1.15)
ax_c.grid(axis='x', alpha=0.3, linestyle='--')
ax_c.set_axisbelow(True)
ax_c.spines['top'].set_visible(False)
ax_c.spines['right'].set_visible(False)

# Count labels at end of bars
for i, count in enumerate(cooccurrence_counts):
    ax_c.text(count + 10, i, f'{count}', va='center', fontsize=10,
              fontweight='bold')

# Panel C label
ax_c.text(-0.08, 1.04, 'C', transform=ax_c.transAxes,
          fontsize=16, fontweight='bold', va='top')

# ═══════════════════════════════════════════════════════════════════
# SAVE — 600 DPI PNG + vector PDF
# ═══════════════════════════════════════════════════════════════════
fig.savefig(OUTPUT_PATH / 'Figure_4_Cooccurrence_FINAL.png',
            dpi=600, bbox_inches='tight', facecolor='white')
fig.savefig(OUTPUT_PATH / 'Figure_4_Cooccurrence_FINAL.pdf',
            bbox_inches='tight', facecolor='white')
print("Saved Figure_4_Cooccurrence_FINAL.png (600 DPI) and .pdf")
plt.close()
