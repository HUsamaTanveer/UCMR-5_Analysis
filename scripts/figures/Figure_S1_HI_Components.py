"""
Figure S1: Compound Contributions to Hazard Index
Single-panel bar chart matching original template STYLE,
but with UPDATED Jan 2026 data from CSV + significance test (peer review).

Data source: hi_component_contributions_enhanced.csv
N source: Raw UCMR5 data — 2,023 systems with any HI-component detected
         (verified Feb 2026: hazard_index_by_pws.csv was BUGGY, used SUM-ALL
          instead of MAX-per-compound; manuscript values 196/230 are correct)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from pathlib import Path

OUTPUT_PATH = Path("/sessions/jolly-relaxed-meitner/mnt/5_PFAS_Research_Finalization/05_Figures")

# ═══════════════════════════════════════════════════════════════════
# DATA — from hi_component_contributions_enhanced.csv (Jan 2026)
# ═══════════════════════════════════════════════════════════════════
compounds = ['PFHxS', 'PFNA', 'HFPO-DA', 'PFBS']
# Total HI contributions (absolute)
contributions = [829.67, 93.10, 90.19, 7.04]
total_hi = sum(contributions)  # 1020.0

# Percentages
percentages = [c / total_hi * 100 for c in contributions]
# [81.3%, 9.1%, 8.8%, 0.7%]

# N = 2,023 systems with any HI-component compound detected
# (verified from raw UCMR5 data; previous N=603 was from buggy CSV)
n_systems = 2023

# Colors — matching original template
colors = ['#CC7A00', '#DAA520', '#2C5F8A', '#808080']

# ═══════════════════════════════════════════════════════════════════
# SIGNIFICANCE TEST — Chi-square goodness-of-fit
# H0: All four compounds contribute equally (25% each)
# ═══════════════════════════════════════════════════════════════════
expected = [total_hi / 4] * 4  # equal distribution
chi2_stat, p_value = stats.chisquare(contributions, f_exp=expected)
# Format p-value
if p_value < 0.001:
    p_str = 'p < 0.001'
else:
    p_str = f'p = {p_value:.3f}'

print(f"Chi-square test: χ² = {chi2_stat:.2f}, {p_str}")

# ═══════════════════════════════════════════════════════════════════
# CREATE FIGURE — bar chart matching original style
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 6))

bars = ax.bar(compounds, percentages, color=colors,
              edgecolor='black', linewidth=1.5)

# Percentage labels on top of each bar
for bar, pct in zip(bars, percentages):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2., height,
            f'{pct:.1f}%',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

# Axis labels
ax.set_ylabel('% contribution to total Hazard Index',
              fontsize=12, fontweight='bold')
ax.set_ylim(0, max(percentages) * 1.15)

# Light grid
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.7)
ax.set_axisbelow(True)

# Clean spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ─── Annotation box (UPDATED: new N + significance test) ─────────
textstr = (f'N = {n_systems:,} systems with HI components\n'
           f'χ² = {chi2_stat:.1f}, {p_str}\n'
           f'(goodness-of-fit vs. equal contribution)')
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8,
             edgecolor='#CC7A00', linewidth=1.5)
ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=10.5,
        verticalalignment='top', horizontalalignment='right', bbox=props)

# Tick styling
ax.tick_params(axis='both', labelsize=11)

plt.tight_layout()

# ═══════════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════════
fig.savefig(OUTPUT_PATH / 'Figure_S1_HI_Components_FINAL.png',
            dpi=600, bbox_inches='tight', facecolor='white')
fig.savefig(OUTPUT_PATH / 'Figure_S1_HI_Components_FINAL.pdf',
            bbox_inches='tight', facecolor='white')
print("Saved Figure_S1_HI_Components_FINAL.png (600 DPI) and .pdf")
plt.close()
