"""
Figure S11: Hazard Index Sensitivity Analysis (Monte Carlo)
Two-scenario comparison: ±10% and ±20% HBWC variation.
Shows simulated distributions of HI exceedance counts.

Data source: hi_sensitivity_monte_carlo.csv
"""

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import shutil

OUTPUT_PATH = Path("/sessions/kind-modest-dijkstra/mnt/5_PFAS_Research_Finalization/05_Figures")
DELIVER_PATH = Path("/sessions/kind-modest-dijkstra/mnt/5_PFAS_Research_Finalization/08_Peer_Review_Deliverables/02_Figures")
DATA_PATH = Path("/sessions/kind-modest-dijkstra/mnt/5_PFAS_Research_Finalization/3_Jan2026_Update_Output_CSVs")

# ═══════════════════════════════════════════════════════════════════
# READ DATA
# ═══════════════════════════════════════════════════════════════════
df = pd.read_csv(DATA_PATH / 'hi_sensitivity_monte_carlo.csv')

print("HI Sensitivity Monte Carlo results:")
for _, row in df.iterrows():
    print(f"  {row['Scenario']}: N={int(row['N_Simulations'])}, "
          f"mean={row['Mean_HI_Exceed']:.1f}, median={row['Median_HI_Exceed']:.0f}, "
          f"P5={row['P5_HI_Exceed']:.0f}, P95={row['P95_HI_Exceed']:.0f}, "
          f"Pct_PWS={row['Pct_of_PWS']:.2f}%")

# ═══════════════════════════════════════════════════════════════════
# COLORS — matching template
# ═══════════════════════════════════════════════════════════════════
CLR_BLUE    = '#4682B4'
CLR_DKRED   = '#8B0000'
CLR_ORANGE  = '#E8751A'

# ═══════════════════════════════════════════════════════════════════
# CREATE FIGURE — Grouped comparison
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 7))

scenarios = df['Scenario'].values
n_scenarios = len(scenarios)
x_pos = np.arange(n_scenarios)
bar_width = 0.5

means = df['Mean_HI_Exceed'].values
medians = df['Median_HI_Exceed'].values
p5 = df['P5_HI_Exceed'].values
p95 = df['P95_HI_Exceed'].values
pct_pws = df['Pct_of_PWS'].values

# Asymmetric error bars from P5 to P95
err_lower = means - p5
err_upper = p95 - means

colors = [CLR_BLUE, CLR_ORANGE]

bars = ax.bar(x_pos, means, width=bar_width, color=colors,
              edgecolor='white', linewidth=1.0, alpha=0.85,
              yerr=[err_lower, err_upper], capsize=8,
              error_kw={'linewidth': 1.5, 'color': '#333333'})

# Value labels
for i, (bar, mean, med, pct) in enumerate(zip(bars, means, medians, pct_pws)):
    # Mean above bar
    ax.text(bar.get_x() + bar.get_width()/2., p95[i] + 5,
            f'Mean: {mean:.1f}\nMedian: {med:.0f}\n({pct:.1f}% of PWS)',
            ha='center', va='bottom', fontsize=9.5, color='#333333',
            fontweight='medium')

# Monte Carlo details annotation
mc_text = (f'Monte Carlo: 10,000 simulations\n'
           f'Error bars: 5th–95th percentile\n'
           f'HBWC = Health-Based Water Concentration')
props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9,
             edgecolor='#CCCCCC', linewidth=1.0)
ax.text(0.97, 0.97, mc_text, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', horizontalalignment='right', bbox=props)

ax.set_xticks(x_pos)
ax.set_xticklabels(scenarios, fontsize=11, fontweight='medium')
ax.set_ylabel('PWS with HI > 1 (count)', fontsize=11, fontweight='medium')
ax.set_xlabel('HBWC Variation Scenario', fontsize=11, fontweight='medium')
ax.set_ylim(0, max(p95) * 1.3)
ax.grid(axis='y', alpha=0.2, linestyle='-', color='#CCCCCC')
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

# ═══════════════════════════════════════════════════════════════════
# SAVE — 600 DPI PNG + PDF
# ═══════════════════════════════════════════════════════════════════
fig.savefig(OUTPUT_PATH / 'Figure_S11_HI_Sensitivity_FINAL.png',
            dpi=600, bbox_inches='tight', facecolor='white')
fig.savefig(OUTPUT_PATH / 'Figure_S11_HI_Sensitivity_FINAL.pdf',
            bbox_inches='tight', facecolor='white')

for ext in ['png', 'pdf']:
    shutil.copy2(OUTPUT_PATH / f'Figure_S11_HI_Sensitivity_FINAL.{ext}',
                 DELIVER_PATH / f'Figure_S11_HI_Sensitivity_FINAL.{ext}')

print(f"\n✓ Saved Figure_S11_HI_Sensitivity_FINAL.png (600 DPI) and .pdf")
print(f"✓ Copied to 08_Peer_Review_Deliverables/02_Figures/")
plt.close()
