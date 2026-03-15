"""
Figure S8: Cook's Distance — Top 20 Influential Observations
Horizontal lollipop chart of the most influential observations from logistic regression.

Data sources: regression_cooks_distance.csv, regression_diagnostics_enhanced.csv
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
cooks_df = pd.read_csv(DATA_PATH / 'regression_cooks_distance.csv')
diag_df = pd.read_csv(DATA_PATH / 'regression_diagnostics_enhanced.csv')

diag = dict(zip(diag_df['Metric'], diag_df['Value']))
n_influential = int(diag['N_Influential'])   # 233
pct_influential = diag['Pct_Influential']    # 2.3%

# Total N from H-L deciles: 10 groups × ~1014 = 10,137
N_total = 10137
threshold_4n = 4 / N_total

print(f"Top 20 influential observations (sorted by Cook's D):")
for _, row in cooks_df.iterrows():
    print(f"  Obs {int(row['observation_index'])}: D={row['cooks_distance']:.6f}, y={int(row['y_observed'])}, p̂={row['y_predicted']:.4f}")
print(f"\n4/N threshold = {threshold_4n:.6f}")
print(f"Total influential (D > 4/N): {n_influential} ({pct_influential}%)")

# ═══════════════════════════════════════════════════════════════════
# COLORS — matching template
# ═══════════════════════════════════════════════════════════════════
CLR_BLUE    = '#4682B4'
CLR_DKRED   = '#8B0000'
CLR_ORANGE  = '#E8751A'

# ═══════════════════════════════════════════════════════════════════
# CREATE FIGURE — Horizontal lollipop chart
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 8))

obs_labels = [f"Obs {int(idx)}" for idx in cooks_df['observation_index'].values]
cooks_vals = cooks_df['cooks_distance'].values
y_obs = cooks_df['y_observed'].values.astype(int)
y_pred = cooks_df['y_predicted'].values
n_obs = len(cooks_vals)
y_pos = np.arange(n_obs)

# Color by whether observation is positive or negative
colors = [CLR_DKRED if y == 1 else CLR_BLUE for y in y_obs]

# Horizontal lollipop
for i in range(n_obs):
    ax.plot([0, cooks_vals[i]], [y_pos[i], y_pos[i]], color=colors[i],
            linewidth=1.5, alpha=0.7)
ax.scatter(cooks_vals, y_pos, color=colors, s=60, zorder=5,
           edgecolors='white', linewidths=0.5)

# 4/N threshold
ax.axvline(x=threshold_4n, color='gray', linestyle='--', linewidth=1.5,
           alpha=0.7, label=f'4/N = {threshold_4n:.5f}')

# Predicted probability annotations — sitting just above each lollipop line
for i in range(n_obs):
    # Place at the midpoint of the lollipop, offset above the line
    mid_x = cooks_vals[i] / 2
    ax.text(mid_x, y_pos[i] - 0.28,
            f'p̂={y_pred[i]:.3f}', va='bottom', ha='center', fontsize=7,
            color='#777777', fontstyle='italic')

# Summary annotation
summary_text = (f'Total observations: {N_total:,}\n'
                f'Influential (D > 4/N): {n_influential} ({pct_influential}%)\n'
                f'Max D = {cooks_vals[0]:.4f}')
props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9,
             edgecolor='#CCCCCC', linewidth=1.0)
ax.text(0.97, 0.05, summary_text, transform=ax.transAxes, fontsize=9,
        verticalalignment='bottom', horizontalalignment='right', bbox=props)

# Legend for colors
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=CLR_DKRED, label='y = 1 (detected)'),
    Patch(facecolor=CLR_BLUE, label='y = 0 (not detected)'),
]
# Place legend in the center-right blank space (short lollipops, lots of white space)
ax.legend(handles=legend_elements, fontsize=9,
          framealpha=0.95, edgecolor='#CCCCCC',
          loc='center right',
          bbox_to_anchor=(0.98, 0.55))

ax.set_yticks(y_pos)
ax.set_yticklabels(obs_labels, fontsize=9)
ax.set_xlabel("Cook's Distance", fontsize=11, fontweight='medium')
ax.set_ylabel('Observation', fontsize=11, fontweight='medium')
ax.set_xlim(-0.00005, max(cooks_vals) * 1.12)  # breathing room on right
ax.invert_yaxis()  # Most influential at top
ax.grid(axis='x', alpha=0.2, linestyle='-', color='#CCCCCC')
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

# ═══════════════════════════════════════════════════════════════════
# SAVE — 600 DPI PNG + PDF
# ═══════════════════════════════════════════════════════════════════
fig.savefig(OUTPUT_PATH / 'Figure_S8_Cooks_Distance_FINAL.png',
            dpi=600, bbox_inches='tight', facecolor='white')
fig.savefig(OUTPUT_PATH / 'Figure_S8_Cooks_Distance_FINAL.pdf',
            bbox_inches='tight', facecolor='white')

for ext in ['png', 'pdf']:
    shutil.copy2(OUTPUT_PATH / f'Figure_S8_Cooks_Distance_FINAL.{ext}',
                 DELIVER_PATH / f'Figure_S8_Cooks_Distance_FINAL.{ext}')

print(f"\n✓ Saved Figure_S8_Cooks_Distance_FINAL.png (600 DPI) and .pdf")
print(f"✓ Copied to 08_Peer_Review_Deliverables/02_Figures/")
plt.close()
