"""
Figure S9: MCA Scree Plot (Eigenvalues and Cumulative Inertia)
Dual-axis: bars for individual inertia %, line for cumulative inertia %.

Data source: mca_eigenvalues_inertia.csv
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
df = pd.read_csv(DATA_PATH / 'mca_eigenvalues_inertia.csv')

dims = df['Dimension'].values
eigenvals = df['Eigenvalue'].values
inertia_pct = df['Inertia_Percent'].values
cum_inertia = df['Cumulative_Inertia_Percent'].values

print("MCA Scree data:")
for d, e, ip, ci in zip(dims, eigenvals, inertia_pct, cum_inertia):
    print(f"  Dim {d}: eigenvalue={e:.5f}, inertia={ip:.2f}%, cumulative={ci:.2f}%")

# ═══════════════════════════════════════════════════════════════════
# COLORS — matching template
# ═══════════════════════════════════════════════════════════════════
CLR_BLUE    = '#4682B4'
CLR_DKRED   = '#8B0000'

# ═══════════════════════════════════════════════════════════════════
# CREATE FIGURE — Dual-axis
# ═══════════════════════════════════════════════════════════════════
fig, ax1 = plt.subplots(figsize=(10, 7))

x_pos = np.arange(len(dims))

# Bar chart — individual inertia %
bars = ax1.bar(x_pos, inertia_pct, color=CLR_BLUE, edgecolor='white',
               width=0.6, alpha=0.85, label='Individual inertia (%)')

# Label bars
for i, (bar, pct) in enumerate(zip(bars, inertia_pct)):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
             f'{pct:.1f}%', ha='center', va='bottom', fontsize=8,
             color='#333333', fontweight='medium')

ax1.set_xlabel('MCA Dimension', fontsize=11, fontweight='medium')
ax1.set_ylabel('Inertia (%)', fontsize=11, fontweight='medium', color=CLR_BLUE)
ax1.set_xticks(x_pos)
ax1.set_xticklabels([f'{int(d)}' for d in dims], fontsize=10)
ax1.tick_params(axis='y', labelcolor=CLR_BLUE)
ax1.set_ylim(0, max(inertia_pct) * 1.25)
ax1.grid(axis='y', alpha=0.2, linestyle='-', color='#CCCCCC')
ax1.set_axisbelow(True)
ax1.spines['top'].set_visible(False)

# Second y-axis — cumulative inertia
ax2 = ax1.twinx()
ax2.plot(x_pos, cum_inertia, color=CLR_DKRED, marker='o', linewidth=2.0,
         markersize=7, markeredgecolor='white', markeredgewidth=0.5,
         label='Cumulative inertia (%)')
ax2.set_ylabel('Cumulative Inertia (%)', fontsize=11, fontweight='medium', color=CLR_DKRED)
ax2.tick_params(axis='y', labelcolor=CLR_DKRED)
ax2.set_ylim(0, 70)
ax2.spines['top'].set_visible(False)

# Reference line at 25% (Dim 1+2 used in biplot)
ax2.axhline(y=cum_inertia[1], color=CLR_DKRED, linestyle=':', linewidth=1.0, alpha=0.5)
ax2.text(len(dims) - 1.5, cum_inertia[1] + 1.5,
         f'Dims 1–2: {cum_inertia[1]:.1f}%\n(used in biplot)',
         fontsize=8.5, color=CLR_DKRED, ha='center')

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right',
           fontsize=9.5, framealpha=0.95, edgecolor='#CCCCCC')

plt.tight_layout()

# ═══════════════════════════════════════════════════════════════════
# SAVE — 600 DPI PNG + PDF
# ═══════════════════════════════════════════════════════════════════
fig.savefig(OUTPUT_PATH / 'Figure_S9_MCA_Scree_FINAL.png',
            dpi=600, bbox_inches='tight', facecolor='white')
fig.savefig(OUTPUT_PATH / 'Figure_S9_MCA_Scree_FINAL.pdf',
            bbox_inches='tight', facecolor='white')

for ext in ['png', 'pdf']:
    shutil.copy2(OUTPUT_PATH / f'Figure_S9_MCA_Scree_FINAL.{ext}',
                 DELIVER_PATH / f'Figure_S9_MCA_Scree_FINAL.{ext}')

print(f"\n✓ Saved Figure_S9_MCA_Scree_FINAL.png (600 DPI) and .pdf")
print(f"✓ Copied to 08_Peer_Review_Deliverables/02_Figures/")
plt.close()
