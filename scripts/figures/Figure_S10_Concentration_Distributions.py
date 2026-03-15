"""
Figure S10: Concentration Distributions of Detected PFAS
Box-and-whisker style summary (median, IQR, P95, max) with MCL reference lines.

Data source: concentration_distribution_summary.csv
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
df = pd.read_csv(DATA_PATH / 'concentration_distribution_summary.csv')

print("Concentration distribution summary:")
for _, row in df.iterrows():
    mcl_str = f"{row['MCL_ngL']:.0f}" if pd.notna(row['MCL_ngL']) else "None"
    print(f"  {row['Compound']}: n={int(row['N_Detected'])}, median={row['Median_ngL']:.1f}, "
          f"mean={row['Mean_ngL']:.1f}, P95={row['P95_ngL']:.1f}, max={row['Max_ngL']:.0f}, MCL={mcl_str}")

# ═══════════════════════════════════════════════════════════════════
# COLORS — matching template
# ═══════════════════════════════════════════════════════════════════
CLR_BLUE    = '#4682B4'
CLR_DKRED   = '#8B0000'
CLR_NPDWR   = '#E74C3C'
CLR_GREEN   = '#4CAF50'

# Regulated vs unregulated compounds
regulated = ['PFOS', 'PFOA', 'PFHxS', 'PFNA', 'HFPO-DA']
compound_colors = [CLR_DKRED if c in regulated else CLR_BLUE for c in df['Compound']]

# ═══════════════════════════════════════════════════════════════════
# CREATE FIGURE — Horizontal summary plot (log scale)
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 7))

compounds = df['Compound'].values
n_compounds = len(compounds)
y_pos = np.arange(n_compounds)

# Plot IQR as horizontal bars
for i, (_, row) in enumerate(df.iterrows()):
    color = compound_colors[i]
    p25 = row['P25_ngL']
    p75 = row['P75_ngL']
    median = row['Median_ngL']
    p95 = row['P95_ngL']
    maximum = row['Max_ngL']

    # IQR bar
    ax.barh(y_pos[i], p75 - p25, left=p25, height=0.5,
            color=color, alpha=0.6, edgecolor=color, linewidth=1.0)

    # Median marker
    ax.plot(median, y_pos[i], marker='|', color='white', markersize=15,
            markeredgewidth=2.5, zorder=5)

    # Whisker to P95
    ax.plot([p75, p95], [y_pos[i], y_pos[i]], color=color,
            linewidth=1.5, alpha=0.7)
    ax.plot(p95, y_pos[i], marker='|', color=color, markersize=8,
            markeredgewidth=1.5, alpha=0.7)

    # Max as diamond
    ax.plot(maximum, y_pos[i], marker='D', color=color, markersize=5,
            alpha=0.5, markeredgecolor=color)

    # n label
    ax.text(1.5, y_pos[i], f'n={int(row["N_Detected"]):,}',
            va='center', fontsize=8, color='#555555')

# MCL reference lines for regulated compounds
mcl_compounds_with_mcl = df[df['MCL_ngL'].notna()]
for _, row in mcl_compounds_with_mcl.iterrows():
    idx = list(compounds).index(row['Compound'])
    ax.plot(row['MCL_ngL'], y_pos[idx], marker='v', color=CLR_NPDWR,
            markersize=10, zorder=6, markeredgecolor='white', markeredgewidth=0.5)

ax.set_xscale('log')
ax.set_yticks(y_pos)
ax.set_yticklabels(compounds, fontsize=10, fontweight='medium')
ax.set_xlabel('Concentration (ng/L)', fontsize=11, fontweight='medium')
ax.set_xlim(1, 600)
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.2, linestyle='-', color='#CCCCCC')
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Legend
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
legend_elements = [
    Patch(facecolor=CLR_DKRED, alpha=0.6, label='Regulated (IQR)'),
    Patch(facecolor=CLR_BLUE, alpha=0.6, label='Unregulated (IQR)'),
    Line2D([0], [0], marker='|', color='white', markeredgecolor='gray',
           markersize=10, markeredgewidth=2, label='Median', linestyle='None'),
    Line2D([0], [0], marker='D', color='gray', markersize=5,
           label='Maximum', linestyle='None', alpha=0.5),
    Line2D([0], [0], marker='v', color=CLR_NPDWR, markersize=8,
           label='MCL', linestyle='None'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9,
          framealpha=0.95, edgecolor='#CCCCCC')

# % above MCL annotation — placed below each bar in blank space
for _, row in mcl_compounds_with_mcl.iterrows():
    idx = list(compounds).index(row['Compound'])
    pct = row['Pct_Above_MCL']
    # Place label at the P75 x-position, shifted down below the bar
    ax.text(row['P75_ngL'], y_pos[idx] + 0.32,
            f'{pct:.0f}% > MCL', va='top', ha='center', fontsize=8,
            color=CLR_NPDWR, fontweight='bold')

plt.tight_layout()

# ═══════════════════════════════════════════════════════════════════
# SAVE — 600 DPI PNG + PDF
# ═══════════════════════════════════════════════════════════════════
fig.savefig(OUTPUT_PATH / 'Figure_S10_Concentration_Distributions_FINAL.png',
            dpi=600, bbox_inches='tight', facecolor='white')
fig.savefig(OUTPUT_PATH / 'Figure_S10_Concentration_Distributions_FINAL.pdf',
            bbox_inches='tight', facecolor='white')

for ext in ['png', 'pdf']:
    shutil.copy2(OUTPUT_PATH / f'Figure_S10_Concentration_Distributions_FINAL.{ext}',
                 DELIVER_PATH / f'Figure_S10_Concentration_Distributions_FINAL.{ext}')

print(f"\n✓ Saved Figure_S10_Concentration_Distributions_FINAL.png (600 DPI) and .pdf")
print(f"✓ Copied to 08_Peer_Review_Deliverables/02_Figures/")
plt.close()
