"""
Figure S7: Logistic Regression Calibration Plot (Hosmer-Lemeshow)
Single-panel: observed proportion vs mean predicted probability by decile,
with perfect calibration line and H-L test statistic.

Data sources: regression_calibration_data.csv, regression_diagnostics_enhanced.csv
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
cal_df = pd.read_csv(DATA_PATH / 'regression_calibration_data.csv')
diag_df = pd.read_csv(DATA_PATH / 'regression_diagnostics_enhanced.csv')

diag = dict(zip(diag_df['Metric'], diag_df['Value']))
hl_stat = diag['HL_Statistic']   # 27.69
hl_p = diag['HL_PValue']         # 0.0005
auc = diag['AUC']                # 0.7004

print("Calibration data:")
for _, row in cal_df.iterrows():
    print(f"  Decile {int(row['decile'])}: pred={row['mean_pred_prob']:.4f}, obs={row['obs_proportion']:.4f} (n={int(row['count'])})")
print(f"\nHL stat={hl_stat:.2f}, p={hl_p:.4f}, AUC={auc:.4f}")

# ═══════════════════════════════════════════════════════════════════
# COLORS — matching template
# ═══════════════════════════════════════════════════════════════════
CLR_BLUE    = '#4682B4'
CLR_DKRED   = '#8B0000'

# ═══════════════════════════════════════════════════════════════════
# CREATE FIGURE
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 8))

pred = cal_df['mean_pred_prob'].values
obs = cal_df['obs_proportion'].values

# Perfect calibration line
ax.plot([0, 0.8], [0, 0.8], color='gray', linestyle='--', linewidth=1.5,
        alpha=0.7, label='Perfect calibration', zorder=1)

# Observed vs predicted scatter with connecting line
ax.plot(pred, obs, color=CLR_BLUE, marker='o', linewidth=2.0,
        markersize=10, markeredgecolor='white', markeredgewidth=1.0,
        label='Observed proportions', zorder=3)

# Label each decile
for i, (p, o) in enumerate(zip(pred, obs)):
    offset_x = 0.015
    offset_y = -0.015 if o < p else 0.015
    ax.annotate(f'D{i+1}', (p, o), fontsize=8, color='#333333',
                textcoords='offset points', xytext=(8, 4),
                fontweight='medium')

# H-L test annotation
hl_text = (f'Hosmer-Lemeshow test:\n'
           f'χ² = {hl_stat:.2f}, p = {hl_p:.4f}\n'
           f'AUC = {auc:.4f}')
props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9,
             edgecolor='#CCCCCC', linewidth=1.0)
ax.text(0.03, 0.97, hl_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='left', bbox=props)

ax.set_xlabel('Mean Predicted Probability', fontsize=11, fontweight='medium')
ax.set_ylabel('Observed Proportion', fontsize=11, fontweight='medium')
ax.set_xlim(0, 0.75)
ax.set_ylim(0, 0.75)
ax.set_aspect('equal')
ax.legend(loc='lower right', fontsize=9.5, framealpha=0.95, edgecolor='#CCCCCC')
ax.grid(alpha=0.2, linestyle='-', color='#CCCCCC')
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

# ═══════════════════════════════════════════════════════════════════
# SAVE — 600 DPI PNG + PDF
# ═══════════════════════════════════════════════════════════════════
fig.savefig(OUTPUT_PATH / 'Figure_S7_Calibration_Plot_FINAL.png',
            dpi=600, bbox_inches='tight', facecolor='white')
fig.savefig(OUTPUT_PATH / 'Figure_S7_Calibration_Plot_FINAL.pdf',
            bbox_inches='tight', facecolor='white')

for ext in ['png', 'pdf']:
    shutil.copy2(OUTPUT_PATH / f'Figure_S7_Calibration_Plot_FINAL.{ext}',
                 DELIVER_PATH / f'Figure_S7_Calibration_Plot_FINAL.{ext}')

print(f"\n✓ Saved Figure_S7_Calibration_Plot_FINAL.png (600 DPI) and .pdf")
print(f"✓ Copied to 08_Peer_Review_Deliverables/02_Figures/")
plt.close()
