"""
Figure S12: UCMR 3 vs UCMR 5 Detection Rates (Matched-System Comparison)
Two-panel: (A) Grouped bars of UCMR 3 vs UCMR 5 MRL-adjusted detection rates,
           (B) Concordance matrix (stacked bars: both detected, UCMR3 only, UCMR5 only, neither).

Data source: ucmr3_comparison_mcnemar_enhanced.csv
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
df = pd.read_csv(DATA_PATH / 'ucmr3_comparison_mcnemar_enhanced.csv')

print("UCMR 3 vs UCMR 5 comparison:")
for _, row in df.iterrows():
    sig = "***" if row['McNemar_P_FDR_Adjusted'] < 0.001 else (
          "**" if row['McNemar_P_FDR_Adjusted'] < 0.01 else (
          "*" if row['McNemar_P_FDR_Adjusted'] < 0.05 else "ns"))
    print(f"  {row['Compound']}: UCMR3={row['UCMR3_Rate_Pct']:.2f}%, "
          f"UCMR5={row['UCMR5_Rate_Pct_MRL_Adj']:.2f}%, "
          f"McNemar p(FDR)={row['McNemar_P_FDR_Adjusted']:.2e} {sig}")

# ═══════════════════════════════════════════════════════════════════
# COLORS — matching template
# ═══════════════════════════════════════════════════════════════════
CLR_BLUE    = '#4682B4'
CLR_ORANGE  = '#E8751A'
CLR_GREEN   = '#4CAF50'
CLR_DKRED   = '#8B0000'
CLR_GRAY    = '#B8B8B8'

# ═══════════════════════════════════════════════════════════════════
# CREATE FIGURE — 1×2 grid
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={'width_ratios': [1.2, 1]})

compounds = df['Compound'].values
n_compounds = len(compounds)
x_pos = np.arange(n_compounds)

# ─── PANEL A: Grouped bars of detection rates ─────────────────────
ax_a = axes[0]
bar_width = 0.35

rates_3 = df['UCMR3_Rate_Pct'].values
rates_5 = df['UCMR5_Rate_Pct_MRL_Adj'].values

bars_3 = ax_a.bar(x_pos - bar_width/2, rates_3, bar_width,
                   color=CLR_BLUE, edgecolor='white', linewidth=0.8,
                   alpha=0.85, label='UCMR 3')
bars_5 = ax_a.bar(x_pos + bar_width/2, rates_5, bar_width,
                   color=CLR_ORANGE, edgecolor='white', linewidth=0.8,
                   alpha=0.85, label='UCMR 5 (MRL-adjusted)')

# Significance stars
for i, row in df.iterrows():
    p_fdr = row['McNemar_P_FDR_Adjusted']
    if p_fdr < 0.001:
        star = '***'
    elif p_fdr < 0.01:
        star = '**'
    elif p_fdr < 0.05:
        star = '*'
    else:
        star = 'ns'
    y_max = max(rates_3[i], rates_5[i])
    ax_a.text(x_pos[i], y_max + 0.08, star, ha='center', fontsize=11,
              fontweight='bold', color='#333333')

# Rate labels on bars
for i in range(n_compounds):
    ax_a.text(x_pos[i] - bar_width/2, rates_3[i] + 0.02,
              f'{rates_3[i]:.1f}%', ha='center', va='bottom', fontsize=7.5,
              color='#333333')
    ax_a.text(x_pos[i] + bar_width/2, rates_5[i] + 0.02,
              f'{rates_5[i]:.2f}%', ha='center', va='bottom', fontsize=7.5,
              color='#333333')

ax_a.set_xticks(x_pos)
ax_a.set_xticklabels(compounds, rotation=45, ha='right', fontsize=10)
ax_a.set_ylabel('Detection Rate (%)', fontsize=11, fontweight='medium')
ax_a.set_xlabel('PFAS Compound', fontsize=11, fontweight='medium')
ax_a.set_ylim(0, max(rates_3) * 1.4)
ax_a.legend(loc='upper right', fontsize=9.5, framealpha=0.95, edgecolor='#CCCCCC')
ax_a.grid(axis='y', alpha=0.2, linestyle='-', color='#CCCCCC')
ax_a.set_axisbelow(True)
ax_a.spines['top'].set_visible(False)
ax_a.spines['right'].set_visible(False)

# Significance threshold box
sig_text = '* p < 0.05  ** p < 0.01  *** p < 0.001\n(BH-adjusted McNemar test)'
props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9,
             edgecolor='#CCCCCC', linewidth=1.0)
ax_a.text(0.02, 0.97, sig_text, transform=ax_a.transAxes, fontsize=8.5,
          verticalalignment='top', horizontalalignment='left', bbox=props)

ax_a.text(-0.02, 1.03, '(A)', transform=ax_a.transAxes,
          fontsize=14, fontweight='bold', va='top')

# ─── PANEL B: Concordance stacked bars ────────────────────────────
ax_b = axes[1]

both_det = df['Concordance_Both_Det'].values
only_3 = df['Concordance_Only_UCMR3'].values
only_5 = df['Concordance_Only_UCMR5'].values
neither = df['Concordance_Neither'].values
total = df['Systems_Tested_Both_Rounds'].values

# Convert to percentages
both_pct = both_det / total * 100
only3_pct = only_3 / total * 100
only5_pct = only_5 / total * 100
neither_pct = neither / total * 100

# Stacked horizontal bars
ax_b.barh(x_pos, both_pct, height=0.6, color=CLR_GREEN, alpha=0.8,
          label='Both detected', edgecolor='white', linewidth=0.5)
ax_b.barh(x_pos, only3_pct, height=0.6, left=both_pct,
          color=CLR_BLUE, alpha=0.8, label='UCMR 3 only',
          edgecolor='white', linewidth=0.5)
ax_b.barh(x_pos, only5_pct, height=0.6, left=both_pct + only3_pct,
          color=CLR_ORANGE, alpha=0.8, label='UCMR 5 only',
          edgecolor='white', linewidth=0.5)
# Neither is too large to show; use rest of bar
# Only show the detection portions (zoom in)

ax_b.set_yticks(x_pos)
ax_b.set_yticklabels(compounds, fontsize=10)
ax_b.set_xlabel('Systems (%)', fontsize=11, fontweight='medium')
ax_b.set_xlim(0, 5)  # Zoom to detection portion
ax_b.invert_yaxis()
ax_b.legend(loc='lower right', fontsize=8.5, framealpha=0.95, edgecolor='#CCCCCC')
ax_b.grid(axis='x', alpha=0.2, linestyle='-', color='#CCCCCC')
ax_b.set_axisbelow(True)
ax_b.spines['top'].set_visible(False)
ax_b.spines['right'].set_visible(False)

# Concordance count labels
for i in range(n_compounds):
    total_det = both_det[i] + only_3[i] + only_5[i]
    ax_b.text(both_pct[i] + only3_pct[i] + only5_pct[i] + 0.05, x_pos[i],
              f'{int(total_det)} det.', va='center', fontsize=7.5,
              color='#555555')

ax_b.text(-0.02, 1.03, '(B)', transform=ax_b.transAxes,
          fontsize=14, fontweight='bold', va='top')

plt.tight_layout()

# ═══════════════════════════════════════════════════════════════════
# SAVE — 600 DPI PNG + PDF
# ═══════════════════════════════════════════════════════════════════
fig.savefig(OUTPUT_PATH / 'Figure_S12_UCMR3_McNemar_FINAL.png',
            dpi=600, bbox_inches='tight', facecolor='white')
fig.savefig(OUTPUT_PATH / 'Figure_S12_UCMR3_McNemar_FINAL.pdf',
            bbox_inches='tight', facecolor='white')

for ext in ['png', 'pdf']:
    shutil.copy2(OUTPUT_PATH / f'Figure_S12_UCMR3_McNemar_FINAL.{ext}',
                 DELIVER_PATH / f'Figure_S12_UCMR3_McNemar_FINAL.{ext}')

print(f"\n✓ Saved Figure_S12_UCMR3_McNemar_FINAL.png (600 DPI) and .pdf")
print(f"✓ Copied to 08_Peer_Review_Deliverables/02_Figures/")
plt.close()
