"""
Figure S5: Temporal Trends in PFAS Monitoring (2023–2025)
Four-panel figure matching ORIGINAL template STYLE,
with Cochran-Armitage trend test statistic added (peer review requirement).

Data sources: temporal_detection_by_quarter.csv, temporal_detection_by_cohort.csv,
              temporal_mcl_quarterly.csv, temporal_cochran_armitage_enhanced.csv
"""

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats
from pathlib import Path

OUTPUT_PATH = Path("/sessions/kind-modest-dijkstra/mnt/5_PFAS_Research_Finalization/05_Figures")
DATA_PATH = Path("/sessions/kind-modest-dijkstra/mnt/5_PFAS_Research_Finalization/3_Jan2026_Update_Output_CSVs")

# ═══════════════════════════════════════════════════════════════════
# READ DATA
# ═══════════════════════════════════════════════════════════════════
quarterly_df = pd.read_csv(DATA_PATH / 'temporal_detection_by_quarter.csv')
cohort_df = pd.read_csv(DATA_PATH / 'temporal_detection_by_cohort.csv')
mcl_df = pd.read_csv(DATA_PATH / 'temporal_mcl_quarterly.csv')

# Cochran-Armitage test result
ca_df = pd.read_csv(DATA_PATH / 'temporal_cochran_armitage_enhanced.csv')
ca_z = ca_df['Z_Statistic'].values[0]
ca_p = ca_df['P_Value'].values[0]

quarters = quarterly_df['quarter'].values
n_quarters = len(quarters)
x_pos = np.arange(n_quarters)

# NPDWR finalized: between 2024Q1 (idx 4) and 2024Q2 (idx 5)
NPDWR_X = 4.5

print(f"Cochran-Armitage: Z = {ca_z:.3f}, p = {ca_p:.2e}")
print(f"\nPanel A - Samples per quarter:")
for q, n in zip(quarters, quarterly_df['N_Samples'].values):
    print(f"  {q}: {int(n):,}")
print(f"\nPanel B - Cumulative PWS:")
for q, n in zip(quarters, quarterly_df['Cumulative_PWS'].values):
    print(f"  {q}: {int(n):,}")
print(f"\nPanel C - Detection by cohort:")
for _, row in cohort_df.iterrows():
    print(f"  {row['entry_quarter']}: {row['Det_Rate']:.1f}% (n={int(row['N_Systems']):,})")
print(f"\nPanel D - MCL exceedance:")
for _, row in mcl_df.iterrows():
    print(f"  {row['quarter']}: {row['Exceedance_Rate_Pct']:.2f}%")

# ═══════════════════════════════════════════════════════════════════
# COLORS — matching original
# ═══════════════════════════════════════════════════════════════════
CLR_BLUE    = '#4682B4'
CLR_BLUE_LT = '#87CEEB'
CLR_GREEN   = '#4CAF50'
CLR_DKRED   = '#8B0000'
CLR_NPDWR   = '#E74C3C'

# ═══════════════════════════════════════════════════════════════════
# CREATE FIGURE — 2×2 grid
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# ─── PANEL A: Number of Samples per Quarter ─────────────────────
ax_a = axes[0, 0]

n_samples = quarterly_df['N_Samples'].values
ax_a.bar(x_pos, n_samples, color=CLR_BLUE, edgecolor='none', width=0.7)
ax_a.axvline(x=NPDWR_X, color=CLR_NPDWR, linestyle='--', linewidth=1.5, alpha=0.8)

ax_a.set_ylabel('Number of Samples', fontsize=11, fontweight='medium')
ax_a.set_xlabel('Quarter', fontsize=11, fontweight='medium')
ax_a.set_xticks(x_pos)
ax_a.set_xticklabels(quarters, rotation=45, ha='right', fontsize=9)
ax_a.set_ylim(0, max(n_samples) * 1.08)
ax_a.grid(axis='y', alpha=0.2, linestyle='-', color='#CCCCCC')
ax_a.set_axisbelow(True)
ax_a.spines['top'].set_visible(False)
ax_a.spines['right'].set_visible(False)
ax_a.text(-0.02, 1.03, '(A)', transform=ax_a.transAxes,
          fontsize=14, fontweight='bold', va='top')

# ─── PANEL B: Cumulative PWS Monitored ──────────────────────────
ax_b = axes[0, 1]

cum_pws = quarterly_df['Cumulative_PWS'].values
ax_b.fill_between(x_pos, cum_pws, alpha=0.3, color=CLR_BLUE_LT)
ax_b.plot(x_pos, cum_pws, color=CLR_BLUE, marker='o', linewidth=2.0,
          markersize=5, markeredgecolor='white', markeredgewidth=0.5)
ax_b.axvline(x=NPDWR_X, color=CLR_NPDWR, linestyle='--', linewidth=1.5, alpha=0.8)

ax_b.set_ylabel('Cumulative PWS Monitored', fontsize=11, fontweight='medium')
ax_b.set_xlabel('Quarter', fontsize=11, fontweight='medium')
ax_b.set_xticks(x_pos)
ax_b.set_xticklabels(quarters, rotation=45, ha='right', fontsize=9)
ax_b.set_ylim(0, max(cum_pws) * 1.1)
ax_b.grid(axis='y', alpha=0.2, linestyle='-', color='#CCCCCC')
ax_b.set_axisbelow(True)
ax_b.spines['top'].set_visible(False)
ax_b.spines['right'].set_visible(False)
ax_b.text(-0.02, 1.03, '(B)', transform=ax_b.transAxes,
          fontsize=14, fontweight='bold', va='top')

# ─── PANEL C: PWS-level Detection Rate by Entry Cohort ──────────
ax_c = axes[1, 0]

cohort_quarters = cohort_df['entry_quarter'].values
det_rates = cohort_df['Det_Rate'].values
cohort_ns = cohort_df['N_Systems'].values
n_cohorts = len(cohort_quarters)
x_c = np.arange(n_cohorts)

# Green bars
bars_c = ax_c.bar(x_c, det_rates, color=CLR_GREEN, edgecolor='none', width=0.7)

# Linear trend line
slope, intercept, r, p, se = scipy_stats.linregress(x_c, det_rates)
trend_y = slope * x_c + intercept
ax_c.plot(x_c, trend_y, color='gray', linestyle='--', linewidth=1.5,
          label=f'Linear trend (slope={slope:.2f}%/quarter)')

# NPDWR line
ax_c.axvline(x=NPDWR_X, color=CLR_NPDWR, linestyle='--', linewidth=1.5,
             alpha=0.8, label='NPDWR finalized')

# n=XXX labels above each bar
for i, (bar, n) in enumerate(zip(bars_c, cohort_ns)):
    ax_c.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
              f'n={int(n):,}', ha='center', va='bottom', fontsize=7.5,
              color='#333333')

# ─── COCHRAN-ARMITAGE TEST (PEER REVIEW REQUIREMENT) ────────────
ca_text = f'Cochran-Armitage trend test:\nZ = {ca_z:.3f}, p = {ca_p:.2e}'
props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9,
             edgecolor='#CCCCCC', linewidth=1.0)
ax_c.text(0.98, 0.55, ca_text, transform=ax_c.transAxes, fontsize=9,
          verticalalignment='top', horizontalalignment='right', bbox=props)

ax_c.set_ylabel('PWS-level detection rate (%)', fontsize=11, fontweight='medium')
ax_c.set_xlabel('Entry cohort (quarter)', fontsize=11, fontweight='medium')
ax_c.set_xticks(x_c)
ax_c.set_xticklabels(cohort_quarters, rotation=45, ha='right', fontsize=9)
ax_c.set_ylim(0, max(det_rates) * 1.15)
ax_c.grid(axis='y', alpha=0.2, linestyle='-', color='#CCCCCC')
ax_c.set_axisbelow(True)
ax_c.spines['top'].set_visible(False)
ax_c.spines['right'].set_visible(False)
ax_c.legend(loc='upper right', fontsize=8.5, framealpha=0.95, edgecolor='#CCCCCC')
ax_c.text(-0.02, 1.03, '(C)', transform=ax_c.transAxes,
          fontsize=14, fontweight='bold', va='top')

# ─── PANEL D: MCL Exceedance Rate per Quarter ───────────────────
ax_d = axes[1, 1]

mcl_quarters = mcl_df['quarter'].values
exc_rates = mcl_df['Exceedance_Rate_Pct'].values
x_d = np.arange(len(mcl_quarters))
mean_exc = np.mean(exc_rates)

ax_d.plot(x_d, exc_rates, color=CLR_DKRED, marker='o', linewidth=2.0,
          markersize=7, markeredgecolor='white', markeredgewidth=0.5,
          label='MCL Exceedance Rate')
ax_d.axhline(y=mean_exc, color='black', linestyle='-', linewidth=1.2,
             label=f'Mean ({mean_exc:.2f}%)')
ax_d.axvline(x=NPDWR_X, color=CLR_NPDWR, linestyle='--', linewidth=1.5,
             alpha=0.8, label='NPDWR finalized')

ax_d.set_ylabel('MCL exceedance rate (%)', fontsize=11, fontweight='medium')
ax_d.set_xlabel('Quarter', fontsize=11, fontweight='medium')
ax_d.set_xticks(x_d)
ax_d.set_xticklabels(mcl_quarters, rotation=45, ha='right', fontsize=9)
ax_d.grid(axis='y', alpha=0.2, linestyle='-', color='#CCCCCC')
ax_d.set_axisbelow(True)
ax_d.spines['top'].set_visible(False)
ax_d.spines['right'].set_visible(False)
ax_d.legend(loc='upper right', fontsize=8.5, framealpha=0.95, edgecolor='#CCCCCC')
ax_d.text(-0.02, 1.03, '(D)', transform=ax_d.transAxes,
          fontsize=14, fontweight='bold', va='top')

plt.tight_layout()

# ═══════════════════════════════════════════════════════════════════
# SAVE — 600 DPI PNG + PDF
# ═══════════════════════════════════════════════════════════════════
fig.savefig(OUTPUT_PATH / 'Figure_S5_Temporal_FINAL.png',
            dpi=600, bbox_inches='tight', facecolor='white')
fig.savefig(OUTPUT_PATH / 'Figure_S5_Temporal_FINAL.pdf',
            bbox_inches='tight', facecolor='white')
print("\nSaved Figure_S5_Temporal_FINAL.png (600 DPI) and .pdf")
plt.close()
