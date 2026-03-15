"""
Figure S2: State-level PFAS Detection Z-Scores
Horizontal bar chart matching ORIGINAL template STYLE,
with Benjamini-Hochberg FDR correction added (peer review requirement).

Data source: population_exposure_by_state.csv (PWS-level detection rates)
National average: 34.4% (3,539 / 10,297 PWS)
"""

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy import stats as scipy_stats
from pathlib import Path

OUTPUT_PATH = Path("/sessions/jolly-relaxed-meitner/mnt/5_PFAS_Research_Finalization/05_Figures")
DATA_PATH = Path("/sessions/jolly-relaxed-meitner/mnt/5_PFAS_Research_Finalization/3_Jan2026_Update_Output_CSVs")

# ═══════════════════════════════════════════════════════════════════
# FIPS → state abbreviation mapping (same as original template)
# ═══════════════════════════════════════════════════════════════════
fips_to_abbr = {
    '01': 'AL', '02': 'AK', '04': 'AZ', '05': 'AR', '06': 'CA', '08': 'CO',
    '09': 'CT', '10': 'DE', '11': 'DC', '12': 'FL', '13': 'GA', '15': 'HI',
    '16': 'ID', '17': 'IL', '18': 'IN', '19': 'IA', '20': 'KS', '21': 'KY',
    '22': 'LA', '23': 'ME', '24': 'MD', '25': 'MA', '26': 'MI', '27': 'MN',
    '28': 'MS', '29': 'MO', '30': 'MT', '31': 'NE', '32': 'NV', '33': 'NH',
    '34': 'NJ', '35': 'NM', '36': 'NY', '37': 'NC', '38': 'ND', '39': 'OH',
    '40': 'OK', '41': 'OR', '42': 'PA', '44': 'RI', '45': 'SC', '46': 'SD',
    '47': 'TN', '48': 'TX', '49': 'UT', '50': 'VT', '51': 'VA', '53': 'WA',
    '54': 'WV', '55': 'WI', '56': 'WY', '72': 'PR', '07': 'PR'
}

# ═══════════════════════════════════════════════════════════════════
# READ DATA — population_exposure_by_state.csv (PWS-level)
# ═══════════════════════════════════════════════════════════════════
df = pd.read_csv(DATA_PATH / 'population_exposure_by_state.csv')

# Handle mixed state column (some FIPS codes, some abbreviations)
def resolve_state(val):
    s = str(val).strip()
    if s.isalpha() and len(s) == 2:
        return s.upper()
    return fips_to_abbr.get(s.zfill(2), None)

df['state_abbr'] = df['state'].apply(resolve_state)
df = df.dropna(subset=['state_abbr'])

# Aggregate by state abbreviation (handles FIPS + abbreviation duplicates)
df_agg = df.groupby('state_abbr').agg(
    total_pws=('total_pws', 'sum'),
    detect_pws=('detect_pws', 'sum')
).reset_index()

# Filter: minimum 50 PWS (corrected to match manuscript methods section)
# Previous threshold was ≥30; updated Feb 2026 to align with manuscript §2.5
df_agg = df_agg[df_agg['total_pws'] >= 50].copy()

# ═══════════════════════════════════════════════════════════════════
# CALCULATE Z-SCORES
# ═══════════════════════════════════════════════════════════════════
NATIONAL_AVG = 0.344

def calc_zscore(detected, total, p0):
    if total == 0:
        return 0.0
    rate = detected / total
    se = np.sqrt(p0 * (1 - p0) / total)
    if se == 0:
        return 0.0
    return (rate - p0) / se

df_agg['z_score'] = df_agg.apply(
    lambda r: calc_zscore(r['detect_pws'], r['total_pws'], NATIONAL_AVG), axis=1)

# ═══════════════════════════════════════════════════════════════════
# BENJAMINI-HOCHBERG FDR CORRECTION (peer review requirement)
# ═══════════════════════════════════════════════════════════════════
# Two-tailed p-values from z-scores
df_agg['p_value'] = 2 * (1 - scipy_stats.norm.cdf(np.abs(df_agg['z_score'])))

# BH correction: sort by p-value, adjusted_p = p * m / rank
m = len(df_agg)
df_sorted = df_agg.sort_values('p_value').reset_index(drop=True)
df_sorted['rank'] = np.arange(1, m + 1)
df_sorted['bh_adjusted_p'] = df_sorted['p_value'] * m / df_sorted['rank']

# Enforce monotonicity (cumulative minimum from bottom)
df_sorted['bh_adjusted_p'] = df_sorted['bh_adjusted_p'][::-1].cummin()[::-1]
df_sorted['bh_adjusted_p'] = df_sorted['bh_adjusted_p'].clip(upper=1.0)

# Map back to original dataframe
bh_map = dict(zip(df_sorted['state_abbr'], df_sorted['bh_adjusted_p']))
df_agg['bh_adjusted_p'] = df_agg['state_abbr'].map(bh_map)
df_agg['fdr_significant'] = df_agg['bh_adjusted_p'] < 0.05

# Count how many are FDR-significant
n_fdr_sig = df_agg['fdr_significant'].sum()
n_above = ((df_agg['z_score'] > 0) & df_agg['fdr_significant']).sum()
n_below = ((df_agg['z_score'] < 0) & df_agg['fdr_significant']).sum()
n_ns = (~df_agg['fdr_significant']).sum()

print(f"Total states/territories: {m}")
print(f"FDR-significant above: {n_above}")
print(f"FDR-significant below: {n_below}")
print(f"Not significant after FDR: {n_ns}")

# ═══════════════════════════════════════════════════════════════════
# COLOR CODING — matching original palette, but using FDR threshold
# ═══════════════════════════════════════════════════════════════════
CLR_ABOVE = '#E8751A'   # orange for significantly above
CLR_BELOW = '#009688'   # teal for significantly below
CLR_NS    = '#B8B8B8'   # gray for not significant

def get_color(row):
    if row['fdr_significant'] and row['z_score'] > 0:
        return CLR_ABOVE
    elif row['fdr_significant'] and row['z_score'] < 0:
        return CLR_BELOW
    else:
        return CLR_NS

df_agg['color'] = df_agg.apply(get_color, axis=1)

# Sort by z-score ascending (lowest at bottom, highest at top)
df_agg = df_agg.sort_values('z_score', ascending=True).reset_index(drop=True)

# Print verification table
print("\nState  |    n  | Rate   | Z-score |   Raw p  |  BH adj p | FDR sig")
print("-" * 75)
for _, row in df_agg.iterrows():
    rate = row['detect_pws'] / row['total_pws'] * 100
    sig = "YES" if row['fdr_significant'] else "no"
    print(f"{row['state_abbr']:>5s}  | {int(row['total_pws']):>5d} | {rate:5.1f}% | {row['z_score']:+7.2f} | {row['p_value']:.2e} | {row['bh_adjusted_p']:.4f}    | {sig}")

# ═══════════════════════════════════════════════════════════════════
# CREATE FIGURE — matching original template style
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 16))

# Y-axis labels: "ST (n=XXX)"
y_labels = [f"{row['state_abbr']} (n={int(row['total_pws'])})"
            for _, row in df_agg.iterrows()]
y_pos = np.arange(len(df_agg))

# Horizontal bars (no black edge — matching original)
ax.barh(y_pos, df_agg['z_score'].values, color=df_agg['color'].values,
        edgecolor='none', height=0.75)

# Reference lines
ax.axvline(x=0, color='black', linestyle='-', linewidth=1.0, alpha=0.6)
ax.axvline(x=1.96, color='black', linestyle=':', linewidth=0.8, alpha=0.4)
ax.axvline(x=-1.96, color='black', linestyle=':', linewidth=0.8, alpha=0.4)

# Axis labels
ax.set_yticks(y_pos)
ax.set_yticklabels(y_labels, fontsize=9)
ax.set_xlabel('Z-score (deviation from national average)', fontsize=12,
              fontweight='medium')
ax.set_title('State-level PFAS detection rates vs. national average (34.4%)',
             fontsize=13, fontweight='bold', pad=15)

# Light grid
ax.grid(axis='x', alpha=0.2, linestyle='-', linewidth=0.5, color='#CCCCCC')
ax.set_axisbelow(True)

# Clean spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ─── Legend — UPDATED with FDR clarification (peer review requirement) ───
legend_elements = [
    mpatches.Patch(facecolor=CLR_ABOVE,
                   label='Significantly above (BH-adjusted p < 0.05)'),
    mpatches.Patch(facecolor=CLR_BELOW,
                   label='Significantly below (BH-adjusted p < 0.05)'),
    mpatches.Patch(facecolor=CLR_NS,
                   label='Not significant after FDR correction'),
    plt.Line2D([0], [0], color='black', linestyle=':', linewidth=0.8, alpha=0.4,
               label='Unadjusted p = 0.05 threshold (z = ±1.96)'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9.5,
          framealpha=0.95, edgecolor='#CCCCCC', fancybox=True)

plt.tight_layout()

# ═══════════════════════════════════════════════════════════════════
# SAVE — 600 DPI PNG + PDF
# ═══════════════════════════════════════════════════════════════════
fig.savefig(OUTPUT_PATH / 'Figure_S2_State_ZScores_FINAL.png',
            dpi=600, bbox_inches='tight', facecolor='white')
fig.savefig(OUTPUT_PATH / 'Figure_S2_State_ZScores_FINAL.pdf',
            bbox_inches='tight', facecolor='white')
print("\nSaved Figure_S2_State_ZScores_FINAL.png (600 DPI) and .pdf")
plt.close()
