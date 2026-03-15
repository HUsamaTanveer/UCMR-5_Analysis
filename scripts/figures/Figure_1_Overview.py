"""
Regenerate Figure 1 — PFAS Detection and Exposure Overview
V2: Fine-tuned to match original template exactly
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from matplotlib.patches import Patch

# ─── Load data ────────────────────────────────────────────────────────────────

csvdir = '/sessions/jolly-relaxed-meitner/mnt/5_PFAS_Research_Finalization/3_Jan2026_Update_Output_CSVs/'
t1 = pd.read_csv(csvdir + 'core_statistics_table1_enhanced.csv')
geo = pd.read_csv(csvdir + 'core_statistics_geographic_enhanced.csv')
pop = pd.read_csv(csvdir + 'population_exposure.csv')

# ─── Colors matching original template ────────────────────────────────────────

DARK_RED   = '#8B1A1A'
STEEL_BLUE = '#4682B4'
ORANGE     = '#E8860C'
GOLD_OCHRE = '#C49B30'
TEAL       = '#2E8B7D'
TEAL_DARK  = '#2D6B5F'
LIGHT_BLUE = '#6FA8DC'
BROWN_GOLD = '#C47A1A'

# ─── Data prep ────────────────────────────────────────────────────────────────

# Panel A
det = t1[t1['pws_det_pct'] > 0.1].sort_values('pws_det_pct', ascending=True).copy()
regulated = {'PFOS', 'PFOA', 'PFHxS', 'PFNA', 'HFPO-DA'}
det['is_reg'] = det['Compound'].isin(regulated)

# Panel B
exc = t1[t1['pws_exceed_pct'].notna()].sort_values('pws_exceed_pct', ascending=True).copy()
mcl_map = {'PFOS': 0.004, 'PFOA': 0.004, 'PFHxS': 0.01, 'PFNA': 0.01, 'HFPO-DA': 0.01}
exc['mcl_label'] = exc['Compound'].map(lambda x: f"{x} (MCL={mcl_map.get(x, '')} µg/L)")

# Panel C
fips_to_abbr = {
    '01':'AL','02':'AK','04':'AZ','05':'AR','06':'CA','08':'CO','09':'CT','10':'DE',
    '11':'DC','12':'FL','13':'GA','15':'HI','16':'ID','17':'IL','18':'IN','19':'IA',
    '20':'KS','21':'KY','22':'LA','23':'ME','24':'MD','25':'MA','26':'MI','27':'MN',
    '28':'MS','29':'MO','30':'MT','31':'NE','32':'NV','33':'NH','34':'NJ','35':'NM',
    '36':'NY','37':'NC','38':'ND','39':'OH','40':'OK','41':'OR','42':'PA','44':'RI',
    '45':'SC','46':'SD','47':'TN','48':'TX','49':'UT','50':'VT','51':'VA','53':'WA',
    '54':'WV','55':'WI','56':'WY','60':'AS','66':'GU','69':'MP','72':'PR','78':'VI'
}
geo['Abbr'] = geo['State'].astype(str).map(fips_to_abbr).fillna(geo['State'])
geo_big = geo[geo['n_pws'] >= 50].copy()

nat_rate = 0.344
nat_n = 10297
def zscore_sig(row):
    p1 = row['detection_rate'] / 100
    p0 = nat_rate
    n1 = row['n_pws']
    se = np.sqrt(p0*(1-p0)*(1/n1 + 1/nat_n))
    z = (p1 - p0) / se if se > 0 else 0
    pval = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, pval

zscores = geo_big.apply(zscore_sig, axis=1, result_type='expand')
geo_big['z'] = zscores[0]
geo_big['pval'] = zscores[1]
reject, pval_corr, _, _ = multipletests(geo_big['pval'], alpha=0.05, method='fdr_bh')
geo_big['sig_above'] = (reject) & (geo_big['z'] > 0)

top15 = geo_big.nlargest(15, 'detection_rate').sort_values('detection_rate', ascending=True).copy()

# Panel D
cascade_labels = [
    'All monitored\n(10,297 systems)',
    'Any PFAS\ndetected',
    'Any regulated\nPFAS detected',
    'MCL\nexceedance',
    'Both PFOS &\nPFOA exceed',
    'Exceedance,\nno treatment'
]
cascade_pops = [302374366, 151355790, 87925860, 80108420, 47702483, 67487984]
cascade_pcts = [100, 50, 29, 26, 16, 22]
CASCADE_COLORS = [STEEL_BLUE, LIGHT_BLUE, TEAL, TEAL_DARK, BROWN_GOLD, BROWN_GOLD]

# ─── BUILD FIGURE ─────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(16, 13))
gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.32,
                       left=0.08, right=0.97, top=0.96, bottom=0.05)

# ═══════════ PANEL A ═══════════════════════════════════════════════════════════
ax = fig.add_subplot(gs[0, 0])
colors_a = [DARK_RED if r else STEEL_BLUE for r in det['is_reg']]
bars_a = ax.barh(range(len(det)), det['pws_det_pct'], color=colors_a, height=0.72, edgecolor='none')

# CIs (peer review addition)
ci_lo = det['det_ci_lo'].values
ci_hi = det['det_ci_hi'].values
ax.errorbar(det['pws_det_pct'].values, range(len(det)),
            xerr=[det['pws_det_pct'].values - ci_lo, ci_hi - det['pws_det_pct'].values],
            fmt='none', color='#444444', capsize=2, linewidth=0.7, capthick=0.7)

# Percentage labels at end of bars
for i, (val, name) in enumerate(zip(det['pws_det_pct'], det['Compound'])):
    ax.text(val + 0.4, i, f'{val:.1f}%', va='center', fontsize=8.5, fontweight='bold', color='#333333')

ax.set_yticks(range(len(det)))
ax.set_yticklabels(det['Compound'], fontsize=9.5)
ax.set_xlabel('PWS detection rate (%)', fontsize=10.5)
ax.set_xlim(0, 24)
ax.text(-0.08, 1.02, 'A', transform=ax.transAxes, fontsize=18, fontweight='bold', va='bottom')

legend_a = [Patch(facecolor=DARK_RED, label='Regulated (MCL)'),
            Patch(facecolor=STEEL_BLUE, label='Unregulated')]
ax.legend(handles=legend_a, loc='lower right', fontsize=8.5, framealpha=0.95,
          edgecolor='#cccccc')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', labelsize=9)

# ═══════════ PANEL B ═══════════════════════════════════════════════════════════
ax = fig.add_subplot(gs[0, 1])

# Colors per compound
color_map_b = {'PFOS': DARK_RED, 'PFOA': DARK_RED, 'PFHxS': GOLD_OCHRE, 'PFNA': '#999999', 'HFPO-DA': '#999999'}
colors_b = [color_map_b[c] for c in exc['Compound']]

# Wider bar spacing for 5 items
y_b = np.arange(len(exc)) * 1.5
bars_b = ax.barh(y_b, exc['pws_exceed_pct'], color=colors_b, height=1.0, edgecolor='none')

# CIs
ci_lo_b = exc['exceed_ci_lo'].values
ci_hi_b = exc['exceed_ci_hi'].values
ax.errorbar(exc['pws_exceed_pct'].values, y_b,
            xerr=[exc['pws_exceed_pct'].values - ci_lo_b, ci_hi_b - exc['pws_exceed_pct'].values],
            fmt='none', color='#444444', capsize=2.5, linewidth=0.7, capthick=0.7)

# Labels: inside bar (white) for large bars, outside for small
for i, (val, ypos, comp) in enumerate(zip(exc['pws_exceed_pct'], y_b, exc['Compound'])):
    label = f'{val:.1f}%' if val >= 1 else f'{val:.2f}%'
    if val > 5:
        # Inside bar, right-aligned, white text
        ax.text(val - 0.3, ypos, label, va='center', ha='right',
                fontsize=9, fontweight='bold', color='white')
    else:
        # Outside bar
        ax.text(val + 0.25, ypos, label, va='center', ha='left',
                fontsize=8.5, fontweight='bold', color='#333333')

ax.set_yticks(y_b)
ax.set_yticklabels(exc['mcl_label'], fontsize=9.5)
ax.set_xlabel('PWS exceeding MCL (%)', fontsize=10.5)
ax.set_xlim(0, 14)
ax.set_ylim(-0.8, y_b[-1] + 1.2)
ax.text(-0.08, 1.02, 'B', transform=ax.transAxes, fontsize=18, fontweight='bold', va='bottom')

# Annotation box — lower right like original
ax.annotate('1,717 systems (16.7%)\nexceed ≥ 1 MCL',
            xy=(11.5, y_b[0] - 0.3), fontsize=8.5, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#F5E6C8', edgecolor='#C49B30', linewidth=1.0))

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', labelsize=9)

# ═══════════ PANEL C ═══════════════════════════════════════════════════════════
ax = fig.add_subplot(gs[1, 0])
colors_c = [ORANGE if sig else STEEL_BLUE for sig in top15['sig_above']]
ylabels_c = [f"{row['Abbr']} (n={int(row['n_pws'])})" for _, row in top15.iterrows()]

bars_c = ax.barh(range(len(top15)), top15['detection_rate'], color=colors_c, height=0.68, edgecolor='none')

# CIs
ax.errorbar(top15['detection_rate'].values, range(len(top15)),
            xerr=[top15['detection_rate'].values - top15['ci_lo'].values,
                  top15['ci_hi'].values - top15['detection_rate'].values],
            fmt='none', color='#444444', capsize=2, linewidth=0.7, capthick=0.7)

# National average dashed line
ax.axvline(x=34.4, color='#333333', linestyle='--', linewidth=1.0, zorder=1)
ax.text(35.5, -1.0, 'National avg (34.4%)', fontsize=8, ha='left', va='center',
        color='#333333', style='italic')

ax.set_yticks(range(len(top15)))
ax.set_yticklabels(ylabels_c, fontsize=9.5)
ax.set_xlabel('PWS detection rate (%)', fontsize=10.5)
ax.set_xlim(0, 75)
ax.text(-0.08, 1.02, 'C', transform=ax.transAxes, fontsize=18, fontweight='bold', va='bottom')

legend_c = [Patch(facecolor=ORANGE, label='Sig. above average'),
            Patch(facecolor=STEEL_BLUE, label='Not significant')]
ax.legend(handles=legend_c, loc='lower right', fontsize=8.5, framealpha=0.95,
          edgecolor='#cccccc')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', labelsize=9)

# ═══════════ PANEL D ═══════════════════════════════════════════════════════════
ax = fig.add_subplot(gs[1, 1])
y_d = list(range(len(cascade_labels)-1, -1, -1))

bars_d = ax.barh(y_d, [p/1e6 for p in cascade_pops], color=CASCADE_COLORS, height=0.62, edgecolor='none')

# Population labels at end of bars
for pos, val, pct in zip(y_d, cascade_pops, cascade_pcts):
    val_m = val / 1e6
    ax.text(val_m + 4, pos, f'{int(val_m)}M ({pct}%)', va='center',
            fontsize=9, fontweight='bold', color='#333333')

ax.set_yticks(y_d)
ax.set_yticklabels(cascade_labels, fontsize=9.5)
ax.set_xlabel('Population served (millions)', fontsize=10.5)
ax.set_xlim(0, 350)
ax.text(-0.08, 1.02, 'D', transform=ax.transAxes, fontsize=18, fontweight='bold', va='bottom')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', labelsize=9)

# ─── Save ─────────────────────────────────────────────────────────────────────

outdir = '/sessions/jolly-relaxed-meitner/mnt/5_PFAS_Research_Finalization/05_Figures/1_Final Figures/'
fig.savefig(outdir + 'Figure_1_Overview_FINAL.png', dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(outdir + 'Figure_1_Overview_FINAL.pdf', dpi=300, bbox_inches='tight', facecolor='white')
print("Saved Figure_1_Overview_FINAL.png and .pdf")
plt.close()
