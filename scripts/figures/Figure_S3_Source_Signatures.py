"""
Figure S3: Source Signatures — Detection rates by compound across source types
Grouped bar chart matching ORIGINAL template STYLE,
with significance threshold definition added (peer review requirement).

Data source: source_attribution_verification.csv
Statistics: source_association_enhanced.csv (chi-square, Cramér's V)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUTPUT_PATH = Path("/sessions/kind-modest-dijkstra/mnt/5_PFAS_Research_Finalization/05_Figures")

# ═══════════════════════════════════════════════════════════════════
# DATA — from source_attribution_verification.csv (Jan 2026)
# Using exact decimal values from CSV
# ═══════════════════════════════════════════════════════════════════
sources = ['Airport Operations', 'Military Base', 'Utilities/Sewage',
           'Manufacturing', 'Fire Training/Services']
source_ns = [286, 240, 266, 236, 223]

# Detection rates (%) — exact from CSV
pfos_rates  = [22.0, 32.1, 20.3, 22.0, 25.6]
pfoa_rates  = [19.9, 26.7, 18.4, 28.8, 18.4]
pfhxs_rates = [21.7, 32.9, 13.2, 15.7, 22.9]
pfbs_rates  = [25.5, 31.7, 24.4, 28.0, 25.6]

# Wilson 95% CI bounds from CSV — compute error bar half-widths
pfos_ci_lo  = [17.6, 26.5, 15.9, 17.2, 20.3]
pfos_ci_hi  = [27.2, 38.2, 25.5, 27.7, 31.7]
pfoa_ci_lo  = [15.7, 21.5, 14.2, 23.4, 13.9]
pfoa_ci_hi  = [24.9, 32.6, 23.5, 34.9, 24.0]
pfhxs_ci_lo = [17.3, 27.3, 9.6, 11.6, 17.8]
pfhxs_ci_hi = [26.8, 39.1, 17.7, 20.9, 28.8]
pfbs_ci_lo  = [20.8, 26.1, 19.7, 22.6, 20.3]
pfbs_ci_hi  = [30.9, 37.8, 29.9, 34.0, 31.7]

def asymmetric_errors(rates, ci_lo, ci_hi):
    """Return [lower_errors, upper_errors] for matplotlib errorbar."""
    lower = [r - lo for r, lo in zip(rates, ci_lo)]
    upper = [hi - r for r, hi in zip(rates, ci_hi)]
    return [lower, upper]

err_pfos  = asymmetric_errors(pfos_rates, pfos_ci_lo, pfos_ci_hi)
err_pfoa  = asymmetric_errors(pfoa_rates, pfoa_ci_lo, pfoa_ci_hi)
err_pfhxs = asymmetric_errors(pfhxs_rates, pfhxs_ci_lo, pfhxs_ci_hi)
err_pfbs  = asymmetric_errors(pfbs_rates, pfbs_ci_lo, pfbs_ci_hi)

# Statistics from source_association_enhanced.csv
chi2_stat = 163.03
p_value_str = 'p < 0.001'
cramers_v = 0.009

# ═══════════════════════════════════════════════════════════════════
# COLORS — matching original exactly
# ═══════════════════════════════════════════════════════════════════
CLR_PFOS  = '#4682B4'   # steel blue
CLR_PFOA  = '#E8751A'   # orange
CLR_PFHXS = '#2E8B57'   # sea green
CLR_PFBS  = '#9467BD'   # purple

# ═══════════════════════════════════════════════════════════════════
# CREATE FIGURE — matching original template
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(15, 7.5))

x = np.arange(len(sources))
width = 0.18

# Grouped bars with Wilson CI error bars
bar_kw = dict(capsize=4, error_kw={'linewidth': 1.0, 'color': '#555555'},
              edgecolor='none')

ax.bar(x - 1.5*width, pfos_rates, width, label='PFOS', color=CLR_PFOS,
       yerr=err_pfos, **bar_kw)
ax.bar(x - 0.5*width, pfoa_rates, width, label='PFOA', color=CLR_PFOA,
       yerr=err_pfoa, **bar_kw)
ax.bar(x + 0.5*width, pfhxs_rates, width, label='PFHxS', color=CLR_PFHXS,
       yerr=err_pfhxs, **bar_kw)
ax.bar(x + 1.5*width, pfbs_rates, width, label='PFBS', color=CLR_PFBS,
       yerr=err_pfbs, **bar_kw)

# ─── X-axis: category name WITH n count underneath ──────────────
x_labels = [f'{src}\nn={n}' for src, n in zip(sources, source_ns)]
ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=11, ha='center')

# ─── PFHxS annotation for Military Base ─────────────────────────
ax.annotate('PFHxS: 32.9% ***',
            xy=(1 + 0.5*width, 33), xytext=(1.8, 45),
            fontsize=11, fontweight='bold', color='#2E8B57',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#2E8B57', linewidth=1.2),
            arrowprops=dict(arrowstyle='->', color='#2E8B57', lw=1.8,
                            connectionstyle='arc3,rad=-0.2'))

# ─── Significance threshold definition (PEER REVIEW REQUIREMENT) ─
sig_text = ('Significance: * p < 0.05, ** p < 0.01, *** p < 0.001\n'
            f'(χ² = {chi2_stat:.1f}, {p_value_str}; '
            f"Cramér's V = {cramers_v:.3f})")
props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9,
             edgecolor='#CCCCCC', linewidth=1.0)
ax.text(0.02, 0.97, sig_text, transform=ax.transAxes, fontsize=9.5,
        verticalalignment='top', horizontalalignment='left', bbox=props)

# ─── Axis styling ─────────────────────────────────────────────
ax.set_ylabel('Detection rate (%)', fontsize=13, fontweight='medium')
ax.set_ylim(0, 55)
ax.grid(axis='y', alpha=0.2, linestyle='-', linewidth=0.5, color='#CCCCCC')
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ─── Legend — upper center, inside plot ──────────────────────────
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.95),
          ncol=4, fontsize=11, frameon=True, framealpha=0.95,
          edgecolor='#CCCCCC', fancybox=True)

plt.tight_layout()

# ═══════════════════════════════════════════════════════════════════
# SAVE — 600 DPI PNG + PDF
# ═══════════════════════════════════════════════════════════════════
fig.savefig(OUTPUT_PATH / 'Figure_S3_Source_Signatures_FINAL.png',
            dpi=600, bbox_inches='tight', facecolor='white')
fig.savefig(OUTPUT_PATH / 'Figure_S3_Source_Signatures_FINAL.pdf',
            bbox_inches='tight', facecolor='white')
print("Saved Figure_S3_Source_Signatures_FINAL.png (600 DPI) and .pdf")
plt.close()
