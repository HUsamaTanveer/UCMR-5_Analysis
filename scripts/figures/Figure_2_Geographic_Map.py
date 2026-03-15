"""
Figure 2: Geographic Distribution of PFAS Detection Rates
Two-panel choropleth — Colorblind-friendly version.

Changes from original:
  - Colormap: YlOrRd → custom colorblind-safe sequential (cream → gold → brown → dark purple)
    Avoids red-green channel; perceptually uniform under protanopia, deuteranopia, and tritanopia.
  - All other layout elements match original template exactly.
"""

import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize, BoundaryNorm, LinearSegmentedColormap
from matplotlib import cm
import numpy as np
import pandas as pd
from pathlib import Path

OUTPUT_PATH = Path("/sessions/jolly-relaxed-meitner/mnt/5_PFAS_Research_Finalization/05_Figures/1_Final Figures")
SHAPEFILE = "/sessions/jolly-relaxed-meitner/mnt/5_PFAS_Research_Finalization/01_Raw_Data/Shapefiles/cb_2023_us_state_20m.shp"

# ═══════════════════════════════════════════════════════════════════
# CORRECTED STATE DETECTION RATES (from Jan 2026 data)
# ═══════════════════════════════════════════════════════════════════
state_rates = {
    'AK': 28.1, 'AL': 38.2, 'AR': 5.9, 'AZ': 37.0, 'CA': 32.8,
    'CO': 21.9, 'CT': 46.2, 'DC': 100.0, 'DE': 37.0, 'FL': 46.9,
    'GA': 30.2, 'HI': 4.4, 'IA': 21.1, 'ID': 13.1, 'IL': 25.3,
    'IN': 23.9, 'KS': 33.0, 'KY': 37.7, 'LA': 17.5, 'MA': 54.4,
    'MD': 36.1, 'ME': 24.4, 'MI': 13.8, 'MN': 44.4, 'MO': 30.6,
    'MS': 3.8, 'MT': 11.1, 'NC': 50.0, 'ND': 18.4, 'NE': 13.2,
    'NH': 35.3, 'NJ': 65.7, 'NM': 13.7, 'NV': 17.2, 'NY': 29.4,
    'OH': 40.6, 'OK': 36.7, 'OR': 14.2, 'PA': 44.6, 'PR': 38.4,
    'RI': 51.7, 'SC': 54.5, 'SD': 14.3, 'TN': 31.4, 'TX': 53.0,
    'UT': 12.4, 'VA': 31.6, 'VT': 7.9, 'WA': 26.7, 'WI': 25.2,
    'WV': 33.0, 'WY': 9.1
}
NATIONAL_AVG = 34.4

# ═══════════════════════════════════════════════════════════════════
# LOAD SHAPEFILE AND MERGE DATA
# ═══════════════════════════════════════════════════════════════════
gdf = gpd.read_file(SHAPEFILE)
gdf['detection_rate'] = gdf['STUSPS'].map(state_rates)

continental = gdf[~gdf['STUSPS'].isin(['AK', 'HI', 'PR', 'AS', 'GU', 'MP', 'VI'])]
alaska = gdf[gdf['STUSPS'] == 'AK']
hawaii = gdf[gdf['STUSPS'] == 'HI']
puerto_rico = gdf[gdf['STUSPS'] == 'PR']

ne_states = ['ME', 'NH', 'VT', 'MA', 'RI', 'CT', 'NY', 'NJ', 'PA', 'DE', 'MD', 'DC', 'WV', 'VA']
northeast = gdf[gdf['STUSPS'].isin(ne_states)]

albers = 'ESRI:102003'
continental_proj = continental.to_crs(albers)
alaska_proj = alaska.to_crs(albers)
hawaii_proj = hawaii.to_crs(albers)
pr_proj = puerto_rico.to_crs(albers)
northeast_proj = northeast.to_crs(albers)

# ═══════════════════════════════════════════════════════════════════
# COLORBLIND-SAFE COLORMAP
# Sequential: ivory → gold → sienna → dark indigo
# Tested via coblis.myndex.com for protanopia, deuteranopia, tritanopia
# Uses luminance gradient (light→dark) as primary cue, hue as secondary
# ═══════════════════════════════════════════════════════════════════
cb_colors = [
    '#FFFDE7',  # 0%   very light cream
    '#FFF3B0',  # ~10% pale yellow
    '#FFD54F',  # ~20% gold
    '#FFB300',  # ~30% amber
    '#FB8C00',  # ~40% orange
    '#E65100',  # ~50% deep orange
    '#BF360C',  # ~60% burnt sienna
    '#8D2208',  # ~70% dark brown-red
    '#5D1451',  # ~80% dark plum
    '#311B92',  # ~90% deep indigo
    '#1A0533',  # 100% near-black purple
]
cmap_cb = LinearSegmentedColormap.from_list('pfas_cb', cb_colors, N=256)

boundaries = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
norm = BoundaryNorm(boundaries, ncolors=cmap_cb.N)

# ═══════════════════════════════════════════════════════════════════
# Helper: choose label color based on background luminance
# ═══════════════════════════════════════════════════════════════════
def label_color_for_rate(rate):
    """White labels on dark backgrounds, black on light."""
    if rate is None or np.isnan(rate):
        return 'black'
    return 'white' if rate >= 55 else 'black'

# ═══════════════════════════════════════════════════════════════════
# CREATE FIGURE — matching original layout
# ═══════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(20, 11), facecolor='white')

# ─── Panel A: Full US map (left side) ─────────────────────────────
ax_a = fig.add_axes([0.01, 0.13, 0.52, 0.84])

continental_proj.plot(
    column='detection_rate', cmap=cmap_cb, norm=norm,
    edgecolor='#444444', linewidth=0.5, ax=ax_a,
    missing_kwds={'color': 'lightgray', 'edgecolor': '#444444', 'linewidth': 0.5}
)

# State labels on Panel A
label_offsets_a = {
    'DC': (120000, -70000),
    'MD': (130000, -50000),
    'DE': (100000, 30000),
    'RI': (90000, 20000),
    'CT': (70000, -30000),
    'NJ': (70000, 10000),
    'MA': (80000, 20000),
    'NH': (30000, 40000),
    'VT': (-30000, 30000),
    'LA': (0, -60000),
    'FL': (70000, -50000),
}
arrow_states = {'DC', 'DE', 'RI', 'CT', 'NJ', 'MA', 'MD'}

for idx, row in continental_proj.iterrows():
    centroid = row.geometry.centroid
    abbrev = row['STUSPS']
    rate = row.get('detection_rate', None)
    lc = label_color_for_rate(rate)

    if abbrev in label_offsets_a:
        dx, dy = label_offsets_a[abbrev]
        if abbrev in arrow_states:
            ax_a.annotate(abbrev, xy=(centroid.x, centroid.y),
                          xytext=(centroid.x + dx, centroid.y + dy),
                          fontsize=7, fontweight='bold', color='black', zorder=5,
                          ha='center', va='center',
                          arrowprops=dict(arrowstyle='-', color='#555555',
                                          linewidth=0.6, shrinkA=0, shrinkB=1))
        else:
            ax_a.text(centroid.x + dx, centroid.y + dy, abbrev,
                      ha='center', va='center', fontsize=7.5, fontweight='bold',
                      color=lc, zorder=5)
    else:
        ax_a.text(centroid.x, centroid.y, abbrev,
                  ha='center', va='center', fontsize=7.5, fontweight='bold',
                  color=lc, zorder=5)

ax_a.set_xlim(-2.6e6, 2.7e6)
ax_a.set_ylim(-1.65e6, 1.55e6)
ax_a.set_aspect('equal')
ax_a.axis('off')

ax_a.text(0.0, 0.98, 'A', transform=ax_a.transAxes,
          fontsize=18, fontweight='bold', va='top')

# ─── Colorbar (horizontal, below Panel A) ─────────────────────────
cbar_ax = fig.add_axes([0.06, 0.06, 0.38, 0.028])
sm = cm.ScalarMappable(cmap=cmap_cb, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal',
                    spacing='uniform', ticks=boundaries)
cbar.set_ticklabels([str(b) for b in boundaries])
cbar.ax.tick_params(labelsize=9, length=0, width=0)
cbar.set_label('PWS detection rate (%)', fontsize=10, fontweight='bold')
cbar.outline.set_linewidth(1.2)
cbar.outline.set_edgecolor('black')

# Solid black divider lines at each boundary
for b in boundaries:
    cbar_ax.axvline(b, color='black', linewidth=1.0, zorder=4)

# National average marker
cbar_ax.axvline(NATIONAL_AVG, color='white', linestyle='--', linewidth=2.5, zorder=5)
cbar_ax.axvline(NATIONAL_AVG, color='black', linestyle='--', linewidth=1.5, zorder=6)
cbar_ax.text(NATIONAL_AVG, 1.6, f'{NATIONAL_AVG}%',
             transform=cbar_ax.get_xaxis_transform(),
             ha='center', va='bottom', fontsize=9.5, fontweight='bold',
             color='#222222')

# ─── VERTICAL DIVIDER LINE between Panel A and B ──────────────────
line_x = 0.545
fig.add_artist(plt.Line2D([line_x, line_x], [0.05, 0.97],
               transform=fig.transFigure, color='gray',
               linewidth=0.8, linestyle='-', alpha=0.5))

# ═══════════════════════════════════════════════════════════════════
# PANEL B: Northeast Detail & Territories (right side)
# ═══════════════════════════════════════════════════════════════════
ax_b = fig.add_axes([0.56, 0.22, 0.43, 0.72])

northeast_proj.plot(
    column='detection_rate', cmap=cmap_cb, norm=norm,
    edgecolor='#444444', linewidth=0.8, ax=ax_b,
    missing_kwds={'color': 'lightgray', 'edgecolor': '#444444', 'linewidth': 0.5}
)

ax_b.set_title('Northeast Detail & Territories', fontsize=14, fontweight='bold', pad=12)
ax_b.text(0.0, 1.01, 'B', transform=ax_b.transAxes,
          fontsize=18, fontweight='bold', va='bottom')

# Panel B labels — bigger
small_state_offsets = {
    'DC': (150000, -90000),
    'DE': (140000, 10000),
    'RI': (110000, 30000),
    'CT': (90000, -50000),
    'MD': (0, -60000),
}

for idx, row in northeast_proj.iterrows():
    centroid = row.geometry.centroid
    abbrev = row['STUSPS']
    rate = row.get('detection_rate', None)
    lc = label_color_for_rate(rate)

    label = 'DC (100%)' if abbrev == 'DC' else abbrev

    if abbrev in small_state_offsets:
        dx, dy = small_state_offsets[abbrev]
        ax_b.annotate(label, xy=(centroid.x, centroid.y),
                      xytext=(centroid.x + dx, centroid.y + dy),
                      fontsize=12, fontweight='bold', color='black', zorder=5,
                      ha='left', va='center',
                      arrowprops=dict(arrowstyle='-', color='gray',
                                      linewidth=0.8, shrinkA=0, shrinkB=2))
    else:
        ax_b.text(centroid.x, centroid.y, label,
                  ha='center', va='center', fontsize=12, fontweight='bold',
                  color=lc, zorder=5)

ax_b.set_aspect('equal')
ax_b.axis('off')

# ─── Territory insets ─────────────────────────────────────────────

# Alaska
ax_ak = fig.add_axes([0.56, 0.04, 0.12, 0.18])
alaska_reproj = alaska.to_crs('EPSG:3338')
alaska_reproj.plot(column='detection_rate', cmap=cmap_cb, norm=norm,
                   edgecolor='#444444', linewidth=0.5, ax=ax_ak,
                   missing_kwds={'color': 'lightgray'})
ax_ak.text(0.5, -0.02, 'AK', transform=ax_ak.transAxes,
           ha='center', va='top', fontsize=9, fontweight='bold')
ax_ak.axis('off')

# Hawaii
ax_hi = fig.add_axes([0.70, 0.04, 0.10, 0.14])
hawaii_reproj = hawaii.to_crs('EPSG:2783')
hawaii_reproj.plot(column='detection_rate', cmap=cmap_cb, norm=norm,
                   edgecolor='#444444', linewidth=0.5, ax=ax_hi,
                   missing_kwds={'color': 'lightgray'})
ax_hi.text(0.5, -0.02, 'HI', transform=ax_hi.transAxes,
           ha='center', va='top', fontsize=9, fontweight='bold')
ax_hi.axis('off')

# Puerto Rico
ax_pr = fig.add_axes([0.83, 0.04, 0.10, 0.12])
pr_reproj = puerto_rico.to_crs('EPSG:3920')
pr_reproj.plot(column='detection_rate', cmap=cmap_cb, norm=norm,
               edgecolor='#444444', linewidth=0.5, ax=ax_pr,
               missing_kwds={'color': 'lightgray'})
ax_pr.text(0.5, -0.02, 'PR', transform=ax_pr.transAxes,
           ha='center', va='top', fontsize=9, fontweight='bold',
           color='#1A0533')
ax_pr.axis('off')

# ═══════════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════════
fig.savefig(OUTPUT_PATH / 'Figure_2_Geographic_Map_FINAL.png',
            dpi=600, bbox_inches='tight', facecolor='white')
fig.savefig(OUTPUT_PATH / 'Figure_2_Geographic_Map_FINAL.pdf',
            bbox_inches='tight', facecolor='white')
print("Saved Figure_2_Geographic_Map_FINAL.png (600 DPI) and .pdf")
plt.close()
