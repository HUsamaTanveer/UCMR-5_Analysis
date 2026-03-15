"""
Figure S4: MCA Biplot — PFAS Co-Detection Patterns
Replaces PCA biplot per peer review (binary data → MCA is appropriate).
Visual style matches ORIGINAL PCA template (color-coded, annotated, inset).

Data source: mca_column_coordinates.csv (Jan 2026)
Inertia: mca_eigenvalues_inertia.csv
"""

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch
import numpy as np
from pathlib import Path
import shutil

OUTPUT_PATH = Path("/sessions/kind-modest-dijkstra/mnt/5_PFAS_Research_Finalization/05_Figures")
DELIVER_PATH = Path("/sessions/kind-modest-dijkstra/mnt/5_PFAS_Research_Finalization/08_Peer_Review_Deliverables/02_Figures")
DATA_PATH = Path("/sessions/kind-modest-dijkstra/mnt/5_PFAS_Research_Finalization/3_Jan2026_Update_Output_CSVs")

# ═══════════════════════════════════════════════════════════════════
# READ MCA COLUMN COORDINATES
# ═══════════════════════════════════════════════════════════════════
df_raw = pd.read_csv(DATA_PATH / 'mca_column_coordinates.csv', index_col=0)

records = []
for idx in df_raw.index:
    if idx.endswith('__1'):
        compound = idx.replace('__1', '')
        dim1 = df_raw.loc[idx, '0']
        dim2 = df_raw.loc[idx, '1']
        records.append({'compound': compound, 'Dim1': dim1, 'Dim2': dim2})

df = pd.DataFrame(records)
df['display_name'] = df['compound']

dim1_inertia = 18.2
dim2_inertia = 6.6

# ═══════════════════════════════════════════════════════════════════
# COMPOUND CATEGORIES
# ═══════════════════════════════════════════════════════════════════
regulated_pfas = ['PFOS', 'PFOA', 'PFHxS', 'PFNA', 'HFPO-DA']
commonly_detected = ['PFPeA', 'PFHxA', 'PFBS', 'PFBA', 'PFHpA', '6:2 FTS']

exclude_compounds = ['PFMBA', 'PFMPA', '11Cl-PF3OUdS', 'PFEESA', 'PFTA', 'PFTrDA', 'lithium']
df = df[~df['compound'].isin(exclude_compounds)].copy()

def categorize(compound):
    if compound in regulated_pfas:
        return 'Regulated'
    elif compound in commonly_detected:
        return 'Common'
    else:
        return 'Rare'

df['category'] = df['compound'].apply(categorize)

print(f"Plotting {len(df)} compounds")
for _, row in df.sort_values('Dim1').iterrows():
    print(f"  {row['compound']:>15s}: ({row['Dim1']:+7.3f}, {row['Dim2']:+7.3f})  [{row['category']}]")

# ═══════════════════════════════════════════════════════════════════
# COLORS
# ═══════════════════════════════════════════════════════════════════
CLR_REGULATED = '#8B1A1A'
CLR_COMMON    = '#4682B4'
CLR_RARE      = '#AAAAAA'

# ═══════════════════════════════════════════════════════════════════
# CREATE FIGURE — wider aspect ratio for better spacing
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 9))

df_reg  = df[df['category'] == 'Regulated']
df_com  = df[df['category'] == 'Common']
df_rare = df[df['category'] == 'Rare']

# ─── Arrows: regulated + common only (bold); rare = very subtle ──
for _, row in pd.concat([df_reg, df_com]).iterrows():
    d1, d2, cat = row['Dim1'], row['Dim2'], row['category']
    arrow_color = '#CD5C5C' if cat == 'Regulated' else '#6CA6CD'
    ax.annotate('', xy=(d1, d2), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=arrow_color,
                                lw=1.2, shrinkA=0, shrinkB=3, alpha=0.55))

for _, row in df_rare.iterrows():
    ax.annotate('', xy=(row['Dim1'], row['Dim2']), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='#DDDDDD',
                                lw=0.6, shrinkA=0, shrinkB=2, alpha=0.35))

# ─── Markers (rare behind, common middle, regulated on top) ──────
ax.scatter(df_rare['Dim1'], df_rare['Dim2'], s=35, marker='o',
           color=CLR_RARE, edgecolor='white', linewidth=0.3, zorder=3)
ax.scatter(df_com['Dim1'], df_com['Dim2'], s=110, marker='o',
           color=CLR_COMMON, edgecolor='white', linewidth=0.5, zorder=4)
ax.scatter(df_reg['Dim1'], df_reg['Dim2'], s=280, marker='*',
           color=CLR_REGULATED, edgecolor='white', linewidth=0.5, zorder=5)

# ─── Main-plot labels with leader lines (hand-tuned offsets) ──────
# Actual positions from data:
#   PFBA   (1.20, +0.104)   PFPeA  (1.67, +0.071)   PFBS  (1.76, +0.051)
#   PFHxA  (1.87, +0.020)   PFHpA  (2.95, -0.099)    6:2FTS(1.70, -0.143)
#   HFPO-DA(1.88, -0.019)   PFHxS  (2.01, -0.075)
#   PFOS   (1.97, -0.073)   PFOA   (2.06, -0.064)
#   PFNA   (2.95, -0.418)

label_config = {
    # Regulated — spread vertically to avoid the PFOS/PFOA/PFHxS pile-up
    'PFOS':    {'offset': (-55, -25), 'ha': 'right'},
    'PFOA':    {'offset': (14,  -22), 'ha': 'left'},
    'PFHxS':   {'offset': (14,   12), 'ha': 'left'},
    'PFNA':    {'offset': (12,  -15), 'ha': 'left'},
    'HFPO-DA': {'offset': (-55,  -8), 'ha': 'right'},
    # Commonly detected — spread around cluster
    'PFBA':    {'offset': (-12,  14), 'ha': 'right'},
    'PFPeA':   {'offset': (-12,  14), 'ha': 'right'},
    'PFBS':    {'offset': (10,   14), 'ha': 'left'},
    'PFHxA':   {'offset': (12,    6), 'ha': 'left'},
    'PFHpA':   {'offset': (10,   10), 'ha': 'left'},
    '6:2 FTS': {'offset': (10,  -14), 'ha': 'left'},
}

for _, row in pd.concat([df_reg, df_com]).iterrows():
    comp = row['compound']
    if comp not in label_config:
        continue
    cfg = label_config[comp]
    cat = row['category']
    color = CLR_REGULATED if cat == 'Regulated' else '#333333'
    fsize = 11 if cat == 'Regulated' else 10

    ax.annotate(row['display_name'],
                (row['Dim1'], row['Dim2']),
                xytext=cfg['offset'], textcoords='offset points',
                fontsize=fsize, fontweight='bold', color=color,
                ha=cfg['ha'], va='center',
                arrowprops=dict(arrowstyle='-', color=color, lw=0.5, alpha=0.35))

# ─── Interpretive annotations ─────────────────────────────────────
ax.annotate('Common / short-chain\ncompounds',
            xy=(1.45, 0.10), xytext=(0.15, 0.38),
            fontsize=10, fontstyle='italic', color='#777777',
            arrowprops=dict(arrowstyle='->', color='#BBBBBB', lw=1.3,
                            connectionstyle='arc3,rad=0.15'),
            ha='left', va='center')

ax.annotate('Rare co-detections',
            xy=(4.2, -0.50), xytext=(3.5, -1.05),
            fontsize=10, fontstyle='italic', color='#777777',
            arrowprops=dict(arrowstyle='->', color='#BBBBBB', lw=1.3),
            ha='center', va='center')

# ─── Axes ─────────────────────────────────────────────────────────
ax.set_xlabel(f'Dimension 1 ({dim1_inertia}% inertia)', fontsize=12, fontweight='medium')
ax.set_ylabel(f'Dimension 2 ({dim2_inertia}% inertia)', fontsize=12, fontweight='medium')
ax.set_xlim(-0.6, 5.3)
ax.set_ylim(-1.20, 0.52)

ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.2)
ax.axvline(x=0, color='black', linewidth=0.5, alpha=0.2)
ax.grid(True, alpha=0.15, linewidth=0.5, color='#CCCCCC')
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', labelsize=10)

# ─── Dashed rectangle around regulated+common cluster ─────────────
rect = mpatches.FancyBboxPatch((1.08, -0.16), 1.15, 0.32,
                                boxstyle='round,pad=0.02',
                                fill=False, edgecolor='#888888',
                                linestyle='--', linewidth=1.0, zorder=6)
ax.add_patch(rect)

# ─── INSET: upper-right (away from data) ─────────────────────────
ax_inset = ax.inset_axes([0.50, 0.52, 0.47, 0.44])

cluster_mask = (df['Dim1'] > 1.05) & (df['Dim1'] < 2.25) & \
               (df['Dim2'] > -0.16) & (df['Dim2'] < 0.15)
df_cluster = df[cluster_mask].copy()

for _, row in df_cluster.iterrows():
    d1, d2, cat = row['Dim1'], row['Dim2'], row['category']
    if cat == 'Regulated':
        ax_inset.scatter(d1, d2, s=200, marker='*', color=CLR_REGULATED,
                         edgecolor='white', linewidth=0.5, zorder=5)
    elif cat == 'Common':
        ax_inset.scatter(d1, d2, s=80, marker='o', color=CLR_COMMON,
                         edgecolor='white', linewidth=0.5, zorder=4)
    else:
        ax_inset.scatter(d1, d2, s=30, marker='o', color=CLR_RARE,
                         edgecolor='white', linewidth=0.3, zorder=3)

# Inset labels — carefully separated
# Actual inset positions (sorted by Dim2 descending for layout):
#   PFBA   (1.20, +0.104)  — top-left
#   PFPeA  (1.67, +0.071)  — center-top
#   PFBS   (1.76, +0.051)  — center-top-right
#   PFHxA  (1.87, +0.020)  — center-right
#   HFPO-DA(1.88, -0.019)  — center-right-low
#   PFOA   (2.06, -0.064)  — far right
#   PFOS   (1.97, -0.073)  — right-low (CLOSE to PFOA and PFHxS)
#   PFHxS  (2.01, -0.075)  — right-low (CLOSE to PFOS)
#   6:2 FTS(1.70, -0.143)  — center-bottom
#   ADONA  (2.02, -0.079)  — right-low (rare)
#   NFDHA  (2.49, +0.149)  — if in range

inset_labels = {
    'PFBA':     {'offset': (-22,  8),  'ha': 'right'},
    'PFPeA':    {'offset': (-20,  8),  'ha': 'right'},
    'PFBS':     {'offset': (8,   10),  'ha': 'left'},
    'PFHxA':    {'offset': (8,    8),  'ha': 'left'},
    'HFPO-DA':  {'offset': (-30,  3),  'ha': 'right'},
    'PFOS':     {'offset': (-30, -6),  'ha': 'right'},
    'PFOA':     {'offset': (8,    6),  'ha': 'left'},
    'PFHxS':    {'offset': (8,  -10),  'ha': 'left'},
    '6:2 FTS':  {'offset': (-10,-12),  'ha': 'right'},
    'ADONA':    {'offset': (8,  -10),  'ha': 'left'},
    'NFDHA':    {'offset': (-10,  6),  'ha': 'right'},
    '9Cl-PF3ONS':{'offset':(-10, -8),  'ha': 'right'},
}

for _, row in df_cluster.iterrows():
    comp = row['compound']
    cat = row['category']
    color = CLR_REGULATED if cat == 'Regulated' else '#444444'
    fsize = 9 if cat in ['Regulated', 'Common'] else 7.5
    fweight = 'bold' if cat in ['Regulated', 'Common'] else 'medium'

    if comp in inset_labels:
        cfg = inset_labels[comp]
        ax_inset.annotate(row['display_name'],
                          (row['Dim1'], row['Dim2']),
                          xytext=cfg['offset'], textcoords='offset points',
                          fontsize=fsize, fontweight=fweight, color=color,
                          ha=cfg['ha'], va='center',
                          arrowprops=dict(arrowstyle='-', color=color,
                                          lw=0.4, alpha=0.3))

ax_inset.set_xlim(1.05, 2.20)
ax_inset.set_ylim(-0.16, 0.14)
ax_inset.set_title('Inset: main cluster', fontsize=9.5, fontweight='bold',
                    loc='left', pad=6)
ax_inset.tick_params(axis='both', labelsize=8)
ax_inset.grid(True, alpha=0.12, linestyle='-')
ax_inset.set_facecolor('#FAFAFA')
for spine in ax_inset.spines.values():
    spine.set_color('#888888')
    spine.set_linewidth(0.8)

# Connect rectangle to inset
con1 = ConnectionPatch(xyA=(2.23, 0.16), coordsA='data', axesA=ax,
                       xyB=(0, 1), coordsB='axes fraction', axesB=ax_inset,
                       color='#AAAAAA', linestyle=':', linewidth=0.8)
con2 = ConnectionPatch(xyA=(2.23, -0.16), coordsA='data', axesA=ax,
                       xyB=(0, 0), coordsB='axes fraction', axesB=ax_inset,
                       color='#AAAAAA', linestyle=':', linewidth=0.8)
fig.add_artist(con1)
fig.add_artist(con2)

# ─── Exclusion note ──────────────────────────────────────────────
ax.text(0.98, 0.02,
        '2 rare compounds with extreme coordinates excluded\n'
        '(PFMBA, PFMPA: near-zero detection artifacts)',
        transform=ax.transAxes, fontsize=8, fontstyle='italic', color='#AAAAAA',
        ha='right', va='bottom')

# ─── Legend ───────────────────────────────────────────────────────
legend_elements = [
    Line2D([0], [0], marker='*', color='w', markerfacecolor=CLR_REGULATED,
           markersize=16, label='Regulated PFAS (individual MCL)',
           markeredgecolor='white', markeredgewidth=0.5),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=CLR_COMMON,
           markersize=10, label='Commonly detected (unregulated)',
           markeredgecolor='white', markeredgewidth=0.5),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=CLR_RARE,
           markersize=7, label='Rarely / never detected',
           markeredgecolor='white', markeredgewidth=0.5),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=10,
          framealpha=0.95, edgecolor='#CCCCCC', fancybox=True)

plt.tight_layout()

# ═══════════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════════
fig.savefig(OUTPUT_PATH / 'Figure_S4_MCA_Biplot_FINAL.png',
            dpi=600, bbox_inches='tight', facecolor='white')
fig.savefig(OUTPUT_PATH / 'Figure_S4_MCA_Biplot_FINAL.pdf',
            bbox_inches='tight', facecolor='white')

for ext in ['png', 'pdf']:
    src = OUTPUT_PATH / f'Figure_S4_MCA_Biplot_FINAL.{ext}'
    dst = DELIVER_PATH / f'Figure_S4_MCA_Biplot_FINAL.{ext}'
    shutil.copy2(src, dst)

print("\n✓ Saved Figure_S4_MCA_Biplot_FINAL.png (600 DPI) and .pdf")
print("✓ Copied to 08_Peer_Review_Deliverables/02_Figures/")
plt.close()
