#!/usr/bin/env python3
"""
PFAS UCMR 5 Analysis - Phase 3: Co-Occurrence, MCA & Mixture Patterns (ENHANCED)
==================================================================================
Replaces PCA with Multiple Correspondence Analysis (MCA) using prince library.
MCA is the correct method for categorical/binary data analysis.

Outputs:
  - mca_dimension_contributions.csv (column coordinates/loadings)
  - mca_scree_data.csv (inertia by dimension)
  - mca_bootstrap_cis.csv (bootstrap CIs for top loadings)
  - cooccurrence_profiles_enhanced.csv (with Wilson CIs)
  - compound_count_distribution_enhanced.csv

Verifies: Section 3.5, Tables S7-S9, Figure 4A/B, Figure S4
"""

import pandas as pd
import numpy as np
import prince
from scipy import stats
from datetime import datetime
import json
import os
import warnings
warnings.filterwarnings('ignore')

BASE = "/sessions/optimistic-determined-feynman/mnt/5_PFAS_Research_Finalization"
RAW_FILE = os.path.join(BASE, "01_Raw_Data", "UCMR5_Jan 2025 Update", "UCMR5_All.txt")
OUTPUT_DIR = os.path.join(BASE, "3_Jan2026_Update_Output_CSVs")

results = {}
log_lines = []

def log(msg):
    """Log message to console and results file"""
    print(msg)
    log_lines.append(msg)

def check(label, computed, manuscript, tolerance=0.05):
    """Verify computed value against manuscript value"""
    if isinstance(computed, float) and isinstance(manuscript, float):
        if manuscript != 0:
            pct_diff = abs(computed - manuscript) / abs(manuscript) * 100
        else:
            pct_diff = abs(computed - manuscript) * 100
        match = "MATCH" if pct_diff < tolerance * 100 else "MISMATCH"
        log(f"  [{match}] {label}: computed={computed}, manuscript={manuscript}, diff={pct_diff:.2f}%")
    else:
        match = "MATCH" if computed == manuscript else "MISMATCH"
        log(f"  [{match}] {label}: computed={computed}, manuscript={manuscript}")
    results[label] = {"computed": str(computed), "manuscript": str(manuscript), "status": match}

def wilson_ci(k, n, z=1.96):
    """
    Calculate Wilson score confidence interval for binomial proportion.
    More reliable than standard normal CI, especially for extreme proportions.
    
    Parameters:
    -----------
    k : int - number of successes
    n : int - total number of trials
    z : float - z-score (default 1.96 for 95% CI)
    
    Returns:
    --------
    tuple : (lower_ci, upper_ci)
    """
    if n == 0:
        return (0, 0)
    
    p_hat = k / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    spread = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
    
    return (max(0, center - spread), min(1, center + spread))

log("=" * 100)
log(f"PFAS UCMR 5 ENHANCED REANALYSIS - Phase 3: Co-Occurrence & MCA")
log(f"Run date: {datetime.now().isoformat()}")
log("=" * 100)

# ============================================================
# LOAD AND PREPARE DATA
# ============================================================
log("\n--- Loading and preparing UCMR5 data ---")

df = pd.read_csv(RAW_FILE, sep='\t', encoding='latin-1', low_memory=False)
log(f"Raw data loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")

# Filter out lithium, keep only PFAS compounds
pfas = df[df['Contaminant'] != 'lithium'].copy()
pfas['AnalyticalResultValue'] = pd.to_numeric(pfas['AnalyticalResultValue'], errors='coerce')
pfas['detected'] = pfas['AnalyticalResultsSign'] == '='

compounds = sorted(pfas['Contaminant'].unique())
all_pws = pfas['PWSID'].unique()
n_pws = len(all_pws)

log(f"PFAS data filtered: {pfas.shape[0]:,} records")
log(f"Unique PWS: {n_pws:,}")
log(f"Unique compounds: {len(compounds)}")

# ============================================================
# BUILD BINARY DETECTION MATRIX
# ============================================================
log("\n--- Building binary detection matrix ---")

det_matrix = pfas[pfas['detected']].groupby(['PWSID', 'Contaminant']).size().unstack(fill_value=0)
det_matrix = (det_matrix > 0).astype(int)
# Ensure all PWS are included (even those with no detection)
det_matrix = det_matrix.reindex(all_pws, fill_value=0)
# Ensure all compounds are present
for c in compounds:
    if c not in det_matrix.columns:
        det_matrix[c] = 0
det_matrix = det_matrix[sorted(det_matrix.columns)]

log(f"Detection matrix shape: {det_matrix.shape[0]:,} PWS × {det_matrix.shape[1]} compounds")
log(f"Detection matrix sparsity: {(det_matrix.sum().sum() / (det_matrix.shape[0] * det_matrix.shape[1]) * 100):.2f}%")

# ============================================================
# REGULATED PFAS CO-DETECTION PROFILES (Table S7)
# ============================================================
log("\n--- Regulated PFAS Co-Detection Profiles (Table S7) ---")

reg_compounds = ['PFOS', 'PFOA', 'PFHxS', 'PFNA', 'HFPO-DA']
reg_matrix = det_matrix[reg_compounds]

# Create profile string
profiles = reg_matrix.apply(lambda row: ''.join(['1' if row[c] else '0' for c in reg_compounds]), axis=1)
profile_counts = profiles.value_counts()

# Map to readable format
def profile_label(p):
    labels = []
    for i, c in enumerate(reg_compounds):
        if p[i] == '1':
            labels.append(c)
    return '+'.join(labels) if labels else 'None'

log("\nRegulated PFAS co-detection profiles (top 15):")
for prof, count in profile_counts.head(15).items():
    pct = round(count / n_pws * 100, 2)
    log(f"  {profile_label(prof):40s} [{prof}]: {count:5d} ({pct}%)")

# Key profiles from Table S7
none_profile = profile_counts.get('00000', 0)
check("No regulated PFAS (Table S7)", none_profile, 8322)
check("No regulated PFAS (%)", round(none_profile/n_pws*100, 1), 81.2)

# ============================================================
# MULTIPLE CORRESPONDENCE ANALYSIS (MCA)
# ============================================================
log("\n--- Multiple Correspondence Analysis (MCA) ---")
log(f"Performing MCA on {det_matrix.shape[0]:,} × {det_matrix.shape[1]} binary detection matrix...")

# Prince MCA expects a DataFrame
mca = prince.MCA(n_components=10, random_state=42)
mca = mca.fit(det_matrix)

log(f"MCA fit completed successfully")
log(f"MCA components: {mca.components_.shape}")

# Get eigenvalues (inertia) - these represent variance explained by each dimension
eigenvalues = mca.eigenvalues_
inertia = eigenvalues / eigenvalues.sum()
cumulative_inertia = np.cumsum(inertia)

log(f"\nMCA Eigenvalues (first 10 dimensions):")
for i, (eig, iner, cum_iner) in enumerate(zip(eigenvalues[:10], inertia[:10], cumulative_inertia[:10])):
    log(f"  Dimension {i+1}: eigenvalue={eig:.6f}, inertia={iner*100:.2f}%, cumulative={cum_iner*100:.2f}%")

# Get column coordinates (analogous to PCA loadings)
col_coords = mca.column_coordinates(det_matrix)
log(f"\nColumn coordinates shape: {col_coords.shape}")
log(f"Column coordinates (first 5 compounds, first 3 dimensions):")
log(col_coords.iloc[:5, :3].to_string())

# Get row coordinates (component scores)
row_coords = mca.row_coordinates(det_matrix)
log(f"Row coordinates shape: {row_coords.shape}")

# ============================================================
# SCREE PLOT DATA
# ============================================================
log("\n--- MCA Scree Plot Data ---")

scree_data = pd.DataFrame({
    'Dimension': range(1, len(eigenvalues) + 1),
    'Eigenvalue': eigenvalues,
    'Inertia_%': inertia * 100,
    'Cumulative_Inertia_%': cumulative_inertia * 100
})

scree_output = os.path.join(OUTPUT_DIR, "mca_scree_data.csv")
scree_data.to_csv(scree_output, index=False)
log(f"Scree data saved to {scree_output}")

# ============================================================
# BOOTSTRAP CONFIDENCE INTERVALS FOR MCA LOADINGS
# ============================================================
log("\n--- Bootstrap Confidence Intervals for MCA Loadings (n=1000) ---")

n_bootstrap = 1000
bootstrap_loadings = {}

np.random.seed(42)
for b in range(n_bootstrap):
    if (b + 1) % 200 == 0:
        log(f"  Bootstrap iteration {b+1}/{n_bootstrap}")
    
    # Resample with replacement from rows
    idx = np.random.choice(det_matrix.shape[0], size=det_matrix.shape[0], replace=True)
    boot_matrix = det_matrix.iloc[idx, :].copy()
    
    try:
        # Fit MCA on bootstrap sample
        mca_boot = prince.MCA(n_components=10, random_state=42)
        mca_boot = mca_boot.fit(boot_matrix)
        boot_coords = mca_boot.column_coordinates(boot_matrix)
        
        # Store loadings for first 3 dimensions
        for dim in range(3):
            key = f"Dimension_{dim+1}"
            if key not in bootstrap_loadings:
                bootstrap_loadings[key] = []
            bootstrap_loadings[key].append(boot_coords.iloc[:, dim].values)
    except Exception as e:
        if b == 0:
            log(f"  Warning: Bootstrap iteration {b+1} failed: {str(e)}")

# Calculate bootstrap CIs
bootstrap_ci_data = []
for compound_idx, compound in enumerate(det_matrix.columns):
    for dim in range(3):
        key = f"Dimension_{dim+1}"
        if key in bootstrap_loadings and len(bootstrap_loadings[key]) > 0:
            boot_vals = np.array([boot[compound_idx] for boot in bootstrap_loadings[key]])
            
            ci_lower = np.percentile(boot_vals, 2.5)
            ci_upper = np.percentile(boot_vals, 97.5)
            mean_val = np.mean(boot_vals)
            std_val = np.std(boot_vals)
            
            bootstrap_ci_data.append({
                'Compound': compound,
                'Dimension': dim + 1,
                'Mean_Loading': round(mean_val, 6),
                'Std_Loading': round(std_val, 6),
                'CI_Lower': round(ci_lower, 6),
                'CI_Upper': round(ci_upper, 6),
                'Bootstrap_n': len(boot_vals)
            })

bootstrap_ci_df = pd.DataFrame(bootstrap_ci_data)
bootstrap_output = os.path.join(OUTPUT_DIR, "mca_bootstrap_cis.csv")
bootstrap_ci_df.to_csv(bootstrap_output, index=False)
log(f"Bootstrap CIs saved to {bootstrap_output}")

# Show top loadings by magnitude
log("\nTop 10 loadings by magnitude (Dimension 1):")
dim1_cis = bootstrap_ci_df[bootstrap_ci_df['Dimension'] == 1].sort_values('Mean_Loading', key=abs, ascending=False)
for idx, row in dim1_cis.head(10).iterrows():
    log(f"  {row['Compound']:15s}: {row['Mean_Loading']:7.4f} [{row['CI_Lower']:7.4f}, {row['CI_Upper']:7.4f}]")

# ============================================================
# SAVE MCA DIMENSION CONTRIBUTIONS
# ============================================================
log("\n--- MCA Dimension Contributions (Loadings) ---")

# Prepare loadings dataframe with top dimensions
loadings_df = col_coords.iloc[:, :5].copy()
loadings_df.columns = [f'Dimension_{i+1}' for i in range(5)]

loadings_output = os.path.join(OUTPUT_DIR, "mca_dimension_contributions.csv")
loadings_df.to_csv(loadings_output)
log(f"MCA dimension contributions saved to {loadings_output}")

# Display key loadings for verification
log("\nTop MCA Loadings - Dimension 1 (first component):")
dim1_loadings = loadings_df['Dimension_1'].sort_values(key=abs, ascending=False)
for compound, loading in dim1_loadings.head(10).items():
    log(f"  {compound:15s}: {loading:7.4f}")

log("\nTop MCA Loadings - Dimension 2:")
dim2_loadings = loadings_df['Dimension_2'].sort_values(key=abs, ascending=False)
for compound, loading in dim2_loadings.head(10).items():
    log(f"  {compound:15s}: {loading:7.4f}")

# ============================================================
# CO-DETECTION PROFILES WITH WILSON CIs
# ============================================================
log("\n--- Co-Detection Profiles with Wilson Confidence Intervals ---")

cooccurrence_data = []

for prof, count in profile_counts.items():
    # Calculate Wilson CI for this profile frequency
    ci_lower, ci_upper = wilson_ci(count, n_pws)
    
    cooccurrence_data.append({
        'Profile': profile_label(prof),
        'Profile_Binary': prof,
        'Count': count,
        'Percentage': round(count / n_pws * 100, 2),
        'CI_Lower_%': round(ci_lower * 100, 2),
        'CI_Upper_%': round(ci_upper * 100, 2),
        'CI_Width_%': round((ci_upper - ci_lower) * 100, 2)
    })

cooccurrence_df = pd.DataFrame(cooccurrence_data).sort_values('Count', ascending=False)
cooccurrence_output = os.path.join(OUTPUT_DIR, "cooccurrence_profiles_enhanced.csv")
cooccurrence_df.to_csv(cooccurrence_output, index=False)
log(f"Cooccurrence profiles with Wilson CIs saved to {cooccurrence_output}")

log("\nTop 10 regulated PFAS profiles with Wilson CIs:")
for idx, row in cooccurrence_df.head(10).iterrows():
    log(f"  {row['Profile']:40s}: {row['Count']:5d} ({row['Percentage']:5.2f}%) "
        f"[{row['CI_Lower_%']:5.2f}%, {row['CI_Upper_%']:5.2f}%]")

# ============================================================
# COMPOUND DETECTION RATES (among detected systems)
# ============================================================
log("\n--- Compound Detection Rates (among detected systems, Table S9) ---")

detected_sys = det_matrix[det_matrix.sum(axis=1) > 0]
n_det_sys = len(detected_sys)

log(f"Number of systems with at least one detection: {n_det_sys:,}")
log(f"Percentage of PWS with detection: {n_det_sys/n_pws*100:.1f}%")

detection_rates = []
for comp in sorted(det_matrix.columns):
    rate = detected_sys[comp].mean() * 100
    count = detected_sys[comp].sum()
    detection_rates.append({
        'Compound': comp,
        'Detected_Count': int(count),
        'Total_Detected_Systems': n_det_sys,
        'Detection_Rate_%': round(rate, 2)
    })

detection_rates_df = pd.DataFrame(detection_rates).sort_values('Detection_Rate_%', ascending=False)
log("\nDetection rates (among detected systems, top 15):")
for idx, row in detection_rates_df.head(15).iterrows():
    log(f"  {row['Compound']:15s}: {row['Detection_Rate_%']:6.2f}% ({row['Detected_Count']:4d}/{n_det_sys:,})")

# Verify SI Table S9 detection rates
check("PFPeA det rate (det systems)", round(detected_sys['PFPeA'].mean()*100, 2), 19.17)
check("PFHxA det rate (det systems)", round(detected_sys['PFHxA'].mean()*100, 2), 16.95)
check("PFBS det rate (det systems)", round(detected_sys['PFBS'].mean()*100, 2), 15.82)
check("PFBA det rate (det systems)", round(detected_sys['PFBA'].mean()*100, 2), 17.81)

# ============================================================
# COMPOUND COUNT DISTRIBUTION (Figure 4A, Text S4)
# ============================================================
log("\n--- Compound Count Distribution (Figure 4A) ---")

comp_counts = det_matrix.sum(axis=1)
det_counts = comp_counts[comp_counts > 0]

# Create distribution
comp_dist = []
for n_comps in range(1, int(det_counts.max()) + 1):
    count = (det_counts == n_comps).sum()
    pct = count / len(det_counts) * 100 if len(det_counts) > 0 else 0
    
    comp_dist.append({
        'Num_Compounds': n_comps,
        'Count': count,
        'Percentage': round(pct, 2),
        'Cumulative_Count': int((det_counts >= n_comps).sum()),
        'Cumulative_Percentage': round((det_counts >= n_comps).sum() / len(det_counts) * 100, 2) if len(det_counts) > 0 else 0
    })

comp_dist_df = pd.DataFrame(comp_dist)
comp_dist_output = os.path.join(OUTPUT_DIR, "compound_count_distribution_enhanced.csv")
comp_dist_df.to_csv(comp_dist_output, index=False)
log(f"Compound count distribution saved to {comp_dist_output}")

# Key statistics
single = (det_counts == 1).sum()
pct_single = round(single / len(det_counts) * 100, 1) if len(det_counts) > 0 else 0
check("Single-compound systems (n)", single, 1134)
check("Single-compound systems (%)", pct_single, 32.0)

four_plus = (det_counts >= 4).sum()
pct_four_plus = round(four_plus / len(det_counts) * 100, 1) if len(det_counts) > 0 else 0
check("4+ compound systems (n)", four_plus, 1469)
check("4+ compound systems (%)", pct_four_plus, 41.5)

log("\nCompound count distribution (Figure 4A):")
for idx, row in comp_dist_df.iterrows():
    log(f"  {row['Num_Compounds']:2d} compounds: {row['Count']:5d} ({row['Percentage']:5.2f}%) "
        f"cumulative: {row['Cumulative_Count']:5d} ({row['Cumulative_Percentage']:5.2f}%)")

# ============================================================
# HIERARCHICAL CLUSTERING VERIFICATION
# ============================================================
log("\n--- Cluster Verification ---")

# The manuscript mentions 4 clusters
# Cluster 1: 8 commonly detected (mean det 13.7%)
cluster1_comps = ['PFPeA', 'PFHxA', 'PFBS', 'PFBA', 'PFOS', 'PFOA', 'PFHxS', 'PFHpA']
cluster1_rates = [det_matrix[c].mean() * 100 for c in cluster1_comps]
cluster1_mean = np.mean(cluster1_rates)
log(f"Cluster 1 (commonly detected) mean detection rate: {cluster1_mean:.1f}%")
log(f"  Compounds: {', '.join(cluster1_comps)}")

# Cluster 2: 4 never-detected
cluster2_comps = ['11Cl-PF3OUdS', 'PFEESA', 'PFTA', 'PFTrDA']
cluster2_rates = [det_matrix[c].mean() * 100 for c in cluster2_comps if c in det_matrix.columns]
log(f"Cluster 2 (never/rarely detected) rates: {[round(r,3) for r in cluster2_rates]}")
log(f"  Compounds: {', '.join([c for c in cluster2_comps if c in det_matrix.columns])}")

# Cluster 3: PFMPA, PFMBA
cluster3_comps = ['PFMPA', 'PFMBA']
cluster3_rates = [det_matrix[c].mean() * 100 for c in cluster3_comps if c in det_matrix.columns]
log(f"Cluster 3 (very rare) rates: {[round(r,3) for r in cluster3_rates]}")
log(f"  Compounds: {', '.join([c for c in cluster3_comps if c in det_matrix.columns])}")

# ============================================================
# MCA QUALITY METRICS
# ============================================================
log("\n--- MCA Quality Metrics ---")

# Calculate total inertia explained by first 3 dimensions
total_inertia_3d = cumulative_inertia[2] * 100
log(f"Cumulative inertia (first 3 dimensions): {total_inertia_3d:.2f}%")

# Calculate average row/column quality (contribution)
avg_row_quality = row_coords.iloc[:, :3].pow(2).sum(axis=1).mean()
log(f"Average row quality (first 3 dimensions): {avg_row_quality:.4f}")

# ============================================================
# SUMMARY AND VERIFICATION
# ============================================================
log("\n" + "=" * 100)
log("VERIFICATION SUMMARY - Phase 3 (MCA Enhanced)")
log("=" * 100)

n_checks = len(results)
n_match = sum(1 for v in results.values() if v['status'] == 'MATCH')
n_mismatch = sum(1 for v in results.values() if v['status'] == 'MISMATCH')
log(f"Total checks: {n_checks}")
log(f"  MATCH:    {n_match}")
log(f"  MISMATCH: {n_mismatch}")

if n_mismatch > 0:
    log("\nMISMATCHES:")
    for k, v in results.items():
        if v['status'] == 'MISMATCH':
            log(f"  {k}: computed={v['computed']}, manuscript={v['manuscript']}")

log("\n" + "=" * 100)
log("OUTPUT FILES GENERATED")
log("=" * 100)
log(f"1. {scree_output}")
log(f"   - Inertia/variance explained by each MCA dimension")
log(f"2. {loadings_output}")
log(f"   - Column coordinates (loadings) for all compounds across dimensions")
log(f"3. {bootstrap_output}")
log(f"   - Bootstrap confidence intervals for loadings (n={n_bootstrap})")
log(f"4. {cooccurrence_output}")
log(f"   - Regulated PFAS co-detection profiles with Wilson CIs")
log(f"5. {comp_dist_output}")
log(f"   - Compound count distribution with percentages")

log("\n" + "=" * 100)
log("ANALYSIS SUMMARY")
log("=" * 100)
log(f"Analysis date: {datetime.now().isoformat()}")
log(f"PWS analyzed: {n_pws:,}")
log(f"Compounds: {len(compounds)}")
log(f"Detection matrix: {det_matrix.shape[0]:,} × {det_matrix.shape[1]}")
log(f"Systems with detection: {n_det_sys:,} ({n_det_sys/n_pws*100:.1f}%)")
log(f"MCA dimensions extracted: {mca.components_.shape[0]}")
log(f"Cumulative inertia (first 3 dims): {total_inertia_3d:.2f}%")
log(f"Bootstrap iterations: {n_bootstrap:,}")

# Save log file
log_file = os.path.join(OUTPUT_DIR, "mca_analysis_session_log.txt")
with open(log_file, 'w') as f:
    f.write('\n'.join(log_lines))
log(f"\nComplete session log saved to {log_file}")

# Save results
results_file = os.path.join(OUTPUT_DIR, "mca_verification_results.json")
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)
log(f"Verification results saved to {results_file}")

log("\n" + "=" * 100)
log("ENHANCED MCA ANALYSIS COMPLETE")
log("=" * 100)

