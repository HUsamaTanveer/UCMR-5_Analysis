#!/usr/bin/env python3
"""
PFAS UCMR 5 Enhanced Temporal Analysis - Phase 2.0
===================================================
Enhanced version with:
- Fixed Mann-Kendall (pymannkendall.original_test)
- Fixed Cochran-Armitage trend test
- Seasonal Kruskal-Wallis with post-hoc Dunn's test
- Wilson confidence intervals for proportions
- Comprehensive logging and verification
"""

import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import json
import os
import sys
import warnings

warnings.filterwarnings('ignore')

# Configuration
BASE = "/sessions/optimistic-determined-feynman/mnt/5_PFAS_Research_Finalization"
RAW_FILE = os.path.join(BASE, "01_Raw_Data", "UCMR5_Jan 2025 Update", "UCMR5_All.txt")
OUTPUT_DIR = os.path.join(BASE, "3_Jan2026_Update_Output_CSVs")

MCL_UGL = {
    "PFOS": 0.004,
    "PFOA": 0.004,
    "PFHxS": 0.010,
    "PFNA": 0.010,
    "HFPO-DA": 0.010
}

# Try to import pymannkendall
try:
    import pymannkendall as pmk
    HAS_PMK = True
except ImportError:
    HAS_PMK = False
    print("WARNING: pymannkendall not available, will use fallback implementation")

# Try to import scikit_posthocs for Dunn's test
try:
    import scikit_posthocs as sp
    HAS_POSTHOCS = True
except ImportError:
    HAS_POSTHOCS = False
    print("WARNING: scikit_posthocs not available, will use manual Dunn's test")

results = {}
log_lines = []

# ============================================================
# LOGGING AND UTILITY FUNCTIONS
# ============================================================

def log(msg):
    """Print and store log message"""
    print(msg)
    log_lines.append(msg)

def check(label, computed, manuscript, tolerance=0.01):
    """Compare computed vs manuscript value"""
    if isinstance(computed, (int, float)) and isinstance(manuscript, (int, float)):
        if manuscript != 0:
            pct_diff = abs(computed - manuscript) / abs(manuscript) * 100
        else:
            pct_diff = abs(computed - manuscript) * 100
        match = "PASS" if pct_diff < tolerance * 100 else "FAIL"
        log(f"  [{match}] {label}: computed={computed:.4f}, manuscript={manuscript:.4f}, diff={pct_diff:.2f}%")
    else:
        match = "PASS" if str(computed) == str(manuscript) else "FAIL"
        log(f"  [{match}] {label}: computed={computed}, manuscript={manuscript}")
    results[label] = {
        "computed": str(computed),
        "manuscript": str(manuscript),
        "status": match
    }

def wilson_ci(successes, trials, z=1.96):
    """
    Calculate Wilson score confidence interval for proportion.
    Default z=1.96 for 95% CI.
    """
    if trials == 0:
        return (0, 0)
    
    p_hat = successes / trials
    
    denominator = 1 + z**2 / trials
    center = (p_hat + z**2 / (2 * trials)) / denominator
    
    adjustment = z * np.sqrt(p_hat * (1 - p_hat) / trials + z**2 / (4 * trials**2)) / denominator
    
    lower = max(0, center - adjustment)
    upper = min(1, center + adjustment)
    
    return (lower, upper)

def mann_kendall_fallback(data):
    """
    Fallback Mann-Kendall implementation (handles ties).
    Returns: tau, p-value, Sen's slope
    """
    n = len(data)
    s = 0
    
    # Count concordant/discordant pairs
    for i in range(n - 1):
        for j in range(i + 1, n):
            diff = data[j] - data[i]
            if diff > 0:
                s += 1
            elif diff < 0:
                s -= 1
    
    # Tau
    denom = n * (n - 1) / 2
    tau = s / denom if denom != 0 else 0
    
    # Variance (handles ties)
    tied_groups = {}
    for val in data:
        tied_groups[val] = tied_groups.get(val, 0) + 1
    
    g = len(tied_groups)  # number of groups with ties
    t_sum = sum(t * (t - 1) * (2 * t + 5) for t in tied_groups.values())
    
    var_s = (n * (n - 1) * (2 * n + 5) - t_sum) / 18
    
    # p-value
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0
    
    p = 2 * stats.norm.sf(abs(z))
    
    # Sen's slope
    slopes = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            if i != j:
                slopes.append((data[j] - data[i]) / (j - i))
    
    if slopes:
        sen_slope = np.median(slopes)
    else:
        sen_slope = 0
    
    return tau, p, sen_slope

def mann_kendall_test(data):
    """
    Use pymannkendall if available, otherwise fallback.
    Returns: tau, p-value, Sen's slope, slope_ci_lower, slope_ci_upper
    """
    if HAS_PMK:
        try:
            result = pmk.original_test(data)
            # pymannkendall returns: (trend, h, z, p, Tau, s, alternative, slope, intercept)
            tau = result.Tau
            p = result.p
            slope = result.slope
            intercept = result.intercept
            
            # Calculate CI for slope using bootstrap
            n = len(data)
            if n > 2:
                # Simple bootstrap for slope CI
                slopes_boot = []
                np.random.seed(42)
                for _ in range(1000):
                    indices = np.random.choice(n, n, replace=True)
                    data_boot = data[indices]
                    _, _, s_boot = mann_kendall_fallback(data_boot)
                    slopes_boot.append(s_boot)
                slope_ci_lower = np.percentile(slopes_boot, 2.5)
                slope_ci_upper = np.percentile(slopes_boot, 97.5)
            else:
                slope_ci_lower = slope
                slope_ci_upper = slope
            
            return tau, p, slope, slope_ci_lower, slope_ci_upper, intercept
        except Exception as e:
            log(f"  WARNING: pymannkendall failed ({e}), using fallback")
            tau, p, slope = mann_kendall_fallback(data)
            return tau, p, slope, slope, slope, 0
    else:
        tau, p, slope = mann_kendall_fallback(data)
        return tau, p, slope, slope, slope, 0

def cochran_armitage_trend(counts_detected, counts_total, scores=None):
    """
    Cochran-Armitage trend test using contingency table approach.
    
    Args:
        counts_detected: array of detected counts per group
        counts_total: array of total counts per group
        scores: array of scores for groups (default: 0, 1, 2, ...)
    
    Returns:
        z, p_value
    """
    counts_detected = np.array(counts_detected)
    counts_total = np.array(counts_total)
    
    k = len(counts_detected)
    
    if scores is None:
        scores = np.arange(k)
    else:
        scores = np.array(scores)
    
    N = counts_total.sum()
    p_bar = counts_detected.sum() / N
    
    T = np.sum(scores * counts_detected) - p_bar * np.sum(scores * counts_total)
    
    var_T = p_bar * (1 - p_bar) * (np.sum(scores**2 * counts_total) - (np.sum(scores * counts_total))**2 / N)
    
    if var_T == 0:
        z = 0
        p_value = 1.0
    else:
        z = T / np.sqrt(var_T)
        p_value = 2 * stats.norm.sf(abs(z))
    
    return z, p_value, T

def kruskal_wallis_seasonal(detection_data_by_season):
    """
    Seasonal Kruskal-Wallis test.
    
    Args:
        detection_data_by_season: dict with keys 'Winter', 'Spring', 'Summer', 'Fall'
                                  values are lists of detection rates
    
    Returns:
        h_stat, p_value, n_groups
    """
    groups = []
    for season in ['Winter', 'Spring', 'Summer', 'Fall']:
        if season in detection_data_by_season:
            data = detection_data_by_season[season]
            if len(data) > 0:
                groups.append(np.array(data))
    
    if len(groups) < 2:
        return np.nan, 1.0
    
    h_stat, p_value = stats.kruskal(*groups)
    return h_stat, p_value

def dunn_test_bonferroni(data_by_group, groups_list):
    """
    Manual implementation of Dunn's test with Bonferroni correction.
    
    Args:
        data_by_group: dict with group names as keys and data arrays as values
        groups_list: list of group names in order
    
    Returns:
        DataFrame with pairwise comparisons
    """
    n_groups = len(groups_list)
    n_comparisons = n_groups * (n_groups - 1) / 2
    alpha_bonf = 0.05 / n_comparisons
    
    # Rank all data combined
    all_data = []
    group_labels = []
    
    for group in groups_list:
        if group in data_by_group:
            data = np.array(data_by_group[group])
            all_data.extend(data)
            group_labels.extend([group] * len(data))
    
    all_data = np.array(all_data)
    ranks = stats.rankdata(all_data)
    
    # Calculate rank sums per group
    rank_sums = {}
    group_ns = {}
    for i, group in enumerate(group_labels):
        if group not in rank_sums:
            rank_sums[group] = 0
            group_ns[group] = 0
        rank_sums[group] += ranks[i]
        group_ns[group] += 1
    
    N = len(all_data)
    
    # Pairwise comparisons
    results_list = []
    
    for i in range(len(groups_list)):
        for j in range(i + 1, len(groups_list)):
            group1 = groups_list[i]
            group2 = groups_list[j]
            
            if group1 not in rank_sums or group2 not in rank_sums:
                continue
            
            n1 = group_ns[group1]
            n2 = group_ns[group2]
            
            r1 = rank_sums[group1]
            r2 = rank_sums[group2]
            
            # Dunn's z-statistic
            z_denom = np.sqrt((N * (N + 1) / 12) * (1 / n1 + 1 / n2))
            
            if z_denom == 0:
                z_stat = 0
                p_val = 1.0
            else:
                z_stat = abs((r1 / n1) - (r2 / n2)) / z_denom
                p_val = 2 * stats.norm.sf(abs(z_stat))
            
            results_list.append({
                'Group1': group1,
                'Group2': group2,
                'Z': z_stat,
                'p_value': p_val,
                'p_bonf': min(p_val * n_comparisons, 1.0),
                'Significant': p_val < alpha_bonf
            })
    
    return pd.DataFrame(results_list)

# ============================================================
# MAIN ANALYSIS
# ============================================================

log("=" * 80)
log("PFAS UCMR 5 ENHANCED TEMPORAL ANALYSIS")
log(f"Run date: {datetime.now().isoformat()}")
log("=" * 80)

# Load data
log("\n--- Phase 0.0: Loading data ---")
log(f"Raw file: {RAW_FILE}")

df = pd.read_csv(RAW_FILE, sep='\t', encoding='latin-1', low_memory=False)
log(f"Loaded {len(df):,} records")

# Filter to PFAS only
pfas = df[df['Contaminant'] != 'lithium'].copy()
pfas['AnalyticalResultValue'] = pd.to_numeric(pfas['AnalyticalResultValue'], errors='coerce')
pfas['detected'] = pfas['AnalyticalResultsSign'] == '='
pfas['CollectionDate'] = pd.to_datetime(pfas['CollectionDate'], format='%m/%d/%Y', errors='coerce')

# Remove invalid dates
pfas = pfas[pfas['CollectionDate'].notna()].copy()

# Create temporal features
pfas['quarter'] = pfas['CollectionDate'].dt.to_period('Q')
pfas['year_q'] = pfas['CollectionDate'].dt.year.astype(str) + 'Q' + pfas['CollectionDate'].dt.quarter.astype(str)
pfas['month'] = pfas['CollectionDate'].dt.month
pfas['year'] = pfas['CollectionDate'].dt.year

# Seasonal assignment (Northern Hemisphere)
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:  # 9, 10, 11
        return 'Fall'

pfas['season'] = pfas['month'].apply(get_season)

# MCL exceedance
pfas['contaminant_clean'] = pfas['Contaminant'].str.strip()
pfas['mcl'] = pfas['contaminant_clean'].map(MCL_UGL)
pfas['mcl_exceedance'] = (pfas['AnalyticalResultValue'] > pfas['mcl']).astype(int)

log(f"Date range: {pfas['CollectionDate'].min().date()} to {pfas['CollectionDate'].max().date()}")
log(f"Unique quarters: {pfas['year_q'].nunique()}")
log(f"Unique seasons: {pfas['season'].nunique()}")
log(f"PFAS records: {len(pfas):,}")

# ============================================================
# PHASE 1.1: QUARTERLY DETECTION RATES
# ============================================================
log("\n--- Phase 1.1: Quarterly Detection Rates ---")

quarterly = pfas.groupby('year_q').agg(
    n_samples=('detected', 'count'),
    n_detected=('detected', 'sum')
).reset_index()
quarterly = quarterly.sort_values('year_q').reset_index(drop=True)
quarterly['det_rate'] = quarterly['n_detected'] / quarterly['n_samples']

# Wilson CIs
quarterly[['ci_lower', 'ci_upper']] = quarterly.apply(
    lambda row: pd.Series(wilson_ci(row['n_detected'], row['n_samples'])),
    axis=1
)

log(f"Quarters analyzed: {len(quarterly)}")
for _, row in quarterly.iterrows():
    log(f"  {row['year_q']}: n={row['n_samples']:,}, detected={row['n_detected']:,}, "
        f"rate={row['det_rate']:.4f}, CI=[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]")

# Save quarterly rates
quarterly_output = quarterly[['year_q', 'n_samples', 'n_detected', 'det_rate', 'ci_lower', 'ci_upper']].copy()
quarterly_output.columns = ['Quarter', 'SampleCount', 'DetectedCount', 'DetectionRate', 'CI_Lower_95', 'CI_Upper_95']
quarterly_output.to_csv(os.path.join(OUTPUT_DIR, 'temporal_quarterly_rates_enhanced.csv'), index=False)
log(f"Saved: temporal_quarterly_rates_enhanced.csv")

# ============================================================
# PHASE 1.2: MANN-KENDALL TREND TEST
# ============================================================
log("\n--- Phase 1.2: Mann-Kendall Trend Test (Fixed) ---")

det_rates = quarterly['det_rate'].values
tau, p_mk, slope, slope_ci_lower, slope_ci_upper, intercept = mann_kendall_test(det_rates)

log(f"Detection rate trend:")
log(f"  Kendall's τ = {tau:.4f}")
log(f"  p-value = {p_mk:.4f}")
log(f"  Sen's slope = {slope:.6f}")
log(f"  Slope CI [95%] = [{slope_ci_lower:.6f}, {slope_ci_upper:.6f}]")
log(f"  Intercept = {intercept:.6f}")

# Save Mann-Kendall results
mk_output = pd.DataFrame({
    'Test': ['Sample Detection Rate'],
    'Tau': [tau],
    'P_Value': [p_mk],
    'Sens_Slope': [slope],
    'Slope_CI_Lower_95': [slope_ci_lower],
    'Slope_CI_Upper_95': [slope_ci_upper],
    'Intercept': [intercept]
})
mk_output.to_csv(os.path.join(OUTPUT_DIR, 'temporal_mannkendall_enhanced.csv'), index=False)
log(f"Saved: temporal_mannkendall_enhanced.csv")

# ============================================================
# PHASE 1.3: COCHRAN-ARMITAGE TREND TEST (Fixed)
# ============================================================
log("\n--- Phase 1.3: Cochran-Armitage Trend Test (Fixed) ---")

ca_z, ca_p, ca_T = cochran_armitage_trend(
    quarterly['n_detected'].values,
    quarterly['n_samples'].values
)

log(f"Cochran-Armitage trend test:")
log(f"  Z-statistic = {ca_z:.4f}")
log(f"  p-value = {ca_p:.4f}")
log(f"  T-statistic = {ca_T:.4f}")
log(f"  Interpretation: {'Significant trend detected' if ca_p < 0.05 else 'No significant trend'}")

# Save Cochran-Armitage results
ca_output = pd.DataFrame({
    'Test': ['Sample Detection Rate Trend'],
    'Z_Statistic': [ca_z],
    'P_Value': [ca_p],
    'T_Statistic': [ca_T],
    'Significant_p05': [ca_p < 0.05]
})
ca_output.to_csv(os.path.join(OUTPUT_DIR, 'temporal_cochran_armitage_enhanced.csv'), index=False)
log(f"Saved: temporal_cochran_armitage_enhanced.csv")

# ============================================================
# PHASE 1.4: MCL EXCEEDANCE QUARTERLY TRENDS
# ============================================================
log("\n--- Phase 1.4: MCL Exceedance Quarterly Analysis ---")

# Filter to records with valid MCL
pfas_mcl = pfas[(pfas['mcl'].notna()) & (pfas['AnalyticalResultValue'].notna())].copy()

quarterly_mcl = pfas_mcl.groupby('year_q').agg(
    n_samples=('mcl_exceedance', 'count'),
    n_exceedance=('mcl_exceedance', 'sum')
).reset_index()
quarterly_mcl = quarterly_mcl.sort_values('year_q').reset_index(drop=True)
quarterly_mcl['exceedance_rate'] = quarterly_mcl['n_exceedance'] / quarterly_mcl['n_samples']

# Wilson CIs for exceedance
quarterly_mcl[['exc_ci_lower', 'exc_ci_upper']] = quarterly_mcl.apply(
    lambda row: pd.Series(wilson_ci(row['n_exceedance'], row['n_samples'])),
    axis=1
)

log(f"Quarters with MCL data: {len(quarterly_mcl)}")
for _, row in quarterly_mcl.iterrows():
    log(f"  {row['year_q']}: n={row['n_samples']:,}, exceedance={row['n_exceedance']:,}, "
        f"rate={row['exceedance_rate']:.4f}, CI=[{row['exc_ci_lower']:.4f}, {row['exc_ci_upper']:.4f}]")

# Mann-Kendall for MCL exceedance
if len(quarterly_mcl) > 2:
    exc_rates = quarterly_mcl['exceedance_rate'].values
    tau_mcl, p_mcl, slope_mcl, slope_ci_l_mcl, slope_ci_u_mcl, intercept_mcl = mann_kendall_test(exc_rates)
    
    log(f"\nMCL Exceedance rate trend:")
    log(f"  Kendall's τ = {tau_mcl:.4f}")
    log(f"  p-value = {p_mcl:.4f}")
    log(f"  Sen's slope = {slope_mcl:.6f}")

# ============================================================
# PHASE 1.5: PRE/POST NPDWR COMPARISON
# ============================================================
log("\n--- Phase 1.5: Pre/Post NPDWR Comparison (Jan 2024) ---")

npdwr_date = pd.Timestamp('2024-01-23')
pfas['pre_post_npdwr'] = pfas['CollectionDate'] < npdwr_date

pre_npdwr = pfas[pfas['pre_post_npdwr']].copy()
post_npdwr = pfas[~pfas['pre_post_npdwr']].copy()

pre_det = pre_npdwr['detected'].sum()
pre_n = len(pre_npdwr)
pre_rate = pre_det / pre_n if pre_n > 0 else 0

post_det = post_npdwr['detected'].sum()
post_n = len(post_npdwr)
post_rate = post_det / post_n if post_n > 0 else 0

# Chi-square test
contingency = [[pre_det, pre_n - pre_det], [post_det, post_n - post_det]]
chi2, p_chi2, dof, expected = stats.chi2_contingency(contingency)

log(f"Pre-NPDWR (before {npdwr_date.date()}):")
log(f"  n={pre_n:,}, detected={pre_det:,}, rate={pre_rate:.4f}")

log(f"Post-NPDWR (from {npdwr_date.date()}):")
log(f"  n={post_n:,}, detected={post_det:,}, rate={post_rate:.4f}")

log(f"Chi-square test: χ²={chi2:.4f}, p={p_chi2:.4f}")

# Wilson CIs
pre_ci = wilson_ci(pre_det, pre_n)
post_ci = wilson_ci(post_det, post_n)

pre_post_output = pd.DataFrame({
    'Period': ['Pre-NPDWR (before 2024-01-23)', 'Post-NPDWR (from 2024-01-23)'],
    'SampleCount': [pre_n, post_n],
    'DetectedCount': [pre_det, post_det],
    'DetectionRate': [pre_rate, post_rate],
    'CI_Lower_95': [pre_ci[0], post_ci[0]],
    'CI_Upper_95': [pre_ci[1], post_ci[1]]
})
pre_post_output.to_csv(os.path.join(OUTPUT_DIR, 'temporal_pre_post_npdwr_enhanced.csv'), index=False)
log(f"Saved: temporal_pre_post_npdwr_enhanced.csv")

# ============================================================
# PHASE 1.6: COHORT ANALYSIS (First sampling by location)
# ============================================================
log("\n--- Phase 1.6: Cohort Analysis (First Sampling Year) ---")

cohort_data = pfas.groupby(['SamplePointID', 'detected']).agg({
    'CollectionDate': 'min'
}).reset_index()
cohort_data['first_year'] = cohort_data['CollectionDate'].dt.year

first_sample = pfas.groupby('SamplePointID').agg({
    'CollectionDate': 'min'
}).reset_index()
first_sample.columns = ['SamplePointID', 'FirstSampleDate']
first_sample['FirstYear'] = first_sample['FirstSampleDate'].dt.year

# Detection at first sample
first_sample_det = pfas.groupby('SamplePointID').apply(
    lambda grp: grp[grp['CollectionDate'] == grp['CollectionDate'].min()]['detected'].sum() > 0
).reset_index()
first_sample_det.columns = ['SamplePointID', 'DetectedAtFirst']

first_sample = first_sample.merge(first_sample_det, on='SamplePointID')

log(f"Unique sampling locations: {first_sample['SamplePointID'].nunique()}")

# Summarize by cohort
cohort_summary = first_sample.groupby('FirstYear').agg(
    LocationCount=('SamplePointID', 'count'),
    DetectedLocations=('DetectedAtFirst', 'sum')
).reset_index()
cohort_summary['DetectionRate'] = cohort_summary['DetectedLocations'] / cohort_summary['LocationCount']

log("Detection rate by first sampling year (cohort):")
for _, row in cohort_summary.iterrows():
    log(f"  {int(row['FirstYear'])}: {int(row['LocationCount'])} locations, "
        f"{int(row['DetectedLocations'])} detected, rate={row['DetectionRate']:.4f}")

# ============================================================
# PHASE 1.7: SEASONAL KRUSKAL-WALLIS AND POST-HOC ANALYSIS
# ============================================================
log("\n--- Phase 1.7: Seasonal Analysis (Kruskal-Wallis) ---")

# Aggregate detection rate by season and quarter
seasonal_quarterly = pfas.groupby(['year_q', 'season']).agg(
    n_samples=('detected', 'count'),
    n_detected=('detected', 'sum')
).reset_index()
seasonal_quarterly['det_rate'] = seasonal_quarterly['n_detected'] / seasonal_quarterly['n_samples']

# Collect rates by season
detection_by_season = {season: [] for season in ['Winter', 'Spring', 'Summer', 'Fall']}

for season in ['Winter', 'Spring', 'Summer', 'Fall']:
    season_data = seasonal_quarterly[seasonal_quarterly['season'] == season]
    detection_by_season[season] = season_data['det_rate'].values.tolist()

log("Detection rates by season (quarterly aggregates):")
for season in ['Winter', 'Spring', 'Summer', 'Fall']:
    rates = detection_by_season[season]
    if rates:
        mean_rate = np.mean(rates)
        std_rate = np.std(rates)
        log(f"  {season}: mean={mean_rate:.4f}, std={std_rate:.4f}, n_quarters={len(rates)}")

# Kruskal-Wallis test
h_stat, p_kw = kruskal_wallis_seasonal(detection_by_season)

log(f"\nKruskal-Wallis test (seasonal detection rates):")
log(f"  H-statistic = {h_stat:.4f}")
log(f"  p-value = {p_kw:.4f}")
log(f"  Significant (p<0.05): {p_kw < 0.05}")

# Post-hoc Dunn's test if significant
if p_kw < 0.05:
    log(f"\nPost-hoc Dunn's test (Bonferroni correction):")
    
    # Prepare data for Dunn's test
    detection_data_for_dunn = {}
    for season in ['Winter', 'Spring', 'Summer', 'Fall']:
        detection_data_for_dunn[season] = np.array(detection_by_season[season])
    
    dunn_results = dunn_test_bonferroni(detection_data_for_dunn, ['Winter', 'Spring', 'Summer', 'Fall'])
    
    for _, row in dunn_results.iterrows():
        sig_marker = "*" if row['Significant'] else ""
        log(f"  {row['Group1']} vs {row['Group2']}: z={row['Z']:.4f}, p={row['p_value']:.4f}, "
            f"p_bonf={row['p_bonf']:.4f}{sig_marker}")
    
    # Save Dunn's results
    dunn_results.to_csv(os.path.join(OUTPUT_DIR, 'temporal_seasonal_dunn_posthoc.csv'), index=False)
    log(f"Saved: temporal_seasonal_dunn_posthoc.csv")

# Save seasonal Kruskal-Wallis summary
seasonal_summary = pd.DataFrame({
    'Season': ['Winter', 'Spring', 'Summer', 'Fall'],
    'MeanDetectionRate': [
        np.mean(detection_by_season['Winter']) if detection_by_season['Winter'] else np.nan,
        np.mean(detection_by_season['Spring']) if detection_by_season['Spring'] else np.nan,
        np.mean(detection_by_season['Summer']) if detection_by_season['Summer'] else np.nan,
        np.mean(detection_by_season['Fall']) if detection_by_season['Fall'] else np.nan
    ]
})
seasonal_kw = pd.DataFrame({
    'Test': ['Seasonal Detection Rates (Kruskal-Wallis)'],
    'H_Statistic': [h_stat],
    'P_Value': [p_kw],
    'Significant_p05': [p_kw < 0.05]
})
seasonal_kw.to_csv(os.path.join(OUTPUT_DIR, 'temporal_seasonal_kruskal_wallis.csv'), index=False)
log(f"Saved: temporal_seasonal_kruskal_wallis.csv")

# ============================================================
# PHASE 1.8: PEAK QUARTER ANALYSIS
# ============================================================
log("\n--- Phase 1.8: Peak Quarter Analysis ---")

peak_quarter = quarterly.loc[quarterly['det_rate'].idxmax()]
peak_date = peak_quarter['year_q']

log(f"Peak detection quarter: {peak_date}")
log(f"  Detection rate: {peak_quarter['det_rate']:.4f}")
log(f"  Sample count: {peak_quarter['n_samples']:,}")
log(f"  Detected samples: {peak_quarter['n_detected']:,}")

# ============================================================
# SUMMARY AND VERIFICATION
# ============================================================
log("\n" + "=" * 80)
log("SUMMARY OF RESULTS")
log("=" * 80)

log(f"\nQuarterly Analysis:")
log(f"  Total quarters: {len(quarterly)}")
log(f"  Date range: {quarterly['year_q'].min()} to {quarterly['year_q'].max()}")
log(f"  Overall detection rate: {quarterly['n_detected'].sum() / quarterly['n_samples'].sum():.4f}")

log(f"\nTrend Tests:")
log(f"  Mann-Kendall τ: {tau:.4f} (p={p_mk:.4f})")
log(f"  Cochran-Armitage Z: {ca_z:.4f} (p={ca_p:.4f})")

log(f"\nSeasonal Analysis:")
log(f"  Kruskal-Wallis H: {h_stat:.4f} (p={p_kw:.4f})")
log(f"  Seasonal effect: {'Yes (p<0.05)' if p_kw < 0.05 else 'No (p≥0.05)'}")

log(f"\nPre/Post NPDWR (2024-01-23):")
log(f"  Pre-NPDWR rate: {pre_rate:.4f}")
log(f"  Post-NPDWR rate: {post_rate:.4f}")
log(f"  Chi-square test p: {p_chi2:.4f}")

log(f"\nOutput files saved to: {OUTPUT_DIR}")
log(f"  - temporal_quarterly_rates_enhanced.csv")
log(f"  - temporal_mannkendall_enhanced.csv")
log(f"  - temporal_cochran_armitage_enhanced.csv")
log(f"  - temporal_seasonal_kruskal_wallis.csv")
log(f"  - temporal_seasonal_dunn_posthoc.csv (if significant)")
log(f"  - temporal_pre_post_npdwr_enhanced.csv")

log("\n" + "=" * 80)
log("ANALYSIS COMPLETE")
log("=" * 80)

# Save log
log_file = os.path.join(OUTPUT_DIR, 'temporal_analysis_enhanced.log')
with open(log_file, 'w') as f:
    f.write('\n'.join(log_lines))

print(f"\nLog saved to: {log_file}")
