"""
Enhanced Core Statistics Script (Script 01)
============================================
Adds Wilson confidence intervals to all detection rates and bootstrap CIs for other statistics.

Manuscript verification:
- Total analytical results: 1,863,306
- Study population: 10,297 PWS
- 66 jurisdictions
- 29 PFAS compounds
- Systems with any detection: 3,539 (34.4%)
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')

# Configuration
BASE = "/sessions/optimistic-determined-feynman/mnt/5_PFAS_Research_Finalization"
RAW_FILE = os.path.join(BASE, "01_Raw_Data", "UCMR5_Jan 2025 Update", "UCMR5_All.txt")
OUTPUT_DIR = os.path.join(BASE, "3_Jan2026_Update_Output_CSVs")

# Create output directory if needed
os.makedirs(OUTPUT_DIR, exist_ok=True)

# MCL thresholds (µg/L)
MCL_THRESHOLDS = {
    'PFOS': 0.004,
    'PFOA': 0.004,
    'PFHxS': 0.010,
    'PFNA': 0.010,
    'HFPO-DA': 0.010
}

# HBWC thresholds for HI (µg/L)
HBWC_HI = {
    'PFHxS': 0.010,
    'PFNA': 0.010,
    'HFPO-DA': 0.010,
    'PFBS': 2.0
}

print("=" * 80)
print("ENHANCED CORE STATISTICS ANALYSIS - Script 01")
print("=" * 80)

# ============================================================================
# 1. Load and Process Raw Data
# ============================================================================
print("\n[1] Loading raw data...")
df = pd.read_csv(RAW_FILE, sep='\t', encoding='latin-1', low_memory=False)
print(f"    Total rows loaded: {len(df):,}")

# Filter out lithium
df = df[df['Contaminant'] != 'lithium'].copy()
print(f"    After filtering lithium: {len(df):,}")

# Define detection: AnalyticalResultsSign == '='
df['IsDetected'] = (df['AnalyticalResultsSign'] == '=').astype(int)

print(f"    Total analytical results (after filtering): {len(df):,}")
print(f"    Total detected results: {df['IsDetected'].sum():,}")

# ============================================================================
# 2. Define Wilson Confidence Interval Function
# ============================================================================
def wilson_ci(x, n, z=1.96):
    """
    Wilson score interval for binomial proportion.
    """
    if n == 0:
        return 0.0, 0.0, 0.0
    
    p_hat = x / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2*n)) / denom
    margin = z * np.sqrt(p_hat*(1-p_hat)/n + z**2/(4*n**2)) / denom
    
    lower = max(0, center - margin)
    upper = min(1, center + margin)
    
    return p_hat, lower, upper

# ============================================================================
# 3. Calculate Overall System Detection Rate with Wilson CI
# ============================================================================
print("\n[2] Calculating overall system detection rates...")

# Get unique PWS with any detection
pws_with_any_detection = df[df['IsDetected'] == 1]['PWSID'].nunique()
total_pws = df['PWSID'].nunique()
overall_detection_rate = pws_with_any_detection / total_pws

point, ci_lo, ci_hi = wilson_ci(pws_with_any_detection, total_pws)
print(f"    Systems with any PFAS detection: {pws_with_any_detection:,} / {total_pws:,}")
print(f"    Overall detection rate: {overall_detection_rate:.1%}")
print(f"    Wilson 95% CI: ({ci_lo:.1%}, {ci_hi:.1%})")

# ============================================================================
# 4. Build Table 1: Compound-Level Statistics with CIs
# ============================================================================
print("\n[3] Building Table 1: Compound-level statistics...")

compounds = sorted(df['Contaminant'].unique())
n_compounds = len(compounds)
print(f"    Total PFAS compounds: {n_compounds}")

table1_data = []

for compound in compounds:
    df_compound = df[df['Contaminant'] == compound].copy()
    
    # PWS-level detection
    pws_detected = df_compound[df_compound['IsDetected'] == 1]['PWSID'].nunique()
    pws_total = df_compound['PWSID'].nunique()
    
    # Wilson CI for PWS detection
    point_det, ci_det_lo, ci_det_hi = wilson_ci(pws_detected, pws_total)
    pws_det_pct = 100 * point_det
    
    # MCL exceedance
    if compound in MCL_THRESHOLDS:
        mcl = MCL_THRESHOLDS[compound]
        df_exceeded = df_compound[
            (df_compound['IsDetected'] == 1) & 
            (df_compound['AnalyticalResultValue'] > mcl)
        ]
        pws_exceeded = df_exceeded['PWSID'].nunique()
        point_exc, ci_exc_lo, ci_exc_hi = wilson_ci(pws_exceeded, pws_total)
        pws_exceed_pct = 100 * point_exc
    else:
        pws_exceeded = None
        pws_exceed_pct = None
        ci_exc_lo = None
        ci_exc_hi = None
    
    # Maximum concentration
    detected_concs = df_compound[df_compound['IsDetected'] == 1]['AnalyticalResultValue']
    max_conc = detected_concs.max() if len(detected_concs) > 0 else None
    
    table1_data.append({
        'Compound': compound,
        'n_systems': pws_total,
        'pws_det_n': pws_detected,
        'pws_det_pct': pws_det_pct,
        'det_ci_lo': ci_det_lo * 100,
        'det_ci_hi': ci_det_hi * 100,
        'pws_exceed_n': pws_exceeded,
        'pws_exceed_pct': pws_exceed_pct,
        'exceed_ci_lo': ci_exc_lo * 100 if ci_exc_lo is not None else None,
        'exceed_ci_hi': ci_exc_hi * 100 if ci_exc_hi is not None else None,
        'max_conc_ngl': max_conc * 1000 if max_conc is not None else None
    })

table1_df = pd.DataFrame(table1_data)
table1_file = os.path.join(OUTPUT_DIR, 'core_statistics_table1_enhanced.csv')
table1_df.to_csv(table1_file, index=False)
print(f"    Table 1 saved: {table1_file}")
print(f"    Sample (first 3 compounds):")
print(table1_df.head(3).to_string(index=False))

# ============================================================================
# 5. System Size Analysis with Odds Ratio and CI
# ============================================================================
print("\n[4] Calculating system size analysis with odds ratio CIs...")

df['PWS_Size'] = df['Size'].map({'L': 'Large', 'S': 'Small'})

large_systems = df[df['PWS_Size'] == 'Large']['PWSID'].unique()
small_systems = df[df['PWS_Size'] == 'Small']['PWSID'].unique()

print(f"    Large systems: {len(large_systems):,}")
print(f"    Small systems: {len(small_systems):,}")

large_detected = df[(df['PWS_Size'] == 'Large') & (df['IsDetected'] == 1)]['PWSID'].nunique()
small_detected = df[(df['PWS_Size'] == 'Small') & (df['IsDetected'] == 1)]['PWSID'].nunique()

print(f"    Large systems with detection: {large_detected:,} / {len(large_systems):,}")
print(f"    Small systems with detection: {small_detected:,} / {len(small_systems):,}")

large_rate = large_detected / len(large_systems)
small_rate = small_detected / len(small_systems)

print(f"    Large system detection rate: {large_rate:.1%}")
print(f"    Small system detection rate: {small_rate:.1%}")

_, large_ci_lo, large_ci_hi = wilson_ci(large_detected, len(large_systems))
_, small_ci_lo, small_ci_hi = wilson_ci(small_detected, len(small_systems))

print(f"    Large system 95% CI: ({large_ci_lo:.1%}, {large_ci_hi:.1%})")
print(f"    Small system 95% CI: ({small_ci_lo:.1%}, {small_ci_hi:.1%})")

# Odds ratio
a = large_detected
b = len(large_systems) - large_detected
c = small_detected
d = len(small_systems) - small_detected

odds_ratio = (a * d) / (b * c)
log_or = np.log(odds_ratio)
se_log_or = np.sqrt(1/a + 1/b + 1/c + 1/d)
or_ci_lo = np.exp(log_or - 1.96 * se_log_or)
or_ci_hi = np.exp(log_or + 1.96 * se_log_or)

print(f"    Odds Ratio (Large vs Small): {odds_ratio:.4f}")
print(f"    Odds Ratio 95% CI: ({or_ci_lo:.4f}, {or_ci_hi:.4f})")

system_size_data = [
    {
        'System_Size': 'Large',
        'n_systems': len(large_systems),
        'n_detected': large_detected,
        'detection_rate': large_rate * 100,
        'ci_lo': large_ci_lo * 100,
        'ci_hi': large_ci_hi * 100,
        'odds_ratio': odds_ratio,
        'or_ci_lo': or_ci_lo,
        'or_ci_hi': or_ci_hi
    },
    {
        'System_Size': 'Small',
        'n_systems': len(small_systems),
        'n_detected': small_detected,
        'detection_rate': small_rate * 100,
        'ci_lo': small_ci_lo * 100,
        'ci_hi': small_ci_hi * 100,
        'odds_ratio': 1.0,
        'or_ci_lo': None,
        'or_ci_hi': None
    }
]

system_size_df = pd.DataFrame(system_size_data)
system_size_file = os.path.join(OUTPUT_DIR, 'core_statistics_system_size_enhanced.csv')
system_size_df.to_csv(system_size_file, index=False)
print(f"    System size analysis saved: {system_size_file}")

# ============================================================================
# 6. Source Water Type Analysis with CIs
# ============================================================================
print("\n[5] Calculating source water type analysis...")

df['SourceWater'] = df['FacilityWaterType'].map({'SW': 'Surface', 'GW': 'Groundwater'})

source_types = [s for s in df['SourceWater'].unique() if pd.notna(s)]
source_data = []

for source in sorted(source_types):
    df_source = df[df['SourceWater'] == source]
    systems_source = df_source['PWSID'].nunique()
    detected_source = df_source[df_source['IsDetected'] == 1]['PWSID'].nunique()
    
    rate = detected_source / systems_source
    _, ci_lo, ci_hi = wilson_ci(detected_source, systems_source)
    
    source_data.append({
        'Source_Water': source,
        'n_pws': systems_source,
        'n_detected': detected_source,
        'detection_rate': rate * 100,
        'ci_lo': ci_lo * 100,
        'ci_hi': ci_hi * 100
    })
    
    print(f"    {source} water: {detected_source:,} / {systems_source:,} systems ({rate:.1%})")
    print(f"      95% CI: ({ci_lo:.1%}, {ci_hi:.1%})")

source_water_df = pd.DataFrame(source_data)
source_water_file = os.path.join(OUTPUT_DIR, 'core_statistics_source_water_enhanced.csv')
source_water_df.to_csv(source_water_file, index=False)
print(f"    Source water analysis saved: {source_water_file}")

# ============================================================================
# 7. State-Level Analysis with Wilson CIs
# ============================================================================
print("\n[6] Calculating state-level analysis...")

states = sorted(df['State'].unique())
n_states = len(states)
print(f"    Total jurisdictions: {n_states}")

geographic_data = []

for state in states:
    df_state = df[df['State'] == state]
    systems_state = df_state['PWSID'].nunique()
    detected_state = df_state[df_state['IsDetected'] == 1]['PWSID'].nunique()
    
    if systems_state > 0:
        rate = detected_state / systems_state
        _, ci_lo, ci_hi = wilson_ci(detected_state, systems_state)
        
        geographic_data.append({
            'State': state,
            'n_pws': systems_state,
            'n_detected': detected_state,
            'detection_rate': rate * 100,
            'ci_lo': ci_lo * 100,
            'ci_hi': ci_hi * 100
        })

geographic_df = pd.DataFrame(geographic_data)
geographic_file = os.path.join(OUTPUT_DIR, 'core_statistics_geographic_enhanced.csv')
geographic_df.to_csv(geographic_file, index=False)
print(f"    Geographic analysis saved: {geographic_file}")
print(f"    Sample (first 5 states):")
print(geographic_df.head().to_string(index=False))

# ============================================================================
# 8. Cramér's V with Bootstrap CI
# ============================================================================
print("\n[7] Calculating Cramér's V with bootstrap CI...")

def cramers_v_fast(state_arr, detect_arr):
    """Fast Cramér's V calculation"""
    contingency = pd.crosstab(pd.Series(state_arr, name='State'), 
                             pd.Series(detect_arr, name='Detected'))
    chi2, _, _, _ = chi2_contingency(contingency)
    n = len(state_arr)
    min_dim = min(contingency.shape[0] - 1, contingency.shape[1] - 1)
    if min_dim == 0:
        return 0
    return np.sqrt(chi2 / (n * min_dim))

state_arr = df['State'].values
detect_arr = df['IsDetected'].values
original_cramers = cramers_v_fast(state_arr, detect_arr)
print(f"    Original Cramér's V: {original_cramers:.4f}")

# Bootstrap with small n
np.random.seed(42)
n_bootstrap = 50
bootstrap_cramers = []

for i in range(n_bootstrap):
    idx = np.random.choice(len(df), size=len(df), replace=True)
    v_boot = cramers_v_fast(state_arr[idx], detect_arr[idx])
    bootstrap_cramers.append(v_boot)
    if (i + 1) % 10 == 0:
        print(f"      Bootstrap {i+1}/{n_bootstrap}")

bootstrap_cramers = np.array(bootstrap_cramers)
cramers_ci_lo = np.percentile(bootstrap_cramers, 2.5)
cramers_ci_hi = np.percentile(bootstrap_cramers, 97.5)

print(f"    Bootstrap 95% CI: ({cramers_ci_lo:.4f}, {cramers_ci_hi:.4f})")

cramers_df = pd.DataFrame({
    'Statistic': ['Cramers_V'],
    'Point_Estimate': [original_cramers],
    'CI_Lower': [cramers_ci_lo],
    'CI_Upper': [cramers_ci_hi],
    'Bootstrap_N': [n_bootstrap]
})

cramers_file = os.path.join(OUTPUT_DIR, 'core_statistics_cramers_v_bootstrap.csv')
cramers_df.to_csv(cramers_file, index=False)
print(f"    Cramér's V bootstrap saved: {cramers_file}")

# ============================================================================
# 9. MCL Exceedance Analysis
# ============================================================================
print("\n[8] MCL exceedance analysis with Wilson CIs...")

mcl_data = []
for compound, mcl in MCL_THRESHOLDS.items():
    df_compound = df[df['Contaminant'] == compound]
    
    pws_total = df_compound['PWSID'].nunique()
    df_exceeded = df_compound[
        (df_compound['IsDetected'] == 1) & 
        (df_compound['AnalyticalResultValue'] > mcl)
    ]
    pws_exceeded = df_exceeded['PWSID'].nunique()
    
    rate = pws_exceeded / pws_total
    _, ci_lo, ci_hi = wilson_ci(pws_exceeded, pws_total)
    
    mcl_data.append({
        'Compound': compound,
        'MCL_ug_L': mcl,
        'n_systems': pws_total,
        'exceed_n': pws_exceeded,
        'exceed_rate': rate * 100,
        'exceed_ci_lo': ci_lo * 100,
        'exceed_ci_hi': ci_hi * 100
    })
    
    print(f"    {compound}: {pws_exceeded:,} / {pws_total:,} ({rate:.1%})")

mcl_df = pd.DataFrame(mcl_data)
mcl_file = os.path.join(OUTPUT_DIR, 'core_statistics_mcl_enhanced.csv')
mcl_df.to_csv(mcl_file, index=False)
print(f"    MCL analysis saved: {mcl_file}")

# ============================================================================
# 10. Hazard Index (HI) Analysis
# ============================================================================
print("\n[9] Hazard Index (HI) analysis...")

pws_list = df['PWSID'].unique()
hi_dict = {pws: 0.0 for pws in pws_list}

for compound, hbwc in HBWC_HI.items():
    df_compound = df[df['Contaminant'] == compound]
    detected_comp = df_compound[df_compound['IsDetected'] == 1]
    if len(detected_comp) > 0:
        max_per_pws = detected_comp.groupby('PWSID')['AnalyticalResultValue'].max()
        for pws, conc in max_per_pws.items():
            if pd.notna(conc):
                hi_dict[pws] += conc / hbwc

hi_values = np.array(list(hi_dict.values()))

# Bootstrap for mean HI
bootstrap_hi_means = []
for i in range(n_bootstrap):
    hi_boot = np.random.choice(hi_values, size=len(hi_values), replace=True)
    bootstrap_hi_means.append(hi_boot.mean())

bootstrap_hi_means = np.array(bootstrap_hi_means)
mean_hi = hi_values.mean()
hi_ci_lo = np.percentile(bootstrap_hi_means, 2.5)
hi_ci_hi = np.percentile(bootstrap_hi_means, 97.5)

hi_gt1 = (hi_values > 1).sum()
hi_gt0_5 = ((hi_values > 0.5) & (hi_values <= 1)).sum()
hi_le0_5 = (hi_values <= 0.5).sum()

print(f"    Mean HI: {mean_hi:.4f}")
print(f"    Bootstrap 95% CI for mean HI: ({hi_ci_lo:.4f}, {hi_ci_hi:.4f})")
print(f"    Systems with HI > 1.0: {hi_gt1:,} ({100*hi_gt1/len(hi_values):.1f}%)")
print(f"    Systems with HI 0.5-1.0: {hi_gt0_5:,} ({100*hi_gt0_5/len(hi_values):.1f}%)")
print(f"    Systems with HI < 0.5: {hi_le0_5:,} ({100*hi_le0_5/len(hi_values):.1f}%)")

hi_summary_df = pd.DataFrame({
    'Statistic': ['Mean_HI', 'HI_gt_1', 'HI_0p5_to_1', 'HI_le_0p5'],
    'Value': [mean_hi, hi_gt1, hi_gt0_5, hi_le0_5],
    'Percent': [None, 100*hi_gt1/len(hi_values), 100*hi_gt0_5/len(hi_values), 100*hi_le0_5/len(hi_values)],
    'CI_Lower': [hi_ci_lo, None, None, None],
    'CI_Upper': [hi_ci_hi, None, None, None]
})

hi_file = os.path.join(OUTPUT_DIR, 'core_statistics_hi_enhanced.csv')
hi_summary_df.to_csv(hi_file, index=False)
print(f"    HI analysis saved: {hi_file}")

# ============================================================================
# 11. Co-Occurrence Analysis
# ============================================================================
print("\n[10] Co-occurrence analysis...")

detected_compounds_per_system = df[df['IsDetected'] == 1].groupby('PWSID')['Contaminant'].nunique()
all_systems_count = df['PWSID'].nunique()

compound_counts = detected_compounds_per_system.value_counts().sort_index()

print(f"    Systems with detection:")
for n_comp, count in compound_counts.items():
    pct = 100 * count / all_systems_count
    print(f"      {n_comp} compounds: {count:,} systems ({pct:.1f}%)")

if len(detected_compounds_per_system) > 0:
    n_detected_values = detected_compounds_per_system.values
    bootstrap_n_compounds = []
    for i in range(n_bootstrap):
        n_comp_boot = np.random.choice(n_detected_values, size=len(n_detected_values), replace=True)
        bootstrap_n_compounds.append(n_comp_boot.mean())
    
    bootstrap_n_compounds = np.array(bootstrap_n_compounds)
    mean_n_compounds = n_detected_values.mean()
    n_comp_ci_lo = np.percentile(bootstrap_n_compounds, 2.5)
    n_comp_ci_hi = np.percentile(bootstrap_n_compounds, 97.5)
    
    print(f"    Mean compounds per detected system: {mean_n_compounds:.2f}")
    print(f"    Bootstrap 95% CI: ({n_comp_ci_lo:.2f}, {n_comp_ci_hi:.2f})")

# ============================================================================
# 12. Verification
# ============================================================================
print("\n" + "=" * 80)
print("VERIFICATION AGAINST MANUSCRIPT VALUES")
print("=" * 80)

verification_results = {
    'Total analytical results (expected 1,863,306)': len(df),
    'Study population PWS (expected 10,297)': total_pws,
    'Number of jurisdictions (expected 66)': n_states,
    'Number of PFAS compounds (expected 29)': n_compounds,
    'Systems with any detection (expected 3,539)': pws_with_any_detection,
    'Overall detection rate (expected 34.4%)': f"{overall_detection_rate:.1%}",
    'Large system detection rate (expected 42.4%)': f"{large_rate:.1%}",
    'Small system detection rate (expected 28.4%)': f"{small_rate:.1%}",
    'Original Cramers V (expected ~0.308)': f"{original_cramers:.4f}"
}

for metric, value in verification_results.items():
    print(f"  {metric}: {value}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY STATISTICS WITH CONFIDENCE INTERVALS")
print("=" * 80)

print(f"\nOverall Detection Rate:")
print(f"  Point estimate: {overall_detection_rate:.1%}")
print(f"  Wilson 95% CI: ({ci_lo:.1%}, {ci_hi:.1%})")

print(f"\nSystem Size (Odds Ratio):")
print(f"  Large vs Small OR: {odds_ratio:.4f}")
print(f"  95% CI: ({or_ci_lo:.4f}, {or_ci_hi:.4f})")

print(f"\nState × Detection Association (Cramér's V):")
print(f"  Point estimate: {original_cramers:.4f}")
print(f"  Bootstrap 95% CI: ({cramers_ci_lo:.4f}, {cramers_ci_hi:.4f})")

print(f"\nMean Hazard Index (HI):")
print(f"  Point estimate: {mean_hi:.4f}")
print(f"  Bootstrap 95% CI: ({hi_ci_lo:.4f}, {hi_ci_hi:.4f})")

print("\n" + "=" * 80)
print("OUTPUT FILES CREATED")
print("=" * 80)
print(f"  1. {table1_file}")
print(f"  2. {system_size_file}")
print(f"  3. {source_water_file}")
print(f"  4. {geographic_file}")
print(f"  5. {mcl_file}")
print(f"  6. {hi_file}")
print(f"  7. {cramers_file}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

