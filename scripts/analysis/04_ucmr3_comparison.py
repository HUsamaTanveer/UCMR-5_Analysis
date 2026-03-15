"""
UCMR 3 vs 5 Comparison: Enhanced Analysis with McNemar's Test and Wilson CIs
==============================================================================
Compares detection rates between UCMR 3 and UCMR 5 data for matched water systems.
Includes:
- Detection rate calculations with MRL adjustment
- Wilson confidence intervals
- McNemar's exact test for concordance
- Benjamini-Hochberg FDR correction
- Mann-Whitney U test for concentration distributions
- Comprehensive output with verification against manuscript values
"""

import os
import sys
import pandas as pd
import numpy as np
from scipy import stats
from scipy.special import comb
from scipy.stats import chi2
import warnings
warnings.filterwarnings('ignore')

# Configuration
BASE = "/sessions/optimistic-determined-feynman/mnt/5_PFAS_Research_Finalization"
UCMR5_FILE = os.path.join(BASE, "01_Raw_Data", "UCMR5_Jan 2025 Update", "UCMR5_All.txt")
UCMR3_FILE = os.path.join(BASE, "01_Raw_Data", "UCMR3", "UCMR3_All.txt")
OUTPUT_DIR = os.path.join(BASE, "3_Jan2026_Update_Output_CSVs")

# PFAS compounds of interest
SHARED_COMPOUNDS = ['PFOS', 'PFOA', 'PFHxS', 'PFNA', 'PFBS', 'PFHpA']

# UCMR 3 MRLs in µg/L (Maximum Reporting Levels)
UCMR3_MRLS = {
    'PFOS': 0.040,
    'PFOA': 0.020,
    'PFHxS': 0.030,
    'PFNA': 0.020,
    'PFBS': 0.090,
    'PFHpA': 0.010
}

# Known UCMR 3 detection rates from manuscript
MANUSCRIPT_UCMR3_RATES = {
    'PFOS': 0.0193,
    'PFOA': 0.0238,
    'PFHxS': 0.0112,
    'PFNA': 0.0028,
    'PFBS': 0.0016,
    'PFHpA': 0.0175
}


def wilson_ci(successes, n, confidence=0.95):
    """
    Calculate Wilson score confidence interval for a proportion.
    Based on Newcombe (1998) - more accurate than normal approximation.
    
    Parameters:
    -----------
    successes : int
        Number of successes
    n : int
        Total sample size
    confidence : float
        Confidence level (default 0.95 for 95%)
    
    Returns:
    --------
    tuple : (point_estimate, lower_bound, upper_bound)
    """
    if n == 0:
        return (0.0, 0.0, 0.0)
    
    p_hat = successes / n
    z = stats.norm.ppf((1 + confidence) / 2)
    z_sq = z * z
    
    denominator = 1 + z_sq / n
    center = (p_hat + z_sq / (2 * n)) / denominator
    margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z_sq / (4 * n * n)) / denominator
    
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    
    return (p_hat, lower, upper)


def mcnemar_exact(concordance_table):
    """
    Perform McNemar's exact test using binomial distribution.
    Use for small discordant counts (b + c < 25).
    
    Parameters:
    -----------
    concordance_table : dict
        {'both_det': a, 'only_u3': b, 'only_u5': c, 'neither': d}
    
    Returns:
    --------
    dict : {'test_stat': value, 'p_value': value, 'method': 'exact_binomial'}
    """
    b = concordance_table['only_u3']
    c = concordance_table['only_u5']
    total_discordant = b + c
    
    if total_discordant == 0:
        return {
            'test_stat': 0.0,
            'p_value': 1.0,
            'method': 'exact_binomial',
            'note': 'No discordant pairs'
        }
    
    # Two-tailed exact binomial test
    result = stats.binomtest(min(b, c), total_discordant, 0.5, alternative='two-sided')
    p_value = result.pvalue
    
    return {
        'test_stat': abs(b - c),
        'p_value': p_value,
        'method': 'exact_binomial',
        'b': b,
        'c': c,
        'total_discordant': total_discordant
    }


def mcnemar_asymptotic(concordance_table):
    """
    McNemar's asymptotic test (chi-square approximation).
    Valid for larger discordant counts.
    """
    b = concordance_table['only_u3']
    c = concordance_table['only_u5']
    total_discordant = b + c
    
    if total_discordant == 0:
        return {
            'test_stat': 0.0,
            'p_value': 1.0,
            'method': 'asymptotic_chi2',
            'note': 'No discordant pairs'
        }
    
    chi2_stat = (b - c) ** 2 / total_discordant
    p_value = 1 - chi2.cdf(chi2_stat, 1)
    
    return {
        'test_stat': chi2_stat,
        'p_value': p_value,
        'method': 'asymptotic_chi2',
        'b': b,
        'c': c,
        'total_discordant': total_discordant
    }


def benjamini_hochberg_correction(pvals):
    """
    Apply Benjamini-Hochberg FDR correction to p-values.
    
    Parameters:
    -----------
    pvals : array-like
        List of p-values
    
    Returns:
    --------
    array : Corrected p-values in original order
    """
    n = len(pvals)
    pvals = np.array(pvals, dtype=float)
    
    # Sort p-values and get indices
    sorted_idx = np.argsort(pvals)
    sorted_pvals = pvals[sorted_idx]
    
    # Calculate adjusted p-values
    adjusted = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if i == n - 1:
            adjusted[i] = sorted_pvals[i]
        else:
            adjusted[i] = min(adjusted[i + 1], sorted_pvals[i] * n / (i + 1))
    
    # Restore original order
    result = np.zeros(n)
    result[sorted_idx] = adjusted
    
    return result


def load_and_filter_ucmr5():
    """Load UCMR5 data, filter out lithium compounds."""
    print("Loading UCMR5 data...")
    df = pd.read_csv(UCMR5_FILE, sep='\t', encoding='latin-1')
    
    # Basic filtering - remove lithium entries
    lithium_keywords = ['lithium', 'Li']
    mask = ~df['Contaminant'].str.contains('|'.join(lithium_keywords), 
                                            case=False, na=False)
    df = df[mask].copy()
    
    print(f"  Loaded {len(df)} records")
    print(f"  Unique PWSIDs: {df['PWSID'].nunique()}")
    print(f"  Unique contaminants: {df['Contaminant'].nunique()}")
    
    return df


def load_and_filter_ucmr3():
    """Load UCMR3 data."""
    print("Loading UCMR3 data...")
    df = pd.read_csv(UCMR3_FILE, sep='\t', encoding='latin-1')
    
    print(f"  Loaded {len(df)} records")
    print(f"  Unique PWSIDs: {df['PWSID'].nunique()}")
    print(f"  Unique contaminants: {df['Contaminant'].nunique()}")
    
    return df


def find_matched_systems(ucmr3_df, ucmr5_df):
    """Find PWSIDs present in both datasets."""
    ucmr3_systems = set(ucmr3_df['PWSID'].unique())
    ucmr5_systems = set(ucmr5_df['PWSID'].unique())
    matched = ucmr3_systems.intersection(ucmr5_systems)
    
    print(f"\nMatched water systems: {len(matched)}")
    print(f"  UCMR3 unique systems: {len(ucmr3_systems)}")
    print(f"  UCMR5 unique systems: {len(ucmr5_systems)}")
    print(f"  Overlap (matched): {len(matched)}")
    
    return matched


def parse_result_value(row):
    """
    Parse analytical result value from UCMR data.
    Handle detection signs: < (below), = (detected), > (above)
    """
    sign = row.get('AnalyticalResultsSign', '<')
    value = row.get('AnalyticalResultValue', np.nan)
    mrl = row.get('MRL', np.nan)
    
    # Try to convert value to float
    try:
        if pd.isna(value) or value == '' or value == '<':
            return np.nan, False  # (value, is_detected)
        value_float = float(value)
        if sign == '=':
            return value_float, True
        elif sign == '<':
            return np.nan, False
        elif sign == '>':
            return value_float, True
        else:
            return np.nan, False
    except (ValueError, TypeError):
        return np.nan, False


def calculate_detection_rates(ucmr3_df, ucmr5_df, matched_systems):
    """
    Calculate detection rates for each shared compound.
    For UCMR5, apply UCMR3 MRLs for fair comparison.
    """
    results = []
    
    print("\n" + "="*80)
    print("DETECTION RATE ANALYSIS")
    print("="*80)
    
    # Filter to matched systems only
    ucmr3_matched = ucmr3_df[ucmr3_df['PWSID'].isin(matched_systems)].copy()
    ucmr5_matched = ucmr5_df[ucmr5_df['PWSID'].isin(matched_systems)].copy()
    
    for compound in SHARED_COMPOUNDS:
        print(f"\n{compound}:")
        
        # UCMR3 data
        ucmr3_comp = ucmr3_matched[
            (ucmr3_matched['Contaminant'] == compound)
        ].copy()
        
        if len(ucmr3_comp) == 0:
            print(f"  UCMR3: No data found")
            continue
        
        ucmr3_comp['value'], ucmr3_comp['detected'] = zip(
            *ucmr3_comp.apply(parse_result_value, axis=1)
        )
        
        ucmr3_n = ucmr3_comp['PWSID'].nunique()
        ucmr3_detected = ucmr3_comp[ucmr3_comp['detected']]['PWSID'].nunique()
        ucmr3_rate = ucmr3_detected / ucmr3_n if ucmr3_n > 0 else 0
        ucmr3_rate_pct = ucmr3_rate * 100
        
        print(f"  UCMR3: {ucmr3_detected}/{ucmr3_n} detected ({ucmr3_rate_pct:.2f}%)")
        
        # Compare to manuscript
        manuscript_rate = MANUSCRIPT_UCMR3_RATES.get(compound, np.nan)
        if not np.isnan(manuscript_rate):
            diff = abs(ucmr3_rate - manuscript_rate) * 100
            print(f"    Manuscript: {manuscript_rate*100:.2f}%, Diff: {diff:.2f}pp")
        
        # UCMR5 data - apply UCMR3 MRL
        ucmr3_mrl = UCMR3_MRLS[compound]
        ucmr5_comp = ucmr5_matched[
            (ucmr5_matched['Contaminant'] == compound)
        ].copy()
        
        if len(ucmr5_comp) == 0:
            print(f"  UCMR5: No data found")
            continue
        
        ucmr5_comp['value'], ucmr5_comp['detected'] = zip(
            *ucmr5_comp.apply(parse_result_value, axis=1)
        )
        
        # Apply UCMR3 MRL threshold to UCMR5
        ucmr5_comp['detected_mrl_adj'] = (
            (ucmr5_comp['detected']) & 
            (ucmr5_comp['value'] >= ucmr3_mrl)
        )
        
        ucmr5_n = ucmr5_comp['PWSID'].nunique()
        ucmr5_detected = ucmr5_comp[ucmr5_comp['detected_mrl_adj']]['PWSID'].nunique()
        ucmr5_rate = ucmr5_detected / ucmr5_n if ucmr5_n > 0 else 0
        ucmr5_rate_pct = ucmr5_rate * 100
        
        print(f"  UCMR5 (MRL-adjusted): {ucmr5_detected}/{ucmr5_n} detected ({ucmr5_rate_pct:.2f}%)")
        print(f"    UCMR3 MRL applied: {ucmr3_mrl} µg/L")
        
        # Calculate Wilson CIs
        ucmr3_p, ucmr3_ci_lo, ucmr3_ci_hi = wilson_ci(ucmr3_detected, ucmr3_n)
        ucmr5_p, ucmr5_ci_lo, ucmr5_ci_hi = wilson_ci(ucmr5_detected, ucmr5_n)
        
        print(f"  UCMR3 95% CI: [{ucmr3_ci_lo*100:.2f}%, {ucmr3_ci_hi*100:.2f}%]")
        print(f"  UCMR5 95% CI: [{ucmr5_ci_lo*100:.2f}%, {ucmr5_ci_hi*100:.2f}%]")
        
        # Rate difference
        rate_diff_pct = (ucmr5_rate - ucmr3_rate) * 100
        print(f"  Difference: {rate_diff_pct:+.2f}pp")
        
        # Build concordance table for McNemar's test
        # Need to map systems to detection status in each round
        ucmr3_detected_systems = set(
            ucmr3_comp[ucmr3_comp['detected']]['PWSID'].unique()
        )
        ucmr5_detected_systems = set(
            ucmr5_comp[ucmr5_comp['detected_mrl_adj']]['PWSID'].unique()
        )
        
        # Get all systems tested in both rounds for this compound
        all_systems_both = set(ucmr3_comp['PWSID'].unique()).intersection(
            set(ucmr5_comp['PWSID'].unique())
        )
        
        a = len(ucmr3_detected_systems.intersection(ucmr5_detected_systems))
        b = len(ucmr3_detected_systems - ucmr5_detected_systems)
        c = len(ucmr5_detected_systems - ucmr3_detected_systems)
        d = len(all_systems_both) - a - b - c
        
        print(f"\n  Concordance table (matched systems with both rounds tested):")
        print(f"    Both detected: {a}")
        print(f"    Only UCMR3: {b}")
        print(f"    Only UCMR5: {c}")
        print(f"    Neither detected: {d}")
        print(f"    Total: {a+b+c+d} (of {len(all_systems_both)} systems)")
        
        # Perform McNemar's test
        concordance_table = {
            'both_det': a,
            'only_u3': b,
            'only_u5': c,
            'neither': d
        }
        
        if (b + c) < 25:
            mcnemar_result = mcnemar_exact(concordance_table)
            print(f"  McNemar's test (exact binomial):")
        else:
            mcnemar_result = mcnemar_asymptotic(concordance_table)
            print(f"  McNemar's test (chi-square):")
        
        print(f"    Test statistic: {mcnemar_result.get('test_stat', np.nan):.4f}")
        print(f"    p-value: {mcnemar_result.get('p_value', np.nan):.6f}")
        print(f"    Method: {mcnemar_result.get('method', 'unknown')}")
        
        # Determine direction of change
        if b > c:
            direction = "UCMR3 > UCMR5"
        elif c > b:
            direction = "UCMR5 > UCMR3"
        else:
            direction = "No change"
        print(f"    Direction: {direction}")
        
        # Store results
        results.append({
            'Compound': compound,
            'UCMR3_N': ucmr3_n,
            'UCMR3_Detected': ucmr3_detected,
            'UCMR3_Rate': ucmr3_rate,
            'UCMR3_Rate_Pct': ucmr3_rate_pct,
            'UCMR3_CI_Lo': ucmr3_ci_lo,
            'UCMR3_CI_Hi': ucmr3_ci_hi,
            'UCMR5_N': ucmr5_n,
            'UCMR5_Detected_MRL_Adj': ucmr5_detected,
            'UCMR5_Rate_MRL_Adj': ucmr5_rate,
            'UCMR5_Rate_Pct_MRL_Adj': ucmr5_rate_pct,
            'UCMR5_CI_Lo': ucmr5_ci_lo,
            'UCMR5_CI_Hi': ucmr5_ci_hi,
            'Rate_Diff_Pp': rate_diff_pct,
            'UCMR3_MRL_Applied': ucmr3_mrl,
            'Concordance_Both_Det': a,
            'Concordance_Only_U3': b,
            'Concordance_Only_U5': c,
            'Concordance_Neither': d,
            'McNemar_Stat': mcnemar_result.get('test_stat', np.nan),
            'McNemar_P': mcnemar_result.get('p_value', np.nan),
            'McNemar_Method': mcnemar_result.get('method', 'unknown'),
            'Direction': direction,
            'Manuscript_UCMR3_Rate': MANUSCRIPT_UCMR3_RATES.get(compound, np.nan),
            'UCMR3_vs_Manuscript_Diff_Pp': (
                (ucmr3_rate - MANUSCRIPT_UCMR3_RATES.get(compound, ucmr3_rate)) * 100
            ) if compound in MANUSCRIPT_UCMR3_RATES else np.nan,
            'Systems_Both_Rounds': len(all_systems_both),
            'UCMR5_Concentration_Data': ucmr5_comp[['PWSID', 'value', 'detected']].copy(),
            'UCMR3_Concentration_Data': ucmr3_comp[['PWSID', 'value', 'detected']].copy()
        })
    
    return results


def apply_fdr_correction(results_df):
    """Apply Benjamini-Hochberg FDR correction to McNemar p-values."""
    print("\n" + "="*80)
    print("BENJAMINI-HOCHBERG FDR CORRECTION")
    print("="*80)
    
    pvals = results_df['McNemar_P'].values
    print(f"Input p-values: {pvals}")
    
    adjusted_pvals = benjamini_hochberg_correction(pvals)
    print(f"Adjusted p-values: {adjusted_pvals}")
    
    results_df['McNemar_P_FDR'] = adjusted_pvals
    
    return results_df


def mann_whitney_analysis(results_dict):
    """
    Perform Mann-Whitney U test on concentration distributions.
    Only for systems where compound detected in BOTH rounds.
    """
    print("\n" + "="*80)
    print("MANN-WHITNEY U TEST (Concentration distributions)")
    print("="*80)
    
    mw_results = []
    
    for result in results_dict:
        compound = result['Compound']
        print(f"\n{compound}:")
        
        # Get concentration data
        ucmr5_conc_df = result['UCMR5_Concentration_Data']
        ucmr3_conc_df = result['UCMR3_Concentration_Data']
        
        # Get systems where both detected
        ucmr5_detected = set(ucmr5_conc_df[ucmr5_conc_df['detected']]['PWSID'])
        ucmr3_detected = set(ucmr3_conc_df[ucmr3_conc_df['detected']]['PWSID'])
        both_detected = ucmr5_detected.intersection(ucmr3_detected)
        
        print(f"  Systems with detection in both rounds: {len(both_detected)}")
        
        if len(both_detected) < 3:
            print(f"  Insufficient paired data (n={len(both_detected)})")
            mw_results.append({
                'Compound': compound,
                'N_Paired': len(both_detected),
                'U_Stat': np.nan,
                'P_Value': np.nan,
                'Effect_Size_Rank_Biserial': np.nan,
                'Note': 'Insufficient data'
            })
            continue
        
        # Get values for paired systems
        ucmr5_values = []
        ucmr3_values = []
        for system in both_detected:
            u5_val = ucmr5_conc_df[ucmr5_conc_df['PWSID'] == system]['value'].dropna()
            u3_val = ucmr3_conc_df[ucmr3_conc_df['PWSID'] == system]['value'].dropna()
            
            if len(u5_val) > 0 and len(u3_val) > 0:
                # Use first detection if multiple samples
                ucmr5_values.append(u5_val.iloc[0])
                ucmr3_values.append(u3_val.iloc[0])
        
        if len(ucmr5_values) < 3:
            print(f"  Insufficient numeric values (n={len(ucmr5_values)})")
            mw_results.append({
                'Compound': compound,
                'N_Paired': len(ucmr5_values),
                'U_Stat': np.nan,
                'P_Value': np.nan,
                'Effect_Size_Rank_Biserial': np.nan,
                'Note': 'Insufficient numeric values'
            })
            continue
        
        ucmr5_values = np.array(ucmr5_values)
        ucmr3_values = np.array(ucmr3_values)
        
        # Perform Mann-Whitney U test
        u_stat, p_value = stats.mannwhitneyu(ucmr5_values, ucmr3_values, 
                                              alternative='two-sided')
        
        # Calculate rank-biserial correlation as effect size
        n = len(ucmr5_values) + len(ucmr3_values)
        r = 1 - (2 * u_stat) / (n * (n - 1))
        
        print(f"  UCMR5 (n={len(ucmr5_values)}): median={np.median(ucmr5_values):.4f}, "
              f"mean={np.mean(ucmr5_values):.4f}")
        print(f"  UCMR3 (n={len(ucmr3_values)}): median={np.median(ucmr3_values):.4f}, "
              f"mean={np.mean(ucmr3_values):.4f}")
        print(f"  U statistic: {u_stat:.1f}")
        print(f"  p-value: {p_value:.6f}")
        print(f"  Effect size (rank-biserial r): {r:.4f}")
        
        mw_results.append({
            'Compound': compound,
            'N_Paired': len(ucmr5_values),
            'U_Stat': u_stat,
            'P_Value': p_value,
            'Effect_Size_Rank_Biserial': r,
            'UCMR5_Median': np.median(ucmr5_values),
            'UCMR3_Median': np.median(ucmr3_values),
            'Note': 'OK'
        })
    
    return mw_results


def save_results(results_dict, mw_results, results_df_fdr):
    """Save all results to CSV files."""
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # McNemar + Wilson results
    output_file_mcnemar = os.path.join(
        OUTPUT_DIR, 
        'ucmr3_comparison_mcnemar_enhanced.csv'
    )
    
    df_output = pd.DataFrame([{
        'Compound': r['Compound'],
        'UCMR3_N': r['UCMR3_N'],
        'UCMR3_Detected': r['UCMR3_Detected'],
        'UCMR3_Rate_Pct': r['UCMR3_Rate_Pct'],
        'UCMR3_CI_Lo_Pct': r['UCMR3_CI_Lo'] * 100,
        'UCMR3_CI_Hi_Pct': r['UCMR3_CI_Hi'] * 100,
        'UCMR5_N': r['UCMR5_N'],
        'UCMR5_Detected_MRL_Adj': r['UCMR5_Detected_MRL_Adj'],
        'UCMR5_Rate_Pct_MRL_Adj': r['UCMR5_Rate_Pct_MRL_Adj'],
        'UCMR5_CI_Lo_Pct': r['UCMR5_CI_Lo'] * 100,
        'UCMR5_CI_Hi_Pct': r['UCMR5_CI_Hi'] * 100,
        'Rate_Diff_Pp': r['Rate_Diff_Pp'],
        'UCMR3_MRL_Applied_ugL': r['UCMR3_MRL_Applied'],
        'Concordance_Both_Det': r['Concordance_Both_Det'],
        'Concordance_Only_UCMR3': r['Concordance_Only_U3'],
        'Concordance_Only_UCMR5': r['Concordance_Only_U5'],
        'Concordance_Neither': r['Concordance_Neither'],
        'Systems_Tested_Both_Rounds': r['Systems_Both_Rounds'],
        'McNemar_Stat': r['McNemar_Stat'],
        'McNemar_P_Value': r['McNemar_P'],
        'McNemar_P_FDR_Adjusted': results_df_fdr.loc[
            results_df_fdr['Compound'] == r['Compound'], 'McNemar_P_FDR'
        ].values[0],
        'McNemar_Method': r['McNemar_Method'],
        'Direction': r['Direction'],
        'Manuscript_UCMR3_Rate_Pct': r['Manuscript_UCMR3_Rate'] * 100 
            if not np.isnan(r['Manuscript_UCMR3_Rate']) else np.nan,
        'Calculated_vs_Manuscript_Diff_Pp': r['UCMR3_vs_Manuscript_Diff_Pp']
    } for r in results_dict])
    
    df_output.to_csv(output_file_mcnemar, index=False)
    print(f"Saved: {output_file_mcnemar}")
    
    # Mann-Whitney results
    output_file_mw = os.path.join(
        OUTPUT_DIR,
        'ucmr3_comparison_mannwhitney.csv'
    )
    
    df_mw = pd.DataFrame(mw_results)
    df_mw.to_csv(output_file_mw, index=False)
    print(f"Saved: {output_file_mw}")
    
    # Concordance summary
    output_file_concordance = os.path.join(
        OUTPUT_DIR,
        'ucmr3_matched_concordance.csv'
    )
    
    df_concordance = pd.DataFrame([{
        'Compound': r['Compound'],
        'Both_Detected': r['Concordance_Both_Det'],
        'Only_UCMR3': r['Concordance_Only_U3'],
        'Only_UCMR5': r['Concordance_Only_U5'],
        'Neither_Detected': r['Concordance_Neither'],
        'Total_Systems': r['Systems_Both_Rounds']
    } for r in results_dict])
    
    df_concordance.to_csv(output_file_concordance, index=False)
    print(f"Saved: {output_file_concordance}")
    
    print("\nAll results saved successfully!")


def print_summary(results_dict, mw_results, results_df_fdr):
    """Print comprehensive summary."""
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    
    print("\nMcNemar's Test Results:")
    print("-" * 120)
    print(f"{'Compound':<10} {'UCMR3':<12} {'UCMR5 MRL-Adj':<15} {'Change':<10} "
          f"{'McNemar p':<12} {'FDR p':<12} {'Direction':<20}")
    print("-" * 120)
    
    for r in results_dict:
        compound = r['Compound']
        ucmr3_rate = r['UCMR3_Rate_Pct']
        ucmr5_rate = r['UCMR5_Rate_Pct_MRL_Adj']
        change = r['Rate_Diff_Pp']
        mcnemar_p = r['McNemar_P']
        fdr_p = results_df_fdr.loc[
            results_df_fdr['Compound'] == compound, 'McNemar_P_FDR'
        ].values[0]
        direction = r['Direction']
        
        print(f"{compound:<10} {ucmr3_rate:>6.2f}%{'':<4} {ucmr5_rate:>6.2f}%{'':<6} "
              f"{change:>+6.2f}pp{'':<1} {mcnemar_p:>10.4f}  {fdr_p:>10.4f}  {direction:<20}")
    
    print("\nManuscript Validation (UCMR3 rates):")
    print("-" * 80)
    print(f"{'Compound':<10} {'Calculated':<15} {'Manuscript':<15} {'Diff':<10}")
    print("-" * 80)
    
    for r in results_dict:
        compound = r['Compound']
        calc = r['UCMR3_Rate_Pct']
        manu = r['Manuscript_UCMR3_Rate'] * 100 if not np.isnan(r['Manuscript_UCMR3_Rate']) else np.nan
        diff = r['UCMR3_vs_Manuscript_Diff_Pp']
        
        manu_str = f"{manu:.2f}%" if not np.isnan(manu) else "N/A"
        diff_str = f"{diff:+.2f}pp" if not np.isnan(diff) else "N/A"
        
        print(f"{compound:<10} {calc:>6.2f}%{'':<6} {manu_str:>14} {diff_str:>9}")


def main():
    """Main execution."""
    print("\n" + "="*80)
    print("UCMR 3 vs 5 COMPARISON: ENHANCED ANALYSIS")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  UCMR5 file: {UCMR5_FILE}")
    print(f"  UCMR3 file: {UCMR3_FILE}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Shared compounds: {SHARED_COMPOUNDS}")
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    ucmr5_df = load_and_filter_ucmr5()
    ucmr3_df = load_and_filter_ucmr3()
    
    # Find matched systems
    matched_systems = find_matched_systems(ucmr3_df, ucmr5_df)
    
    # Calculate detection rates and perform McNemar's test
    results_dict = calculate_detection_rates(ucmr3_df, ucmr5_df, matched_systems)
    
    # Convert to DataFrame for FDR correction
    results_df = pd.DataFrame([{
        'Compound': r['Compound'],
        'McNemar_P': r['McNemar_P']
    } for r in results_dict])
    
    # Apply FDR correction
    results_df = apply_fdr_correction(results_df)
    
    # Merge FDR results back
    for r in results_dict:
        r['McNemar_P_FDR'] = results_df.loc[
            results_df['Compound'] == r['Compound'], 'McNemar_P_FDR'
        ].values[0]
    
    # Perform Mann-Whitney U test
    mw_results = mann_whitney_analysis(results_dict)
    
    # Save results
    save_results(results_dict, mw_results, results_df)
    
    # Print summary
    print_summary(results_dict, mw_results, results_df)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
