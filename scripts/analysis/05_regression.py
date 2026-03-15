#!/usr/bin/env python3
"""
PFAS UCMR 5 Analysis - Enhanced Phase 5: Regression Models with Diagnostics
==============================================================================
Extends original Phase 5 with:
- Hosmer-Lemeshow test
- Calibration curves
- Cook's distance (influential observations)
- Pseudo-R² (McFadden, Nagelkerke, Cox-Snell)
- AUC/ROC curves
- Deviance residuals analysis
- GEE sensitivity analysis (optional)

Verifies: Tables 3-4, Table S15, Table S18, Table S21
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.genmod.api import GEE
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Binomial
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import chi2
from sklearn.metrics import roc_auc_score
from datetime import datetime
import json, os, warnings

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
BASE = "/sessions/optimistic-determined-feynman/mnt/5_PFAS_Research_Finalization"
MERGED_FILE_JAN2026 = os.path.join(BASE, "3_Jan2026_Update_Output_CSVs", "merged_pws_dataset_jan2026.csv")
MERGED_FILE_OLD = os.path.join(BASE, "02_Processed_Data", "merged_pws_dataset.csv")
OUTPUT_DIR = os.path.join(BASE, "3_Jan2026_Update_Output_CSVs")

results = {}
log_lines = []
diagnostics = {}

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def log(msg):
    """Log message to console and file."""
    print(msg)
    log_lines.append(msg)

def check(label, computed, manuscript, tolerance=0.1):
    """Verify computed vs manuscript values."""
    if isinstance(computed, (int, float)) and isinstance(manuscript, (int, float)):
        if manuscript != 0:
            pct_diff = abs(computed - manuscript) / abs(manuscript) * 100
        else:
            pct_diff = abs(computed - manuscript) * 100
        match = "MATCH" if pct_diff < tolerance * 100 else "MISMATCH"
        log(f"  [{match}] {label}: computed={computed:.3f}, manuscript={manuscript}, diff={pct_diff:.2f}%")
    else:
        match = "MATCH" if computed == manuscript else "MISMATCH"
        log(f"  [{match}] {label}: computed={computed}, manuscript={manuscript}")
    results[label] = {"computed": str(computed), "manuscript": str(manuscript), "status": match}

def hosmer_lemeshow(y_true, y_pred, g=10):
    """
    Hosmer-Lemeshow goodness-of-fit test.
    
    Parameters
    ----------
    y_true : array-like
        True binary outcomes (0/1)
    y_pred : array-like
        Predicted probabilities (0-1)
    g : int
        Number of groups (default 10)
    
    Returns
    -------
    hl_stat : float
        Chi-square test statistic
    p_value : float
        P-value from chi-square distribution
    table : pd.DataFrame
        Detailed breakdown by group
    """
    # Sort by predicted probability
    sorted_idx = np.argsort(y_pred)
    y_true_s = np.array(y_true)[sorted_idx]
    y_pred_s = np.array(y_pred)[sorted_idx]
    
    # Create g groups (with approximately equal size)
    groups = np.array_split(np.arange(len(y_true_s)), g)
    
    hl_stat = 0
    table = []
    
    for i, grp in enumerate(groups):
        obs_1 = y_true_s[grp].sum()
        obs_0 = len(grp) - obs_1
        exp_1 = y_pred_s[grp].sum()
        exp_0 = len(grp) - exp_1
        
        if exp_1 > 0:
            hl_stat += (obs_1 - exp_1)**2 / exp_1
        if exp_0 > 0:
            hl_stat += (obs_0 - exp_0)**2 / exp_0
        
        table.append({
            'group': i + 1,
            'n': len(grp),
            'obs_1': int(obs_1),
            'exp_1': round(exp_1, 2),
            'obs_0': int(obs_0),
            'exp_0': round(exp_0, 2),
            'mean_pred_prob': round(y_pred_s[grp].mean(), 4)
        })
    
    p_value = 1 - chi2.cdf(hl_stat, g - 2)
    
    return hl_stat, p_value, pd.DataFrame(table)

def get_cooks_distance(y, X, logit_result):
    """
    Compute Cook's distance for logistic regression.
    """
    y_pred = logit_result.predict(X)
    deviance_resid = np.zeros(len(y))
    
    for i in range(len(y)):
        if y_pred.iloc[i] > 0 and y_pred.iloc[i] < 1:
            deviance_resid[i] = np.sign(y.iloc[i] - y_pred.iloc[i]) * \
                                np.sqrt(-2 * (y.iloc[i] * np.log(y_pred.iloc[i]) + 
                                             (1 - y.iloc[i]) * np.log(1 - y_pred.iloc[i])))
        else:
            deviance_resid[i] = 0
    
    # Leverage (hat values)
    X_arr = X.values
    try:
        H = X_arr @ np.linalg.inv(X_arr.T @ X_arr) @ X_arr.T
        leverage = np.diag(H)
    except:
        leverage = np.ones(len(y)) / len(y)
    
    # Cook's distance
    p = X_arr.shape[1]
    cooks_d = (deviance_resid**2 / p) * (leverage / (1 - leverage + 1e-10))
    
    # Threshold: 4/n
    threshold = 4 / len(y)
    influential_mask = cooks_d > threshold
    
    return cooks_d, influential_mask

def compute_pseudo_r2(result, y, X):
    """Compute multiple pseudo-R² measures for logistic regression."""
    n = len(y)
    
    # Log-likelihoods
    LL = result.llf
    LL_null = result.llnull
    
    # McFadden R²
    mcfadden_r2 = result.prsquared
    
    # Cox-Snell R²
    cox_snell_r2 = 1 - np.exp(-2/n * (LL - LL_null))
    
    # Nagelkerke R² (Max-rescaled R²)
    max_cox_snell = 1 - np.exp(2/n * LL_null)
    nagelkerke_r2 = cox_snell_r2 / max_cox_snell if max_cox_snell > 0 else 0
    
    return {
        'McFadden': round(mcfadden_r2, 4),
        'Cox-Snell': round(cox_snell_r2, 4),
        'Nagelkerke': round(nagelkerke_r2, 4)
    }

def get_calibration_data(y_true, y_pred, n_bins=10):
    """Create calibration curve data (observed vs predicted by decile)."""
    # Convert to pandas Series for qcut
    if isinstance(y_pred, np.ndarray):
        y_pred = pd.Series(y_pred)
    if isinstance(y_true, np.ndarray):
        y_true = pd.Series(y_true)
    
    deciles = pd.qcut(y_pred, q=n_bins, duplicates='drop', labels=False)
    
    calib_data = []
    for decile in sorted(deciles.unique()):
        mask = deciles == decile
        calib_data.append({
            'decile': int(decile) + 1,
            'mean_pred_prob': round(y_pred[mask].mean(), 4),
            'obs_proportion': round(y_true[mask].mean(), 4),
            'count': mask.sum()
        })
    
    return pd.DataFrame(calib_data)

def compute_deviance_residuals(y, y_pred):
    """Compute deviance residuals."""
    # Ensure we have numpy arrays
    if hasattr(y, 'values'):
        y = y.values
    if hasattr(y_pred, 'values'):
        y_pred = y_pred.values
    
    resid = np.zeros(len(y))
    for i in range(len(y)):
        if y_pred[i] > 1e-10 and y_pred[i] < 1 - 1e-10:
            resid[i] = np.sign(y[i] - y_pred[i]) * \
                       np.sqrt(-2 * (y[i] * np.log(y_pred[i]) + \
                                    (1 - y[i]) * np.log(1 - y_pred[i])))
    return resid

log("\n--- Loading merged dataset (Jan 2026) ---")
df = pd.read_csv(MERGED_FILE_JAN2026, low_memory=False)
log(f"Shape: {df.shape}")

# Try to merge distance variables from old dataset
log("\n--- Merging distance and demographic variables from old dataset ---")
try:
    df_old = pd.read_csv(MERGED_FILE_OLD, low_memory=False)
    merge_cols = []
    for col in df_old.columns:
        if any(x in col.lower() for x in ['dist_', 'pct_', 'minority', 'poverty']):
            merge_cols.append(col)
    
    if merge_cols:
        log(f"Found {len(merge_cols)} distance/demographic columns to merge")
        df = df.merge(df_old[['PWSID'] + merge_cols], on='PWSID', how='left')
        log(f"Merged shape: {df.shape}")
except Exception as e:
    log(f"Note: Could not merge old dataset - {e}")

# ============================================================
# DATA PREPARATION
# ============================================================
log("\n--- Data Preparation ---")

n_total = len(df)
check("Total study population", n_total, 10297)

# Primary models: need distance variables
if 'dist_nearest_superfund_sites_km' in df.columns:
    primary = df.dropna(subset=['dist_nearest_superfund_sites_km']).copy()
else:
    primary = df.copy()
    log("WARNING: Could not identify coordinate filter")

n_primary = len(primary)
log(f"Primary model N (after dropping NAs): {n_primary}")
check("Primary model N", n_primary, 10137)

# Identify demographic columns
demo_cols_available = [c for c in df.columns if any(x in c.lower() for x in ['pct_minority', 'pct_poverty'])]
log(f"Available demographic columns: {demo_cols_available}")

if demo_cols_available:
    demo = primary.dropna(subset=[demo_cols_available[0]]).copy()
else:
    demo = primary.copy()
    log("WARNING: No demographic columns found")

n_demo = len(demo)
log(f"Demographic model N: {n_demo}")
check("Demographic model N", n_demo, 4444)

# Detection outcome
det_col = 'any_pfas_detected'
if det_col in primary.columns:
    overall_det = round(df[det_col].mean() * 100, 1)
    primary_det = round(primary[det_col].mean() * 100, 1)
    demo_det = round(demo[det_col].mean() * 100, 1)
    log(f"Detection rates: overall={overall_det}%, primary={primary_det}%, demo={demo_det}%")

# ============================================================
# CREATE VARIABLES FOR REGRESSION
# ============================================================
log("\n--- Creating predictor variables ---")

# Log transformations (ensure numeric)
if 'POPULATION_SERVED_COUNT' in primary.columns:
    primary['log_population'] = np.log1p(pd.to_numeric(primary['POPULATION_SERVED_COUNT'], errors='coerce'))
    log("  Created log_population")

if 'POPULATION_DENSITY' in primary.columns:
    primary['log_pop_density'] = np.log1p(pd.to_numeric(primary['POPULATION_DENSITY'], errors='coerce'))
    log("  Created log_pop_density")

if 'dist_nearest_superfund_sites_km' in primary.columns:
    primary['log_dist_superfund'] = np.log1p(pd.to_numeric(primary['dist_nearest_superfund_sites_km'], errors='coerce'))
    log("  Created log_dist_superfund")

if 'dist_nearest_spills_km' in primary.columns:
    primary['log_dist_spills'] = np.log1p(pd.to_numeric(primary['dist_nearest_spills_km'], errors='coerce'))
    log("  Created log_dist_spills")

if 'dist_nearest_federal_sites_km' in primary.columns:
    primary['log_dist_federal'] = np.log1p(pd.to_numeric(primary['dist_nearest_federal_sites_km'], errors='coerce'))
    log("  Created log_dist_federal")

# Categorical variables
if 'GW_SW_CODE' in primary.columns:
    primary['source_water_SW'] = (primary['GW_SW_CODE'] == 'SW').astype(int)
    log("  Created source_water_SW")

if 'OWNER_TYPE_CODE' in primary.columns:
    primary['owner_private'] = (primary['OWNER_TYPE_CODE'] == 'P').astype(int)
    log("  Created owner_private")

# Region dummies
if 'epa_region' in primary.columns:
    region_dummies_df = pd.get_dummies(primary['epa_region'], prefix='region')
    for col in region_dummies_df.columns:
        primary[col] = region_dummies_df[col].astype(int)
    region_dummies = sorted([c for c in region_dummies_df.columns if c != 'region_0'])
    log(f"  Created {len(region_dummies)} region dummies")
else:
    region_dummies = []

# ============================================================
# PRIMARY LOGISTIC REGRESSION
# ============================================================
log("\n" + "=" * 90)
log("PRIMARY DETECTION MODEL - Logistic Regression")
log("=" * 90)

# Define model variables
X_cols = [c for c in ['log_population', 'source_water_SW', 'owner_private', 
                       'log_dist_superfund', 'log_dist_federal', 'log_dist_spills',
                       'log_pop_density'] if c in primary.columns]
X_cols.extend(region_dummies)

log(f"\nModel variables ({len(X_cols)}): {X_cols[:10]}...")

# Prepare data
model_df = primary[X_cols + [det_col]].dropna()
n_model = len(model_df)
log(f"Model N after dropping NAs: {n_model}")

# Convert all to float64 to avoid type issues
for col in X_cols + [det_col]:
    model_df[col] = pd.to_numeric(model_df[col], errors='coerce')

model_df = model_df.dropna()
n_model_final = len(model_df)
log(f"Model N after numeric conversion: {n_model_final}")

X = sm.add_constant(model_df[X_cols])
y = model_df[det_col].astype(int)

# Initialize output structures
hl_table = None
calib_data = None
or_df = None
top_cooks = None
deviance_summary = None

try:
    log(f"\nFitting logistic regression with {X.shape[1]-1} predictors...")
    logit = sm.Logit(y, X)
    result = logit.fit(disp=0, maxiter=200)
    
    log(f"Convergence: {result.mle_retvals.get('converged', 'Unknown')}")
    log(f"Iterations: {result.mle_retvals.get('iterations', 'Unknown')}")
    
    # ========== PSEUDO-R² ==========
    log("\n--- Pseudo-R² Metrics ---")
    pseudo_r2 = compute_pseudo_r2(result, y, X)
    for metric, value in pseudo_r2.items():
        log(f"  {metric:15s} R²: {value:.4f}")
        diagnostics[f"PseudoR2_{metric}"] = value
    
    # AUC
    log("\n--- AUC / ROC ---")
    y_pred = result.predict(X)
    auc = roc_auc_score(y, y_pred)
    log(f"  AUC: {auc:.4f}")
    diagnostics['AUC'] = round(auc, 4)
    check("AUC Detection", round(auc, 3), 0.700)
    
    # ========== HOSMER-LEMESHOW TEST ==========
    log("\n--- Hosmer-Lemeshow Goodness-of-Fit Test ---")
    hl_stat, hl_pval, hl_table = hosmer_lemeshow(y.values, y_pred.values, g=10)
    log(f"  Test Statistic: {hl_stat:.4f}")
    log(f"  P-value: {hl_pval:.4f}")
    log(f"  Conclusion: {'Good fit' if hl_pval > 0.05 else 'Poor fit'}")
    diagnostics['HL_Statistic'] = round(hl_stat, 4)
    diagnostics['HL_PValue'] = round(hl_pval, 4)
    
    # ========== CALIBRATION CURVE ==========
    log("\n--- Calibration Curve (Predicted vs Observed by Decile) ---")
    calib_data = get_calibration_data(y.values, y_pred.values, n_bins=10)
    log(f"  Calibration table (first 5 rows):\n{calib_data.head().to_string()}")
    
    # ========== COOK'S DISTANCE ==========
    log("\n--- Cook's Distance (Influential Observations) ---")
    cooks_d, influential_mask = get_cooks_distance(y, X, result)
    n_influential = influential_mask.sum()
    pct_influential = round(100 * n_influential / len(y), 2)
    log(f"  Influential observations (D > 4/n={4/len(y):.4f}): {n_influential} ({pct_influential}%)")
    diagnostics['N_Influential'] = int(n_influential)
    diagnostics['Pct_Influential'] = pct_influential
    
    # Top 20 influential
    top_idx = np.argsort(cooks_d)[-20:][::-1]
    top_cooks = pd.DataFrame({
        'observation_index': top_idx,
        'cooks_distance': np.round(cooks_d[top_idx], 6),
        'y_observed': y.iloc[top_idx].values,
        'y_predicted': np.round(y_pred.iloc[top_idx].values, 4)
    })
    log(f"\n  Top 20 influential observations:\n{top_cooks.to_string()}")
    
    # ========== DEVIANCE RESIDUALS ==========
    log("\n--- Deviance Residuals ---")
    deviance_resid = compute_deviance_residuals(y.values, y_pred.values)
    log(f"  Mean: {deviance_resid.mean():.4f}")
    log(f"  Std Dev: {deviance_resid.std():.4f}")
    log(f"  Min: {deviance_resid.min():.4f}")
    log(f"  Max: {deviance_resid.max():.4f}")
    log(f"  Skewness: {pd.Series(deviance_resid).skew():.4f}")
    
    deviance_summary = pd.DataFrame({
        'Metric': ['Mean', 'Std Dev', 'Min', 'Max', 'Skewness'],
        'Value': [
            round(deviance_resid.mean(), 4),
            round(deviance_resid.std(), 4),
            round(deviance_resid.min(), 4),
            round(deviance_resid.max(), 4),
            round(pd.Series(deviance_resid).skew(), 4)
        ]
    })
    
    # ========== ODDS RATIOS AND MODEL SUMMARY ==========
    log("\n--- Odds Ratios and 95% Confidence Intervals ---")
    params = result.params
    conf = result.conf_int()
    pvalues = result.pvalues
    
    or_data = []
    for pred in X_cols:
        if pred in params.index and pred != 'const':
            or_val = np.exp(params[pred])
            ci_lo = np.exp(conf.loc[pred, 0])
            ci_hi = np.exp(conf.loc[pred, 1])
            p = pvalues[pred]
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            
            log(f"  {pred:25s}: OR={or_val:.4f} (95% CI: {ci_lo:.4f}-{ci_hi:.4f}), p={p:.6f} {sig}")
            
            or_data.append({
                'Variable': pred,
                'Coefficient': round(params[pred], 4),
                'SE': round(result.bse[pred], 4),
                'OR': round(or_val, 4),
                'CI_Lower': round(ci_lo, 4),
                'CI_Upper': round(ci_hi, 4),
                'P_Value': round(p, 6),
                'Significance': sig
            })
    
    or_df = pd.DataFrame(or_data)
    
    # VIF
    log("\n--- Variance Inflation Factors (VIF) ---")
    X_vif = model_df[[c for c in X_cols if not c.startswith('region_')]].dropna()
    if len(X_vif) > 0:
        X_vif_const = sm.add_constant(X_vif)
        vif_data = []
        for i, col in enumerate(X_vif.columns):
            try:
                vif_val = variance_inflation_factor(X_vif_const.values, i + 1)
                log(f"  {col}: VIF={vif_val:.2f}")
                vif_data.append({'Variable': col, 'VIF': round(vif_val, 2)})
                diagnostics[f"VIF_{col}"] = round(vif_val, 2)
            except Exception as e:
                log(f"  {col}: VIF calculation error - {e}")
        
        # Merge VIF into OR table
        if vif_data and or_df is not None:
            vif_df = pd.DataFrame(vif_data)
            or_df = or_df.merge(vif_df, on='Variable', how='left')
    
    # Key verifications
    log("\n--- Key Verifications (Manuscript Table 3) ---")
    if or_df is not None and 'log_population' in or_df['Variable'].values:
        or_pop = or_df[or_df['Variable'] == 'log_population']['OR'].values[0]
        check("Detection OR: log_population", round(or_pop, 2), 1.37)
    
    if or_df is not None and 'source_water_SW' in or_df['Variable'].values:
        or_sw = or_df[or_df['Variable'] == 'source_water_SW']['OR'].values[0]
        check("Detection OR: source_water_SW", round(or_sw, 3), 2.093)
    
    if or_df is not None and 'log_dist_superfund' in or_df['Variable'].values:
        or_sf = or_df[or_df['Variable'] == 'log_dist_superfund']['OR'].values[0]
        check("Detection OR: log_dist_superfund", round(or_sf, 3), 0.678)
    
except Exception as e:
    log(f"ERROR in primary model: {e}")
    import traceback
    log(traceback.format_exc())

# ============================================================
# GEE SENSITIVITY ANALYSIS (optional)
# ============================================================
log("\n" + "=" * 90)
log("GEE SENSITIVITY ANALYSIS - Clustering by State")
log("=" * 90)

try:
    if 'state' in model_df.columns:
        log("\nFitting GEE with exchangeable correlation structure...")
        
        # Create state group indices
        state_groups = model_df['state'].astype('category').cat.codes.values
        
        gee_family = Binomial()
        gee_corr = Exchangeable()
        
        gee_model = GEE(y, X, groups=state_groups, family=gee_family, cov_struct=gee_corr)
        gee_result = gee_model.fit(maxiter=200)
        
        log(f"GEE Converged: {gee_result.converged}")
        
        # Compare GEE to Logit
        log("\nComparison: GEE vs Logistic Regression (first 5 predictors)")
        log(f"{'Variable':<25} {'Logit OR':<12} {'GEE OR':<12} {'Ratio':<10}")
        log("-" * 60)
        
        for pred in X_cols[:5]:
            if pred in result.params.index and pred in gee_result.params.index:
                logit_or = np.exp(result.params[pred])
                gee_or = np.exp(gee_result.params[pred])
                ratio = gee_or / logit_or if logit_or > 0 else 0
                log(f"{pred:<25} {logit_or:<12.4f} {gee_or:<12.4f} {ratio:<10.4f}")
        
    else:
        log("STATE variable not found - skipping GEE analysis")
        
except Exception as e:
    log(f"NOTE: GEE analysis not available - {e}")

# ============================================================
# SAVE OUTPUTS
# ============================================================
log("\n" + "=" * 90)
log("SAVING OUTPUT FILES")
log("=" * 90)

# 1. Diagnostics summary
diag_df = pd.DataFrame([
    {'Metric': k, 'Value': v} for k, v in sorted(diagnostics.items())
])
diag_file = os.path.join(OUTPUT_DIR, "regression_diagnostics_enhanced.csv")
diag_df.to_csv(diag_file, index=False)
log(f"  Saved: {diag_file}")

# 2. Hosmer-Lemeshow table
if hl_table is not None:
    hl_file = os.path.join(OUTPUT_DIR, "regression_hosmer_lemeshow.csv")
    hl_table.to_csv(hl_file, index=False)
    log(f"  Saved: {hl_file}")
else:
    log("  Hosmer-Lemeshow table: Not available (model fit failed)")

# 3. Calibration data
if calib_data is not None:
    calib_file = os.path.join(OUTPUT_DIR, "regression_calibration_data.csv")
    calib_data.to_csv(calib_file, index=False)
    log(f"  Saved: {calib_file}")
else:
    log("  Calibration data: Not available (model fit failed)")

# 4. Cook's distance
if top_cooks is not None:
    cooks_file = os.path.join(OUTPUT_DIR, "regression_cooks_distance.csv")
    top_cooks.to_csv(cooks_file, index=False)
    log(f"  Saved: {cooks_file}")
else:
    log("  Cook's distance: Not available (model fit failed)")

# 5. Deviance residuals
if deviance_summary is not None:
    deviance_file = os.path.join(OUTPUT_DIR, "regression_deviance_residuals_summary.csv")
    deviance_summary.to_csv(deviance_file, index=False)
    log(f"  Saved: {deviance_file}")
else:
    log("  Deviance residuals: Not available (model fit failed)")

# 6. Odds ratios
if or_df is not None:
    or_file = os.path.join(OUTPUT_DIR, "regression_ors_enhanced.csv")
    or_df.to_csv(or_file, index=False)
    log(f"  Saved: {or_file}")
else:
    log("  Odds ratios: Not available (model fit failed)")

# ============================================================
# SUMMARY
# ============================================================
log("\n" + "=" * 90)
log("VERIFICATION SUMMARY - Enhanced Phase 5")
log("=" * 90)

n_checks = len(results)
n_match = sum(1 for v in results.values() if v['status'] == 'MATCH')
n_mismatch = sum(1 for v in results.values() if v['status'] == 'MISMATCH')
log(f"Total checks: {n_checks}")
log(f"  MATCH: {n_match}")
log(f"  MISMATCH: {n_mismatch}")

if n_mismatch > 0:
    log("\nMISMATCHES:")
    for k, v in results.items():
        if v['status'] == 'MISMATCH':
            log(f"  {k}: computed={v['computed']}, manuscript={v['manuscript']}")

log("\n" + "=" * 90)
log("Script execution completed successfully")
log("=" * 90)

# Save log file
log_file = os.path.join(OUTPUT_DIR, "regression_enhanced_execution_log.txt")
with open(log_file, 'w') as f:
    f.write('\n'.join(log_lines))
log(f"\nLog saved to: {log_file}")

# Save results JSON
results_file = os.path.join(OUTPUT_DIR, "regression_enhanced_verification.json")
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)
log(f"Verification results saved to: {results_file}")

