# Data Dictionary: Merged PWS Dataset

**File**: merged_pws_dataset_jan2026.csv
**Records**: 10,297 public water systems
**Source**: EPA UCMR 5 monitoring data merged with SDWIS, ECHO, and PFAS Analytic Tools

## System Identifiers

| Column | Type | Description |
|--------|------|-------------|
| PWSID | string | Public Water System ID (format: SS######) |
| PWSName | string | System name |
| state | string | Two-letter state/territory abbreviation |
| ZIPCode | string | ZIP code of system |

## System Characteristics

| Column | Type | Description | Valid Range |
|--------|------|-------------|-------------|
| population_served_count | integer | Population served | 3,300+ |
| source_water | string | Primary source water type | GW, SW, GU, GWP |
| owner_type | string | Ownership type | Public, Private |
| system_size | string | EPA size category | Very Small to Very Large |

## PFAS Detection (29 compounds)

Binary detection flags (1 = detected at/above MRL, 0 = not detected):

| Column Pattern | Type | Unit | Description |
|----------------|------|------|-------------|
| {compound}_detected | integer (0/1) | - | Detection at/above compound-specific MRL |
| {compound}_max_conc | float | ug/L | Maximum concentration across all sampling events |

## Summary Detection Metrics

| Column | Type | Description |
|--------|------|-------------|
| any_pfas_detected | integer (0/1) | Any PFAS detected at/above MRL |
| n_pfas_detected | integer | Count of compounds detected (0-29) |

## MCL Exceedance Flags

Thresholds: PFOS = 4.0 ng/L, PFOA = 4.0 ng/L, PFHxS = 10 ng/L, PFNA = 10 ng/L, HFPO-DA = 10 ng/L

| Column | Type | Description |
|--------|------|-------------|
| {compound}_mcl_exceed | integer (0/1) | Maximum exceeds individual MCL |
| any_mcl_exceedance | integer (0/1) | Any of 5 individual MCLs exceeded |

## Hazard Index

HBWC values: PFHxS = 10 ng/L, PFNA = 10 ng/L, HFPO-DA = 10 ng/L, PFBS = 2,000 ng/L

| Column | Type | Description |
|--------|------|-------------|
| HI_value | float | Sum of (max conc / HBWC) for HI components |
| hi_exceed | integer (0/1) | HI > 1.0 with 2+ HI-component detections |

## Proximity Metrics (km)

| Column | Type | Source |
|--------|------|--------|
| dist_superfund | float | EPA PFAS Analytic Tools (n=464 sites) |
| dist_spills | float | Documented spills (n=1,378) |
| dist_federal | float | Federal/military installations |
| dist_industry | float | Industry sector facilities (n=208,205) |

## Demographic Variables (EPA ECHO, 3-mile buffer)

| Column | Type | Description |
|--------|------|-------------|
| pct_minority | float | Percent minority population |
| pct_lowincome | float | Percent low-income population |
| median_income | float | Median household income (USD) |

## Treatment

| Column | Type | Description |
|--------|------|-------------|
| pfas_treatment | integer (0/1) | Documented PFAS treatment |
| treatment_type | string | Technology type (GAC, IEX, NRO, PAC) |

## Notes

- All concentrations in micrograms per liter (ug/L = ppb)
- MCL/HBWC thresholds per April 2024 NPDWR final rule
- Detection = analytical result at/above compound-specific MRL
- Missing values coded as NaN
- Two systems excluded from original 10,299 (lithium-only monitoring)
