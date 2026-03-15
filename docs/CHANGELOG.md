# Changelog

## Version 2.0 (February 2026) — Comprehensive Methodology Update

### Statistical Methods
- Replaced PCA with Multiple Correspondence Analysis (MCA) for binary detection matrix analysis
- Replaced custom Mann-Kendall with validated pymannkendall library (corrected tie handling)
- Replaced Pearson correlation proxy with proper Cochran-Armitage trend test
- Added Wilson score 95% CIs to all detection rate proportions
- Added McNemar's exact test with Benjamini-Hochberg FDR correction for UCMR 3 vs 5 comparison
- Added Mann-Whitney U tests for concentration distribution comparisons
- Added Hosmer-Lemeshow GOF test and logistic regression diagnostics
- Added Monte Carlo sensitivity analysis for HI exceedance (10,000 simulations)
- Added bootstrap CIs for population exposure estimates
- Added seasonal Kruskal-Wallis analysis

### Manuscript
- Updated all PCA terminology to MCA throughout main text and SI
- Fixed PFOS loading value error
- Removed duplicate sentence in Section 3.2
- Fixed citation tag error ([RGUELFO] to [R14])
- Added confidence intervals to all reported statistics
- Updated Mann-Kendall trend statistics to corrected values
- Added Hosmer-Lemeshow reference in regression results

### Figures
- Regenerated Figure 1 with Wilson CI error bars
- Regenerated Figure 3 forest plot with enhanced annotations
- Regenerated Figure 4 as MCA biplot (replacing PCA biplot)
- Added 5 new supplementary figures

### Supporting Information
- Updated all PCA references to MCA
- Added Text S11 (HI sensitivity analysis) and Text S12 (seasonal variation)
- Added Tables S22-S26 (regression diagnostics, McNemar, Kruskal-Wallis, HI sensitivity, MCA)
- Updated table count from 21 to 26, figure count from 5 to 10

## Version 1.0 (January 2026) — Initial Analysis

- Initial analysis using January 2026 UCMR 5 data update
- 6 analysis scripts covering core statistics, temporal trends, co-occurrence, UCMR 3 comparison, logistic regression, and population exposure
- 21 SI tables, 5 SI figures, 10 SI text sections
