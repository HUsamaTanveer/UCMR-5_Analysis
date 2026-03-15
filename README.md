# PFAS Occurrence in U.S. Drinking Water: A Comprehensive Analysis of EPA UCMR 5 Data

This repository contains the analysis code, processed output data, and figures for the manuscript submitted to *ACS ES&T Water*.

## Overview

Analysis of per- and polyfluoroalkyl substance (PFAS) occurrence in U.S. public water systems using EPA's Fifth Unregulated Contaminant Monitoring Rule (UCMR 5) dataset. The study examines 29 PFAS compounds across 10,297 public water systems comprising 1,863,306 analytical results.

## Repository Structure

```
UCMR-5_Analysis/
├── scripts/
│   ├── analysis/                # Core analysis scripts (run in order)
│   │   ├── 01_core_statistics.py        # Detection rates, Wilson CIs, geographic analysis
│   │   ├── 02_temporal_analysis.py      # Cochran-Armitage trends, Mann-Kendall tests
│   │   ├── 03_cooccurrence_mca.py       # Co-occurrence patterns, MCA with bootstrap CIs
│   │   ├── 04_ucmr3_comparison.py       # UCMR 3 vs. 5 matched-system McNemar's tests
│   │   ├── 05_regression.py             # Logistic regression with full diagnostics
│   │   ├── 06_raa_population.py         # Running annual average, population exposure
│   │   └── 07_ipw_sensitivity.py        # Inverse probability weighting sensitivity
│   └── figures/                 # Figure generation scripts
│       ├── Figure_1_Overview.py         # Main text Figures 1-6
│       ├── ...
│       └── Figure_S12_UCMR3_McNemar.py  # Supplementary Figures S1-S12
├── data/
│   ├── output/                  # Processed CSV outputs backing all manuscript numbers
│   └── README.md                # Raw data download instructions
├── figures/
│   ├── main/                    # Main text figures (PNG)
│   └── supplementary/           # Supplementary figures (PNG)
├── docs/
│   ├── data_dictionary.md       # Variable definitions for all output CSVs
│   ├── CHANGELOG.md             # Analysis version history
│   └── Reproducibility_Guide.docx  # Step-by-step reproduction instructions
├── .gitignore
├── requirements.txt
└── README.md
```

## Getting Started

### Requirements

- Python 3.10+
- Dependencies listed in `requirements.txt`

```bash
pip install -r requirements.txt
```

### Raw Data

Raw input data is not included in this repository due to size (~7.8 GB). See [`data/README.md`](data/README.md) for download links and instructions for obtaining all source datasets from EPA, Census Bureau, and other public sources.

### Running the Analysis

Analysis scripts in `scripts/analysis/` are numbered and should be run in order:

```bash
cd scripts/analysis
python 01_core_statistics.py
python 02_temporal_analysis.py
python 03_cooccurrence_mca.py
python 04_ucmr3_comparison.py
python 05_regression.py
python 06_raa_population.py
python 07_ipw_sensitivity.py
```

Figure scripts in `scripts/figures/` can be run after the analysis scripts have generated the output CSVs.

## Key Outputs

The `data/output/` directory contains all processed CSV files that back the numbers in the manuscript, including detection rates, regression coefficients, MCA coordinates, temporal trends, and sensitivity analyses. See `docs/data_dictionary.md` for complete variable definitions.

## License

This project is provided for academic reproducibility purposes. Please cite the associated manuscript if you use this code or data.

## Citation

> Tanveer, H.U. *PFAS Occurrence in U.S. Drinking Water: A Comprehensive Analysis of EPA UCMR 5 Monitoring Data.* Submitted to ACS ES&T Water, 2026.
