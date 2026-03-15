# Raw Data Sources

Raw input data is not included in this repository due to size. Download the following datasets and place them in `data/raw/` (excluded from version control via `.gitignore`).

## Required Datasets

### 1. EPA UCMR 5 (Primary Dataset)
- **Source:** [EPA UCMR 5 Occurrence Data](https://www.epa.gov/dwucmr/occurrence-data-unregulated-contaminant-monitoring-rule#5)
- **Files needed:** `UCMR5_All.txt`, `UCMR5_ZIPCodes.txt`, `UCMR5_AddtlDataElem.txt`
- **Note:** Use the January 2025 update version

### 2. EPA UCMR 3 (Temporal Comparison)
- **Source:** [EPA UCMR 3 Occurrence Data](https://www.epa.gov/dwucmr/occurrence-data-unregulated-contaminant-monitoring-rule#3)
- **Files needed:** `UCMR3_All.txt`, `UCMR3_ZIPCodes.txt`

### 3. EPA SDWIS (Safe Drinking Water Information System)
- **Source:** [EPA SDWIS/Fed Data Downloads](https://www.epa.gov/ground-water-and-drinking-water/safe-drinking-water-information-system-sdwis-federal-reporting)
- **Files needed:** `SDWA_PUB_WATER_SYSTEMS.csv`, `SDWA_GEOGRAPHIC_AREAS.csv`, `SDWA_FACILITIES.csv`, `SDWA_SERVICE_AREAS.csv`, and other SDWA tables

### 4. U.S. Census Bureau (Demographics)
- **Source:** [ACS 5-Year Estimates 2023, Table DP05](https://data.census.gov/table/ACSDP5Y2023.DP05)
- **Files needed:** `ACSDP5Y2023.DP05-Data.csv`, `ACSDP5Y2023.DP05-Column-Metadata.csv`
- **Gazetteer:** [2023 ZCTA Gazetteer](https://www.census.gov/geographies/reference-files/time-series/geo/gazetteer-files.html)

### 5. EPA ECHO Demographics
- **Source:** [EPA ECHO Exporter](https://echo.epa.gov/tools/data-downloads)
- **Files needed:** `ECHO_DEMOGRAPHICS.csv`, `FRS_PROGRAM_LINKS.csv`

### 6. PFAS Source Data
- **Source:** Various EPA databases (TRI, Superfund, TSCA)
- **Files needed:** `TRI_OnSite.xlsx`, `Superfund_Sites.xlsx`, `Federal_sites.xlsx`, `Industry_Sectors.xlsx`, `Production_TSCA.xlsx`, `Spills.xlsx`

### 7. Census TIGER/Line Shapefiles
- **Source:** [Census Bureau Cartographic Boundary Files](https://www.census.gov/geographies/mapping-files/time-series/geo/cartographic-boundary.html)
- **Files needed:** `cb_2023_us_state_20m.*` (shapefile set)

## Directory Structure

After downloading, organize the files as:
```
data/raw/
├── UCMR5/
├── UCMR5_Jan2025_Update/
├── UCMR3/
├── SDWIS/
├── Census/
├── Demographics/
├── PFAS_Sources/
└── Shapefiles/
```
