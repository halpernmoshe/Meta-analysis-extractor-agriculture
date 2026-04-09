# Ground Truth Datasets

These files are the ground truth datasets used to validate the extraction pipeline.
All are redistributable under their original open-access licenses.

## Files

### loladze_co2_dataset.xlsx
- **Source**: Loladze 2014, eLife 3:e02245
- **License**: CC-BY 4.0 (eLife)
- **Download**: https://doi.org/10.7554/eLife.02245
- **Description**: 1,482 observations of mineral element concentrations in plants grown under elevated CO2. Contains columns for Reference, Element, eCO2, aCO2, and Additional Info.
- **Used by**: `validate_full_46.py`, `validate_agent_extraction.py`, `loladze_meta_comparison.py`

### hui2023_ground.xlsx
- **Source**: Hui et al. 2025, Nature Communications 16:3913
- **License**: CC-BY 4.0 (Nature Communications)
- **Download**: https://doi.org/10.1038/s41467-025-57895-1 (Supplementary Data)
- **Description**: Zinc biofortification data for wheat across soil, foliar, and soil+foliar application methods. Three data sheets with grain Zn concentration observations.
- **Used by**: `validate_hui2023.py`, `validate_hui2023_agent.py`, `hui_meta_comparison.py`

### li2022_Data_Sheet_2.xlsx
- **Source**: Li et al. 2022, Frontiers in Plant Science 13:836702
- **License**: CC-BY 4.0 (Frontiers)
- **Download**: https://doi.org/10.3389/fpls.2022.836702 (Supplementary Material)
- **Description**: Biostimulant effects on crop yield in field trials. Contains 1,108 observations across 181 studies with fresh yield data.
- **Used by**: `validate_li2022.py`, `validate_li2022_agent.py`, `li_meta_comparison.py`

## Environment Variable Overrides

If you prefer to store GT files elsewhere, set these environment variables:
- `GT_PATH_LOLADZE` - path to Loladze CO2 dataset
- `GT_PATH_HUI` - path to Hui 2023 ground truth
- `GT_PATH_LI` - path to Li 2022 data sheet
