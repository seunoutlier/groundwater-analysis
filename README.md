# Groundwater Level Analysis Under Urban Abstraction Pressure

Analysis of 20 years (2005–2024) of daily groundwater monitoring data to quantify aquifer decline driven by high urban pumping rates. Combines exploratory analysis, machine-learning prediction, and unsupervised clustering to identify aquifer stress regimes.

## Key Findings

| Metric | 2005–2009 | 2020–2024 | Change |
|---|---|---|---|
| Mean GWL (mbgl) | ~6 | ~16 | ~10 m deeper |
| Mean Pumping (ML/d) | ~25 | ~45 | ~80 % increase |
| Mean EC (µS/cm) | ~480 | ~680 | ~42 % increase |

- **Gradient Boosting** achieves R² > 0.999 for one-step-ahead GWL prediction
- K-Means clustering reveals a progressive shift from *Low Stress* toward *Critical Stress* regimes over the 20-year record

## Repository Structure

```
groundwater-analysis/
├── data/
│   └── groundwater_urban_abstraction.csv   # 7,300 daily records
├── notebooks/
│   └── groundwater_analysis.ipynb          # Interactive notebook
├── src/
│   ├── __init__.py
│   └── analysis.py                         # Core analysis module
├── outputs/
│   └── figures/                            # Auto-generated plots
├── run_analysis.py                         # CLI entry point
├── requirements.txt
├── LICENSE
└── README.md
```

## Quickstart

```bash
# Clone the repo
git clone https://github.com/<your-username>/groundwater-analysis.git
cd groundwater-analysis

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python run_analysis.py

# Or use a custom data path
python run_analysis.py --data path/to/your/file.csv
```

Figures are saved automatically to `outputs/figures/`.

For interactive exploration, open the Jupyter notebook:

```bash
jupyter notebook notebooks/groundwater_analysis.ipynb
```

## Dataset

Daily time series with 17 columns:

| Column | Description |
|---|---|
| `Date` | Date (2005-01-01 to 2024-12-31) |
| `GWL_mbgl` | Groundwater level, metres below ground level |
| `Pumping_Rate_MLd` | Abstraction rate (megalitres/day) |
| `Rainfall_mm` | Daily rainfall |
| `Temperature_C` | Daily mean temperature |
| `Effective_Recharge_mm` | Estimated aquifer recharge |
| `EC_uScm` | Electrical conductivity (water quality proxy) |
| `Nitrate_mgL` | Nitrate concentration |
| `Year`, `Month`, `DayOfYear`, `Weekday`, `Season` | Temporal features |
| `GWL_7d_avg`, `GWL_30d_avg` | Smoothed groundwater levels |
| `Pumping_7d_avg` | 7-day average pumping |
| `Rain_30d_sum` | 30-day cumulative rainfall |

## Analysis Pipeline

### 1. Exploratory Data Analysis
- Four-panel time-series overview (GWL, rainfall, pumping, EC)
- Correlation matrix of key hydro-geological variables
- Seasonal and annual trend decomposition
- Period-based pumping vs GWL scatter plots

### 2. Feature Engineering
Nine additional features are derived from the raw data: lagged GWL (1, 7, 30 days), lagged pumping, rolling rainfall sums, first-difference terms, a pumping–temperature interaction, and a cumulative abstraction deficit.

### 3. Machine Learning Models
Four models are trained on an 80/20 chronological split:

| Model | Description |
|---|---|
| Linear Regression | Baseline parametric model |
| Lasso (α = 0.01) | L1-regularised linear model |
| Random Forest | 300-tree ensemble, max depth 20 |
| Gradient Boosting | 500 estimators, learning rate 0.05 |

Evaluation uses MAE, RMSE, R², and 5-fold time-series cross-validation.

### 4. Clustering — Aquifer Stress Regimes
K-Means (k = 4) segments the daily records into four stress regimes based on GWL, pumping, rainfall, EC, and temperature. The elbow method confirms k = 4. PCA provides a 2-D projection of the cluster structure.

## Generated Figures

| # | Figure | Description |
|---|---|---|
| 01 | Time-series overview | GWL, rainfall, pumping, EC (2005–2024) |
| 02 | Correlation matrix | Pairwise correlations of 8 key variables |
| 03 | Seasonal & annual | Annual mean GWL/pumping + monthly GWL profile |
| 04 | Pumping vs GWL scatter | Colour-coded by 5-year period |
| 05 | Model predictions | Observed vs predicted for all 4 models |
| 06 | Residual analysis | Distribution + predicted-vs-observed (best model) |
| 07 | Feature importance | Gradient Boosting feature ranking |
| 08 | Elbow method | K-Means inertia curve |
| 09 | Stress regimes | Cluster scatter + yearly stacked-bar |
| 10 | PCA clusters | 2-D projection of stress regimes |

## Requirements

- Python ≥ 3.10
- pandas, numpy, matplotlib, scikit-learn

See `requirements.txt` for exact version constraints.

## License

MIT — see [LICENSE](LICENSE).
