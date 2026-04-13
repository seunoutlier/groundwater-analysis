# Groundwater Level Analysis Under Urban Abstraction Pressure

**Quantifying 20 years of aquifer decline driven by intensive urban pumping (2005–2024)**

A data-driven hydrogeological study that combines time-series analysis, machine learning forecasting, and unsupervised clustering to identify **Aquifer Stress Regimes** — an objective framework to track when urban abstraction pushes groundwater systems toward critical thresholds.

![Time Series Overview](outputs/figures/01_time_series_overview.png)

## Key Findings

| Metric                  | 2005–2009     | 2020–2024     | Change                  |
|-------------------------|---------------|---------------|-------------------------|
| Mean GWL (mbgl)        | 9.7          | 20.3         | **+10.6 m deeper**     |
| Mean Pumping (ML/d)    | ~25          | ~45          | **+79%**               |
| Mean EC (µS/cm)        | ~480         | ~560         | **+16%**               |

- **Gradient Boosting** achieves **R² > 0.999** for one-step-ahead GWL prediction
- K-Means clustering reveals a progressive shift from **Low Stress** toward **Critical Stress** regimes over the 20-year period

## Why This Project Matters

In urban aquifers, pumping often exceeds natural recharge, leading to declining water levels and deteriorating water quality. This repository provides:
- Clear visualisation of long-term trends
- Accurate ML-based forecasting of groundwater levels
- A **novel Aquifer Stress Regime** classification using unsupervised learning — turning raw monitoring data into actionable insights for sustainable urban water management

## Repository Structure

```bash
groundwater-analysis/
├── data/
│   └── groundwater_urban_abstraction.csv          # Raw daily dataset (~7300 records)
├── notebooks/
│   └── groundwater_analysis.ipynb                 # Complete interactive analysis
├── outputs/
│   └── figures/                                   # All 10 automatically generated figures
├── requirements.txt
├── README.md


Generated Figures




























































#FigureDescription01Time-series overviewGWL, pumping, rainfall & EC trends (2005–2024)02Correlation matrixRelationships between key hydrogeological variables03Annual & seasonal trendsLong-term decline and monthly GWL patterns04Pumping vs GWL scatterPeriod-coloured to show increasing stress over time05Model predictionsObserved vs predicted for all four models06Residual analysisDiagnostics for the best-performing model07Feature importanceWhat drives groundwater level changes08Elbow methodDetermining optimal number of stress regimes09Stress regimes (yearly)Progressive shift toward Critical Stress10PCA projection of regimes2D visualisation of cluster separation
Quick Start
Bash# 1. Clone the repository
git clone https://github.com/YOUR-USERNAME/groundwater-analysis.git
cd groundwater-analysis

# 2. (Recommended) Create virtual environment
python -m venv .venv
source .venv/bin/activate          # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the analysis
jupyter notebook notebooks/groundwater_analysis.ipynb
Running the notebook will automatically generate and save all figures to outputs/figures/.
Dataset
Daily time series (2005-01-01 to 2024-12-31) with 17 columns including:

GWL_mbgl — Groundwater level (metres below ground level)
Pumping_Rate_MLd — Urban abstraction rate (ML/day)
Rainfall_mm, Effective_Recharge_mm, Temperature_C
EC_uScm — Electrical conductivity (water quality / salinity proxy)
Temporal features and engineered variables (lagged GWL, rolling rainfall, pumping-temperature interaction, abstraction deficit)

Analysis Pipeline

Exploratory Data Analysis — Trends, correlations, seasonal decomposition
Feature Engineering — Lagged variables, interaction terms, and cumulative stress indicators
Machine Learning Forecasting — Linear Regression, Lasso, Random Forest, Gradient Boosting (using chronological train/test split)
Novel Clustering — K-Means to define four Aquifer Stress Regimes (Low, Moderate, High, Critical)

Requirements

Python ≥ 3.10
pandas, numpy, matplotlib, seaborn, scikit-learn, statsmodels

Exact versions are listed in requirements.txt.
Novelty & Impact
Most groundwater studies focus only on trend detection or basic regression. This project goes further by introducing Aquifer Stress Regimes — a simple yet powerful unsupervised learning approach that classifies daily conditions into practical management zones.
Useful for:

Urban water utilities and planners
Hydrogeologists studying over-exploited aquifers
Researchers applying machine learning to water resources

License
MIT License — feel free to use, adapt, and build upon this work for research or practical applications.

Contributing to sustainable urban groundwater management 🌍
Built with data science + hydrogeology
text### How to use:
1. Copy everything above.
2. Go to your GitHub repository → click **Add a README** (or edit the existing one).
3. Paste the content.
4. Replace `YOUR-USERNAME` with your actual GitHub username.
5. Commit the changes.

The images will display correctly once you upload the `outputs/figures/` folder to your repo.

Would you also like the exact `requirements.txt` content to complete the repository? Just reply with **"yes"**. 

Your project is now ready with a polished, professional README!
└── LICENSE
