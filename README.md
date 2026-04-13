# Groundwater Level Analysis Under Urban Abstraction Pressure

**Quantifying 20 years of aquifer decline driven by intensive urban pumping (2005–2024)**

A data-driven hydrogeological study combining time-series analysis, machine learning forecasting, and unsupervised clustering to identify **Aquifer Stress Regimes** — an interpretable framework for tracking when urban abstraction pushes groundwater systems toward critical thresholds.

---

## 📊 Preview

![Time Series](outputs/figures/figure_01_timeseries.png)
![Stress Regimes](outputs/figures/figure_09_stress_regimes.png)

---

## 🔑 Key Findings

| Metric               | 2005–2009 | 2020–2024 | Change              |
|--------------------|----------|----------|---------------------|
| Mean GWL (mbgl)    | 9.7      | 20.3     | **+10.6 m deeper**  |
| Mean Pumping (ML/d)| ≈25      | ≈45      | **+79%**            |
| Mean EC (µS/cm)    | ≈480     | ≈560     | **+16%**            |

- Gradient Boosting achieves **very high short-term predictive accuracy (R² > 0.999)** for one-step-ahead forecasts, largely driven by strong temporal autocorrelation in groundwater levels  
- K-Means clustering reveals a progressive shift from **Low Stress** toward **Critical Stress** regimes over the 20-year period  

---

## 🧠 Key Insights

- Groundwater levels declined by over **10 metres**, indicating sustained overdraft  
- Pumping nearly **doubled**, significantly exceeding recharge trends  
- Rising electrical conductivity (EC) suggests **increasing salinity risk**  
- System behaviour shows a clear transition toward **critical stress conditions after ~2015**  

---

## 🌍 Why This Project Matters

Urban aquifers are increasingly stressed by abstraction exceeding natural recharge.  

This repository provides:

- Early warning signals of unsustainable groundwater use  
- Data-driven insights for urban water management  
- A transferable analytical framework for other aquifer systems  

---

## 🧪 Aquifer Stress Regimes

Aquifer Stress Regimes are derived using **K-Means clustering** on key indicators:

- Groundwater level (GWL)  
- Pumping rate  
- Electrical conductivity (EC)  

The resulting clusters are interpreted as:

- **Low Stress** — stable levels, low abstraction  
- **Moderate Stress** — early signs of decline  
- **High Stress** — sustained drawdown  
- **Critical Stress** — rapid decline and water quality deterioration  

This transforms raw monitoring data into **actionable management categories**.

---

## 📁 Repository Structure

```bash
groundwater-analysis/
├── data/
│   └── groundwater_urban_abstraction.csv
├── notebooks/
│   └── groundwater_analysis.ipynb
├── outputs/
│   └── figures/
├── requirements.txt
├── README.md
└── LICENSE
