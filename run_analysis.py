#!/usr/bin/env python3
"""
run_analysis.py
===============
End-to-end groundwater analysis pipeline.

Usage
-----
    python run_analysis.py                          # uses default data path
    python run_analysis.py --data path/to/file.csv  # custom CSV path
"""

import argparse
from pathlib import Path

from src.analysis import (
    load_data,
    inspect_data,
    plot_time_series,
    plot_correlation_matrix,
    plot_seasonal_annual,
    plot_pumping_vs_gwl_by_period,
    engineer_features,
    train_and_evaluate,
    plot_predictions,
    plot_residuals,
    plot_feature_importance,
    plot_elbow,
    fit_clusters,
    plot_stress_regimes,
    plot_pca_clusters,
    print_summary,
)

DEFAULT_DATA = Path("data/groundwater_urban_abstraction.csv")


def main(data_path: Path) -> None:
    print("\n🔷 1 / 7  Loading & inspecting data …")
    df = load_data(data_path)
    inspect_data(df)

    print("\n🔷 2 / 7  Exploratory visualisations …")
    plot_time_series(df)
    plot_correlation_matrix(df)
    plot_seasonal_annual(df)
    plot_pumping_vs_gwl_by_period(df)

    print("\n🔷 3 / 7  Feature engineering …")
    df_ml = engineer_features(df)

    print("\n🔷 4 / 7  Training ML models …")
    results, predictions, models, y_test, dates_test, _ = train_and_evaluate(df_ml)

    print("\n🔷 5 / 7  Model diagnostics …")
    plot_predictions(predictions, results, y_test, dates_test)
    plot_residuals(predictions, results, y_test)
    plot_feature_importance(models)

    print("\n🔷 6 / 7  Clustering — aquifer stress regimes …")
    plot_elbow(df)
    df = fit_clusters(df, k=4)
    plot_stress_regimes(df)
    plot_pca_clusters(df)

    print("\n🔷 7 / 7  Summary …")
    print_summary(df, results)

    print("\n✅ Done — figures saved to outputs/figures/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Groundwater Urban Abstraction Analysis")
    parser.add_argument(
        "--data", type=Path, default=DEFAULT_DATA,
        help="Path to the groundwater CSV file",
    )
    args = parser.parse_args()
    main(args.data)
