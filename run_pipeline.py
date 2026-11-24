import time
from src.anomalies import reconstruct_unconstrained_demand
from src.cleaning import run_cleaning
from src.features import analyze_macro_lead_lag
from src.forecasting import run_forecasting_pipeline


def main():
    t0 = time.time()
    print("[1/4] Workstream 1: Running deterministic forensic cleaning...")
    train_clean, macro_clean, sku_clean, dq_report = run_cleaning()
    print(
        f"      Cleaned train.csv: {dq_report['train.csv']['rows_before']:,} -> "
        f"{dq_report['train.csv']['rows_after']:,} unique rows."
    )

    print("[2/4] Workstream 2: Reconstructing 3-layer unconstrained demand & anomaly register...")
    train_unconstrained, anomaly_df = reconstruct_unconstrained_demand(train_clean, sku_clean)
    print(f"      Logged {len(anomaly_df):,} anomaly/event records to outputs/anomaly_register.csv.")

    print("[3/4] Workstream 3: Running macro lead/lag discovery & regime stability tests...")
    lag_df, best_lag_map = analyze_macro_lead_lag(train_unconstrained, macro_clean)
    print(f"      Evaluated {len(lag_df)} (Family, Indicator, Lag) pairs -> outputs/macro_lead_lag_analysis.csv.")

    print("[4/4] Workstream 4: 3-origin 19-mo backtest, 4-model ensemble, MinT reconciliation & CQR...")
    _, metrics_df, forecast_df = run_forecasting_pipeline(
        train_unconstrained, macro_clean, sku_clean, best_lag_map
    )

    ov = metrics_df[metrics_df["Slice_Type"].isin(["Overall", "Calibration"])]
    print("\n=== BACKTEST SUMMARY (3-Origin x 19-Month Horizon) ===")
    print(ov.to_string(index=False))
    print(f"\nPipeline complete in {time.time() - t0:.1f}s.")
    print("Submission ready at: outputs/submission.csv")
    print(f"Total 19-month forecast units: {int(forecast_df['Units_Sold_Forecast'].sum()):,}")


if __name__ == "__main__":
    main()