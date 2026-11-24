import json
from pathlib import Path

import pandas as pd

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def test_clean_train_invariants():
    df = pd.read_parquet(OUT_DIR / "clean_train.parquet")
    sku = pd.read_parquet(OUT_DIR / "clean_sku_master.parquet")

    # Rule 1 & 2: Unique (Date, SKU_ID, Region) and valid Year/Month matching Date
    assert df.duplicated(["Date", "SKU_ID", "Region"]).sum() == 0
    assert (df["Year"] == df["Date"].str[:4].astype(int)).all()
    assert (df["Month"] == df["Date"].str[5:7].astype(int)).all()

    # No future 2062 transposed dates & 100% SKU_ID match against sku_master
    assert df["Date"].max() <= "2026-10"
    assert df["Date"].min() >= "2019-01"
    assert set(df["SKU_ID"].unique()).issubset(set(sku["SKU_ID"].unique()))

    # No negative Units_Sold; 0 unresolved Regions; Price * Units == Revenue where observed
    obs = df["Units_Sold"].notna()
    assert (df.loc[obs, "Units_Sold"] >= 0).all()
    assert df["Region"].isna().sum() == 0
    assert ((df.loc[obs, "Units_Sold"] * df.loc[obs, "Unit_Price_USD"] - df.loc[obs, "Revenue_USD"]).abs() <= 0.05).all()

    # One BU per Product_Family invariant
    assert sku.groupby("Product_Family")["Business_Unit"].nunique().max() == 1


def test_macro_cleaned():
    mac = pd.read_parquet(OUT_DIR / "clean_macro.parquet")
    assert mac.duplicated("Date").sum() == 0
    assert mac.isna().sum().sum() == 0
    assert mac["ISM_Manufacturing_PMI"].min() > 20.0
    assert mac.loc[mac["Year"] == 2019, "Auto_Production_YoY_Pct"].abs().max() > 1.0


def test_submission_and_business_rules():
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    sub = pd.read_csv(OUT_DIR / "submission.csv")
    fc = pd.read_parquet(OUT_DIR / "forecast_full.parquet")
    sku = pd.read_parquet(OUT_DIR / "clean_sku_master.parquet")

    # Exact row count, order, schema, non-negative integers, zero nulls
    assert list(sub.columns) == ["Date", "SKU_ID", "Region", "Units_Sold_Forecast"]
    assert len(sub) == len(test_df) == 58219
    assert sub["Units_Sold_Forecast"].isna().sum() == 0
    assert (sub["Units_Sold_Forecast"] >= 0).all()
    assert (sub["Units_Sold_Forecast"] % 1 == 0).all()
    assert (sub["Date"].values == test_df["Date"].values).all()
    assert (sub["SKU_ID"].values == test_df["SKU_ID"].values).all()
    assert (sub["Region"].values == test_df["Region"].values).all()

    # Rule 5: Strictly zero sales after EOL_Date and before Launch_Date
    m = sub.merge(sku[["SKU_ID", "Launch_Date", "EOL_Date"]], on="SKU_ID", how="left")
    post_eol = m["EOL_Date"].notna() & (m["Date"] > m["EOL_Date"])
    pre_launch = m["Launch_Date"].notna() & (m["Date"] < m["Launch_Date"])
    assert (m.loc[post_eol, "Units_Sold_Forecast"] == 0).all()
    assert (m.loc[pre_launch, "Units_Sold_Forecast"] == 0).all()

    # Planned Launch SKUs must receive non-zero transferred demand via Replaces_SKU
    for new_sku in ["PMP-020", "PTC-029", "THM-012"]:
        assert sub.loc[sub["SKU_ID"] == new_sku, "Units_Sold_Forecast"].sum() > 0

    # Quantile monotonicity: P10 <= P50 <= P90
    assert (fc["P10"] <= fc["P50"]).all()
    assert (fc["P50"] <= fc["P90"]).all()


def test_dq_report_valid_json():
    with open(OUT_DIR / "data_quality_report.json", "r", encoding="utf-8") as f:
        rep = json.load(f)
    assert "train.csv" in rep and "macro_indicators.csv" in rep and "sku_master.csv" in rep