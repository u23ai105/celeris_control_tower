import difflib
import json
import re
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.config import (
    CANONICAL_CUSTOMERS,
    CHANNEL_MAP,
    DATA_DIR,
    FAMILY_TO_BU,
    MACRO_COLS,
    OUT_DIR,
    REGION_MAP,
    TRAIN_END,
    TRAIN_START,
)


def normalize_sku(s) -> str:
    if pd.isna(s):
        return np.nan
    s_clean = str(s).strip().upper().replace("_", "-")
    m = re.match(r"^([A-Z]{3})-?(\d+)$", s_clean)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):03d}"
    return s_clean


def normalize_date(s) -> Tuple[str, str]:
    """Returns (YYYY-MM, flag) where flag indicates format issue or transposition."""
    raw = str(s).strip()
    if re.match(r"^\d{4}-\d{2}$", raw):
        if raw.startswith("2062-"):
            return "2026-" + raw[5:], "transposed_year_2062"
        return raw, "ok"
    if re.match(r"^\d{4}/\d{2}$", raw):
        return raw.replace("/", "-"), "nonstandard_format"
    if re.match(r"^\d{4}-\d{2}-\d{2}$", raw):
        return raw[:7], "nonstandard_format"
    if re.match(r"^\d{2}/\d{4}$", raw):
        return f"{raw[3:]}-{raw[:2]}", "nonstandard_format"
    if re.match(r"^\d{2}-\d{4}$", raw):
        return f"{raw[3:]}-{raw[:2]}", "nonstandard_format"
    dt = pd.to_datetime(raw, errors="coerce")
    if pd.notna(dt):
        return dt.strftime("%Y-%m"), "nonstandard_format"
    return raw, "unparseable"


def build_customer_map(raw_names: pd.Series) -> Dict[str, str]:
    canon_lower = [c.lower() for c in CANONICAL_CUSTOMERS]
    mapping = {}
    for raw in raw_names.dropna().unique():
        cleaned = re.sub(r"\s+", " ", str(raw).strip())
        cleaned = re.sub(r"\s+Inc\.?$", "", cleaned, flags=re.IGNORECASE)
        low = cleaned.lower()
        if low in canon_lower:
            mapping[raw] = CANONICAL_CUSTOMERS[canon_lower.index(low)]
            continue
        match = difflib.get_close_matches(low, canon_lower, n=1, cutoff=0.6)
        if match:
            mapping[raw] = CANONICAL_CUSTOMERS[canon_lower.index(match[0])]
        else:
            # Handle missing-space typos like SolarisAutomotive
            no_space = [c.replace(" ", "") for c in canon_lower]
            m2 = difflib.get_close_matches(low.replace(" ", ""), no_space, n=1, cutoff=0.7)
            mapping[raw] = CANONICAL_CUSTOMERS[no_space.index(m2[0])] if m2 else cleaned
    return mapping


def clean_macro(report: dict) -> pd.DataFrame:
    raw = pd.read_csv(DATA_DIR / "macro_indicators.csv")
    issues = {}

    # 1. Deduplicate Date keeping latest revised print (2022-06 duplicate)
    dup_mask = raw.duplicated("Date", keep=False)
    issues["duplicate_date_rows_removed"] = int(raw.duplicated("Date", keep="last").sum())
    df = raw.drop_duplicates("Date", keep="last").sort_values("Date").reset_index(drop=True)

    # 2. Fix 2019 Auto_Production_YoY_Pct decimal-fraction bug (stored as -0.015 instead of -1.5)
    mask_2019_frac = (df["Year"] == 2019) & (df["Auto_Production_YoY_Pct"].abs() < 1.0)
    issues["auto_prod_2019_scale_100x_fixed"] = int(mask_2019_frac.sum())
    df.loc[mask_2019_frac, "Auto_Production_YoY_Pct"] *= 100.0

    # 3. Fix ISM_Manufacturing_PMI 10x decimal shift (2023-05 = 4.79 -> 47.9)
    mask_pmi_shift = df["ISM_Manufacturing_PMI"] < 15.0
    issues["pmi_decimal_shift_fixed"] = int(mask_pmi_shift.sum())
    df.loc[mask_pmi_shift, "ISM_Manufacturing_PMI"] *= 10.0

    # 4. Interpolate isolated missing values cleanly over time
    nan_count = int(df[MACRO_COLS].isna().sum().sum())
    issues["missing_cells_interpolated"] = nan_count
    for col in MACRO_COLS:
        df[col] = df[col].interpolate(method="linear", limit_direction="both")

    # Authoritative Year & Month from Date
    df["Year"] = df["Date"].str[:4].astype(int)
    df["Month"] = df["Date"].str[5:7].astype(int)

    report["macro_indicators.csv"] = {
        "rows_before": int(len(raw)),
        "rows_after": int(len(df)),
        "issues": issues,
    }
    return df


def clean_sku_master(report: dict) -> pd.DataFrame:
    raw = pd.read_csv(DATA_DIR / "sku_master.csv")
    df = raw.copy()
    issues = {}

    # Enforce 1-to-1 Product_Family -> Business_Unit invariant (fixes PMP-004 contradiction)
    expected_bu = df["Product_Family"].map(FAMILY_TO_BU)
    bu_mismatch = df["Business_Unit"] != expected_bu
    issues["business_unit_contradictions_fixed"] = int(bu_mismatch.sum())
    issues["business_unit_contradiction_skus"] = df.loc[bu_mismatch, "SKU_ID"].tolist()
    df.loc[bu_mismatch, "Business_Unit"] = expected_bu[bu_mismatch]

    report["sku_master.csv"] = {
        "rows_before": int(len(raw)),
        "rows_after": int(len(df)),
        "issues": issues,
    }
    return df


def clean_train(sku_master: pd.DataFrame, report: dict) -> pd.DataFrame:
    raw = pd.read_csv(DATA_DIR / "train.csv", low_memory=False)
    issues = {}
    dq_flags = []

    # 1. Deduplicate Record_ID keeping latest Load_Timestamp (Business Rule 2)
    raw_n = len(raw)
    df = raw.sort_values("Load_Timestamp").drop_duplicates("Record_ID", keep="last").copy()
    issues["duplicate_record_ids_removed"] = int(raw_n - len(df))

    # 2. Clean Date & fix transposed 2062 -> 2026
    parsed = df["Date"].apply(normalize_date)
    df["Date_Clean"] = [p[0] for p in parsed]
    date_flags = pd.Series([p[1] for p in parsed], index=df.index)
    issues["nonstandard_date_formats_fixed"] = int((date_flags == "nonstandard_format").sum())
    issues["transposed_year_2062_fixed"] = int((date_flags == "transposed_year_2062").sum())

    # 3. Authoritative Year/Month from Date (Business Rule 1)
    auth_y = df["Date_Clean"].str[:4].astype(int)
    auth_m = df["Date_Clean"].str[5:7].astype(int)
    ym_mismatch = (df["Year"] != auth_y) | (df["Month"] != auth_m)
    issues["year_month_disagreements_fixed"] = int(ym_mismatch.sum())
    df["Date"] = df["Date_Clean"]
    df["Year"] = auth_y
    df["Month"] = auth_m
    df.drop(columns=["Date_Clean"], inplace=True)

    # 4. Normalize SKU_ID
    sku_clean = df["SKU_ID"].apply(normalize_sku)
    issues["malformed_sku_ids_fixed"] = int((sku_clean != df["SKU_ID"]).sum())
    df["SKU_ID"] = sku_clean

    # 5. Canonicalize Sales_Channel & Customer_Name
    chan_clean = df["Sales_Channel"].astype(str).str.strip().str.lower().map(CHANNEL_MAP)
    issues["sales_channel_variants_fixed"] = int((chan_clean != df["Sales_Channel"]).sum())
    df["Sales_Channel"] = chan_clean

    cust_map = build_customer_map(df["Customer_Name"])
    cust_clean = df["Customer_Name"].map(cust_map)
    issues["customer_name_variants_fixed"] = int((cust_clean != df["Customer_Name"]).sum())
    df["Customer_Name"] = cust_clean

    # 6. Canonicalize Region & recover missing Regions deterministically
    reg_raw_notna = df["Region"].notna()
    reg_clean = df["Region"].astype(str).str.strip().str.lower().map(REGION_MAP)
    issues["region_label_variants_fixed"] = int(
        (reg_raw_notna & (reg_clean != df["Region"])).sum()
    )
    df["Region"] = reg_clean

    # Step 6a: Recover missing Region by single missing (Date, SKU) slot
    sku_regions = (
        df[df["Region"].notna()].groupby("SKU_ID")["Region"].apply(set).to_dict()
    )
    present_ds = (
        df[df["Region"].notna()].groupby(["Date", "SKU_ID"])["Region"].apply(set).to_dict()
    )
    # Step 6b: Fallback (SKU, Customer, Channel) -> modal Region
    scc_reg = (
        df[df["Region"].notna()]
        .groupby(["SKU_ID", "Customer_Name", "Sales_Channel"])["Region"]
        .agg(lambda x: x.mode().iloc[0])
        .to_dict()
    )
    sc_reg = (
        df[df["Region"].notna()]
        .groupby(["SKU_ID", "Customer_Name"])["Region"]
        .agg(lambda x: x.mode().iloc[0])
        .to_dict()
    )

    miss_reg_idx = df[df["Region"].isna()].index
    rec_slot, rec_scc, rec_unres = 0, 0, 0
    for idx in miss_reg_idx:
        row = df.loc[idx]
        all_r = set(sku_regions.get(row["SKU_ID"], set()))
        if row["Date"] < "2022-01":
            all_r.discard("Middle East & Africa")
        pres = present_ds.get((row["Date"], row["SKU_ID"]), set())
        diff = all_r - pres
        if len(diff) == 1:
            chosen = next(iter(diff))
            df.at[idx, "Region"] = chosen
            present_ds.setdefault((row["Date"], row["SKU_ID"]), set()).add(chosen)
            rec_slot += 1
        else:
            key3 = (row["SKU_ID"], row["Customer_Name"], row["Sales_Channel"])
            key2 = (row["SKU_ID"], row["Customer_Name"])
            cand = scc_reg.get(key3, sc_reg.get(key2, None))
            if cand is not None and (cand in diff or len(diff) == 0):
                df.at[idx, "Region"] = cand
                present_ds.setdefault((row["Date"], row["SKU_ID"]), set()).add(cand)
                rec_scc += 1
            elif len(diff) > 0:
                chosen = sorted(diff)[0]
                df.at[idx, "Region"] = chosen
                present_ds.setdefault((row["Date"], row["SKU_ID"]), set()).add(chosen)
                rec_scc += 1
            else:
                rec_unres += 1

    issues["missing_regions_recovered_slot"] = rec_slot
    issues["missing_regions_recovered_context"] = rec_scc
    issues["missing_regions_unresolved"] = rec_unres

    # 7. Restore Product_Family, Business_Unit, SKU_Description from clean sku_master
    mst_lookup = sku_master.set_index("SKU_ID")[
        ["Product_Family", "Business_Unit", "SKU_Description"]
    ]
    issues["missing_product_family_filled"] = int(df["Product_Family"].isna().sum())
    issues["missing_sku_description_filled"] = int(df["SKU_Description"].isna().sum())
    df["Product_Family"] = df["SKU_ID"].map(mst_lookup["Product_Family"])
    df["Business_Unit"] = df["SKU_ID"].map(mst_lookup["Business_Unit"])
    df["SKU_Description"] = df["SKU_ID"].map(mst_lookup["SKU_Description"])

    # 8. Parse Units_Sold strings (dashes & commas) and fix negative sign flips
    u_raw_str = df["Units_Sold"].astype(str).str.strip()
    issues["dash_units_converted_to_nan"] = int((u_raw_str == "-").sum())
    issues["comma_formatted_units_parsed"] = int(u_raw_str.str.contains(",", na=False).sum())
    u_num = pd.to_numeric(u_raw_str.str.replace(",", "", regex=False), errors="coerce")

    neg_mask = u_num < 0
    issues["negative_units_sign_flipped"] = int(neg_mask.sum())
    df["Units_Sold_Raw_Num"] = u_num
    df["Units_Sold"] = u_num.abs()
    df["Cleaning_Action"] = "clean"
    df.loc[neg_mask, "Cleaning_Action"] = "sign_flip_fixed"

    # 9. Canonical USD Price Table per (SKU_ID, Price_Regime) per Rule 4 + EV-027 6% surcharge
    df["Price_Regime"] = df["Date"].str[:4]
    surcharge_mask = (df["Date"] >= "2022-07") & (df["Date"] <= "2022-12")
    df.loc[surcharge_mask, "Price_Regime"] = "2022-H2"

    pos_p = df[df["Unit_Price_USD"] > 0]
    canon_p = (
        pos_p.groupby(["SKU_ID", "Price_Regime"])["Unit_Price_USD"]
        .median()
        .rename("Canon_Price_USD")
    )
    df = df.merge(canon_p, on=["SKU_ID", "Price_Regime"], how="left")
    # Fallback for any SKU-regime with all-zero prices: interpolate SKU price across regimes
    sku_med_p = pos_p.groupby("SKU_ID")["Unit_Price_USD"].median()
    df["Canon_Price_USD"] = df["Canon_Price_USD"].fillna(df["SKU_ID"].map(sku_med_p))

    # Diagnose price/currency corruptions
    pr = df["Unit_Price_USD"] / df["Canon_Price_USD"]
    eur_mask = (df["Unit_Price_USD"] > 0) & pr.between(0.85, 0.93)
    inr10_mask = (df["Unit_Price_USD"] > 0) & pr.between(9.5, 10.5)
    cents_mask = (df["Unit_Price_USD"] > 0) & pr.between(95.0, 105.0)
    zero_p_pos_u = (df["Unit_Price_USD"] == 0) & (df["Units_Sold"] > 0)
    issues["eur_currency_errors_fixed"] = int(eur_mask.sum())
    issues["india_10x_price_errors_fixed"] = int(inr10_mask.sum())
    issues["cents_100x_price_errors_fixed"] = int(cents_mask.sum())
    issues["zero_price_on_positive_units_fixed"] = int(zero_p_pos_u.sum())

    # 10. Recover missing Units_Sold from Revenue_USD / Canon_Price_USD
    # Use raw Revenue_USD (adjusting EUR/10x if row also had currency error)
    eff_price = np.where(eur_mask | inr10_mask, df["Unit_Price_USD"], df["Canon_Price_USD"])
    miss_u_rev_pos = df["Units_Sold"].isna() & (df["Revenue_USD"] > 0)
    implied_u = df["Revenue_USD"] / eff_price
    df.loc[miss_u_rev_pos, "Units_Sold"] = implied_u[miss_u_rev_pos].round()
    df.loc[miss_u_rev_pos, "Cleaning_Action"] = "recovered_units_from_revenue"

    miss_u_rev_zero = df["Units_Sold"].isna() & (df["Revenue_USD"] == 0)
    df.loc[miss_u_rev_zero, "Units_Sold"] = 0.0
    df.loc[miss_u_rev_zero, "Cleaning_Action"] = "recovered_zero_from_revenue"

    issues["missing_units_recovered_positive"] = int(miss_u_rev_pos.sum())
    issues["missing_units_recovered_zero"] = int(miss_u_rev_zero.sum())
    issues["truly_missing_units_retained_nan"] = int(df["Units_Sold"].isna().sum())

    # 11. Reconcile 10x and 100x unit-of-measure errors against Revenue / Canon_Price
    valid_price = pr.between(0.98, 1.02)
    u_from_rev = df["Revenue_USD"] / df["Canon_Price_USD"]
    u_is_int = (u_from_rev - u_from_rev.round()).abs() < 0.05
    u10_mask = (
        valid_price
        & (df["Units_Sold"] > 0)
        & u_is_int
        & (df["Units_Sold"] / u_from_rev.replace(0, np.nan)).between(9.8, 10.2)
    )
    u100_mask = (
        valid_price
        & (df["Units_Sold"] > 0)
        & u_is_int
        & (df["Units_Sold"] / u_from_rev.replace(0, np.nan)).between(98.0, 102.0)
    )
    issues["units_10x_uom_errors_fixed"] = int(u10_mask.sum())
    issues["units_100x_uom_errors_fixed"] = int(u100_mask.sum())

    df.loc[u10_mask, "Units_Sold"] = u_from_rev[u10_mask].round()
    df.loc[u10_mask, "Cleaning_Action"] = "fixed_10x_unit_error"
    df.loc[u100_mask, "Units_Sold"] = u_from_rev[u100_mask].round()
    df.loc[u100_mask, "Cleaning_Action"] = "fixed_100x_unit_error"

    # 12. Reconcile Unit_Price_USD and Revenue_USD strictly
    rev_mismatch_before = (
        df["Units_Sold"].notna()
        & df["Revenue_USD"].notna()
        & ((df["Units_Sold"] * df["Canon_Price_USD"] - df["Revenue_USD"]).abs() > 1.0)
    )
    issues["revenue_reconciliations_applied"] = int(rev_mismatch_before.sum())
    df["Unit_Price_USD"] = df["Canon_Price_USD"].round(2)
    df["Revenue_USD"] = (df["Units_Sold"] * df["Unit_Price_USD"]).round(2)

    # Deduplicate safety check on (Date, SKU_ID, Region)
    key_dups = df.duplicated(["Date", "SKU_ID", "Region"], keep="last").sum()
    issues["post_recovery_key_dups_removed"] = int(key_dups)
    if key_dups > 0:
        df = df.sort_values("Load_Timestamp").drop_duplicates(
            ["Date", "SKU_ID", "Region"], keep="last"
        )

    # Filter strictly within valid history range
    df = df[(df["Date"] >= TRAIN_START) & (df["Date"] <= TRAIN_END)].copy()
    df = df.sort_values(["SKU_ID", "Region", "Date"]).reset_index(drop=True)

    # Flag BSS-018 early-launch contradiction (keep valid sales, log in report)
    df_l = df.merge(sku_master[["SKU_ID", "Launch_Date", "EOL_Date"]], on="SKU_ID", how="left")
    early_sales = df_l[df_l["Date"] < df_l["Launch_Date"]]
    issues["master_launch_date_contradiction_rows"] = int(len(early_sales))
    issues["master_launch_date_contradiction_skus"] = sorted(
        early_sales["SKU_ID"].unique().tolist()
    )

    report["train.csv"] = {
        "rows_before": int(raw_n),
        "rows_after": int(len(df)),
        "unique_skus": int(df["SKU_ID"].nunique()),
        "unique_sku_regions": int(df[["SKU_ID", "Region"]].drop_duplicates().shape[0]),
        "issues": issues,
    }
    return df


def run_cleaning() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    report = {}
    macro_clean = clean_macro(report)
    sku_clean = clean_sku_master(report)
    train_clean = clean_train(sku_clean, report)

    macro_clean.to_parquet(OUT_DIR / "clean_macro.parquet", index=False)
    sku_clean.to_parquet(OUT_DIR / "clean_sku_master.parquet", index=False)
    train_clean.to_parquet(OUT_DIR / "clean_train.parquet", index=False)

    with open(OUT_DIR / "data_quality_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return train_clean, macro_clean, sku_clean, report