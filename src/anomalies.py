import numpy as np
import pandas as pd

from src.config import DATA_DIR, OUT_DIR


def _rolling_idi_regime_shift(y: np.ndarray, dates: np.ndarray, min_nz: int = 8):
    """Detects significant shifts in inter-demand interval (ordering-cycle changes)."""
    nz_idx = np.where((~np.isnan(y)) & (y > 0))[0]
    if len(nz_idx) < min_nz:
        return None
    gaps = np.diff(nz_idx)
    if len(gaps) < 6:
        return None

    best_score, best_split = 0.0, None
    for k in range(3, len(gaps) - 2):
        left, right = gaps[:k], gaps[k:]
        m1, m2 = left.mean(), right.mean()
        pooled_std = np.sqrt(0.5 * (left.var() + right.var())) + 0.25
        diff = abs(m2 - m1)
        score = diff / pooled_std
        if diff >= 0.75 and score > best_score:
            best_score = score
            best_split = (nz_idx[k], m1, m2)

    if best_split is not None and best_score >= 2.2:
        idx, m1, m2 = best_split
        return dates[idx], float(m1), float(m2), float(best_score)
    return None


def _cusum_structural_break(y: np.ndarray, dates: np.ndarray, min_len: int = 24):
    """Detects persistent level shifts (mean step-change) via standardized CUSUM."""
    valid = ~np.isnan(y)
    if valid.sum() < min_len:
        return None
    yv = y[valid]
    dv = dates[valid]
    if yv.std() < 1e-6:
        return None

    best_t, best_info = 0.0, None
    for k in range(12, len(yv) - 6):
        left, right = yv[:k], yv[k:]
        m1, m2 = left.mean(), right.mean()
        se = np.sqrt(left.var() / len(left) + right.var() / len(right)) + 1e-6
        t_stat = abs(m2 - m1) / se
        rel_change = (m2 - m1) / (abs(m1) + 1.0)
        if t_stat > best_t and abs(rel_change) >= 0.35:
            best_t = t_stat
            best_info = (dv[k], float(m1), float(m2), float(rel_change), float(t_stat))

    if best_info is not None and best_t >= 4.5:
        return best_info
    return None


def reconstruct_unconstrained_demand(
    train_clean: pd.DataFrame, sku_master: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Workstream 2 (Rectified V2):
      1. 3-Layer Unconstrained Demand Reconstruction:
         - Layer 1: Documented stockouts (inventory_stockouts.csv -> Requested_Units)
         - Layer 2: Backorder echo removal in non-stockout month t+1
         - Layer 3: Undocumented stockout uplift (2021-03..2022-09 ELA chip shortage)
      2. Real Event & Anomaly Register:
         - Documented/undocumented stockouts, backorder echoes, pre-buying spikes (EV-030),
           COVID/strike/logistics/capacity events, ERP migration gaps (EV-033), CUSUM
           structural breaks, ordering-cycle shifts, customer switches, lost accounts,
           master-vs-transaction contradictions, and positive-only rolling log-MAD outliers
           (prevents recurring holiday/cycle zeros from being misflagged as outliers).
    """
    df = train_clean.copy()
    df = df.sort_values(["SKU_ID", "Region", "Date"]).reset_index(drop=True)
    stockouts = pd.read_csv(DATA_DIR / "inventory_stockouts.csv")
    anomaly_rows = []

    # ------------------------------------------------------------------
    # LAYER 1: Documented Stockouts -> Replace with Requested_Units
    # ------------------------------------------------------------------
    st_cols = [
        "Date",
        "SKU_ID",
        "Region",
        "Requested_Units",
        "Carried_In_Units",
        "Shipped_Units",
        "Backordered_Units",
        "Lost_Units",
        "Fill_Rate_Pct",
        "Stockout_Days",
        "Reason",
    ]
    df = df.merge(stockouts[st_cols], on=["Date", "SKU_ID", "Region"], how="left")
    df["Is_Doc_Stockout"] = df["Requested_Units"].notna()
    df["Unconstrained_Demand"] = df["Units_Sold"].astype(float)
    df["Demand_Source"] = "observed_clean"
    df["Sample_Weight"] = 1.0

    doc_mask = df["Is_Doc_Stockout"]
    df.loc[doc_mask, "Unconstrained_Demand"] = df.loc[doc_mask, "Requested_Units"].astype(float)
    df.loc[doc_mask, "Demand_Source"] = "documented_stockout_requested"

    for r in df.loc[
        doc_mask,
        [
            "Date",
            "SKU_ID",
            "Region",
            "Product_Family",
            "Units_Sold",
            "Requested_Units",
            "Lost_Units",
            "Backordered_Units",
            "Reason",
        ],
    ].itertuples(index=False):
        anomaly_rows.append(
            {
                "Date": r.Date,
                "SKU_ID": r.SKU_ID,
                "Region": r.Region,
                "Product_Family": r.Product_Family,
                "Anomaly_Type": "Documented Stockout",
                "Severity": "High" if r.Lost_Units > 0 else "Medium",
                "Observed_Units": r.Units_Sold,
                "Adjusted_Units": r.Requested_Units,
                "Reason": f"Supply constraint ({r.Reason}); lost={r.Lost_Units}, backordered={r.Backordered_Units}",
                "Action_Taken": "Replaced Units_Sold with Requested_Units for unconstrained training",
            }
        )

    # ------------------------------------------------------------------
    # LAYER 2: Backorder Echo Subtraction on Next Non-Stockout Month t+1
    # ------------------------------------------------------------------
    df["Prev_Backordered"] = (
        df.groupby(["SKU_ID", "Region"])["Backordered_Units"].shift(1).fillna(0.0)
    )
    cur_ym = df["Year"] * 12 + df["Month"]
    prev_ym = df.groupby(["SKU_ID", "Region"])["Year"].shift(1) * 12 + df.groupby(
        ["SKU_ID", "Region"]
    )["Month"].shift(1)
    is_consec = (cur_ym - prev_ym) == 1

    echo_mask = (
        (~df["Is_Doc_Stockout"])
        & is_consec
        & (df["Prev_Backordered"] > 0)
        & df["Unconstrained_Demand"].notna()
    )
    df["Backorder_Echo_Removed"] = 0.0
    echo_deduct = np.minimum(
        df.loc[echo_mask, "Prev_Backordered"], df.loc[echo_mask, "Unconstrained_Demand"]
    )
    df.loc[echo_mask, "Backorder_Echo_Removed"] = echo_deduct
    df.loc[echo_mask, "Unconstrained_Demand"] = (
        df.loc[echo_mask, "Unconstrained_Demand"] - echo_deduct
    ).clip(lower=0.0)
    df.loc[echo_mask, "Demand_Source"] = "backorder_echo_adjusted"

    for r in df.loc[
        echo_mask,
        [
            "Date",
            "SKU_ID",
            "Region",
            "Product_Family",
            "Units_Sold",
            "Unconstrained_Demand",
            "Backorder_Echo_Removed",
        ],
    ].itertuples(index=False):
        anomaly_rows.append(
            {
                "Date": r.Date,
                "SKU_ID": r.SKU_ID,
                "Region": r.Region,
                "Product_Family": r.Product_Family,
                "Anomaly_Type": "Backorder Echo",
                "Severity": "Medium",
                "Observed_Units": r.Units_Sold,
                "Adjusted_Units": r.Unconstrained_Demand,
                "Reason": f"Prior-month backorder carry-in ({r.Backorder_Echo_Removed:.0f} units) inflated shipments",
                "Action_Taken": "Subtracted prior Backordered_Units from non-stockout month t+1",
            }
        )

    # ------------------------------------------------------------------
    # LAYER 3: Undocumented ELA Semiconductor Shortage (EV-019: 2021-03..2022-09)
    # ------------------------------------------------------------------
    ela_window = (
        (df["Product_Family"] == "Electrical Assemblies")
        & (df["Date"] >= "2021-03")
        & (df["Date"] <= "2022-09")
    )
    doc_ela = df[ela_window & df["Is_Doc_Stockout"]].copy()
    monthly_fill = (
        doc_ela.groupby("Date")
        .apply(
            lambda g: g["Shipped_Units"].sum() / max(g["Requested_Units"].sum(), 1.0),
            include_groups=False,
        )
        .clip(0.35, 0.95)
        .to_dict()
    )
    global_ela_fill = float(np.median(list(monthly_fill.values()))) if monthly_fill else 0.58

    normal_ela = df[
        (df["Product_Family"] == "Electrical Assemblies")
        & (~ela_window)
        & (~df["Date"].between("2020-02", "2020-06"))
        & (df["Unconstrained_Demand"] > 0)
    ]
    ela_base = (
        normal_ela.groupby(["SKU_ID", "Region"])["Unconstrained_Demand"]
        .median()
        .rename("ELA_Normal_Med")
    )
    df = df.merge(ela_base, on=["SKU_ID", "Region"], how="left")

    undoc_mask = (
        ela_window
        & (~df["Is_Doc_Stockout"])
        & (df["Unconstrained_Demand"] > 0)
        & (df["ELA_Normal_Med"].notna())
        & (df["Unconstrained_Demand"] < 0.78 * df["ELA_Normal_Med"])
    )
    df["Is_Undoc_Stockout"] = undoc_mask
    if undoc_mask.any():
        est_fill = df.loc[undoc_mask, "Date"].map(monthly_fill).fillna(global_ela_fill)
        uplifted = (df.loc[undoc_mask, "Unconstrained_Demand"] / est_fill).round()
        uplifted = np.minimum(uplifted, (1.5 * df.loc[undoc_mask, "ELA_Normal_Med"]).round())
        df.loc[undoc_mask, "Unconstrained_Demand"] = uplifted
        df.loc[undoc_mask, "Demand_Source"] = "undocumented_ela_uplift"
        df.loc[undoc_mask, "Sample_Weight"] = 0.75

        for r in df.loc[
            undoc_mask,
            [
                "Date",
                "SKU_ID",
                "Region",
                "Product_Family",
                "Units_Sold",
                "Unconstrained_Demand",
                "ELA_Normal_Med",
            ],
        ].itertuples(index=False):
            anomaly_rows.append(
                {
                    "Date": r.Date,
                    "SKU_ID": r.SKU_ID,
                    "Region": r.Region,
                    "Product_Family": r.Product_Family,
                    "Anomaly_Type": "Undocumented Stockout (EV-019)",
                    "Severity": "High",
                    "Observed_Units": r.Units_Sold,
                    "Adjusted_Units": r.Unconstrained_Demand,
                    "Reason": f"Depressed ELA shipment during chip shortage vs normal median {r.ELA_Normal_Med:.1f}",
                    "Action_Taken": "Uplifted using monthly empirical ELA fill-rate curve; downweighted 0.75x",
                }
            )

    # ------------------------------------------------------------------
    # REAL EVENT FLAGGING & ANOMALY DETECTION
    # ------------------------------------------------------------------
    prebuy_mask = df["Date"].isin(["2022-11", "2022-12"])
    df["Is_Prebuy_Window"] = prebuy_mask
    df.loc[prebuy_mask, "Sample_Weight"] = np.minimum(df.loc[prebuy_mask, "Sample_Weight"], 0.70)

    covid_mask = (
        ((df["Region"] == "Asia Pacific") & df["Date"].between("2020-02", "2020-03"))
        | ((df["Region"] == "Europe") & df["Date"].between("2020-03", "2020-05"))
        | (
            df["Region"].isin(["India", "Latin America", "North America"])
            & df["Date"].between("2020-04", "2020-06")
        )
    )
    strike_mask = (
        (df["Customer_Name"] == "Nexon Motor Group")
        & (df["Region"] == "Europe")
        & df["Date"].isin(["2023-10", "2023-11"])
    )
    red_sea_mask = (df["Sales_Channel"] == "Export") & df["Date"].between("2024-01", "2024-03")
    erp_gap_mask = (
        (df["Region"] == "Latin America")
        & df["Date"].between("2023-03", "2023-05")
        & df["Unconstrained_Demand"].isna()
    )

    df["Is_Known_Event"] = (
        doc_mask | undoc_mask | prebuy_mask | covid_mask | strike_mask | red_sea_mask
    )
    df.loc[covid_mask, "Sample_Weight"] = 0.25
    df.loc[strike_mask, "Sample_Weight"] = 0.50
    df.loc[red_sea_mask, "Sample_Weight"] = 0.75

    # Log explicit calendar disruption summaries (EV-009..013 COVID, EV-038 Strike, EV-039 Red Sea, EV-052 Helios)
    for dt_c, reg_c in [
        ("2020-02", "Asia Pacific"),
        ("2020-03", "Europe"),
        ("2020-04", "North America"),
        ("2020-04", "India"),
        ("2020-04", "Latin America"),
    ]:
        sub_c = df[covid_mask & (df["Region"] == reg_c)]
        if not sub_c.empty:
            anomaly_rows.append(
                {
                    "Date": dt_c,
                    "SKU_ID": "REGION-WIDE",
                    "Region": reg_c,
                    "Product_Family": "All Families",
                    "Anomaly_Type": "COVID-19 Disruption (EV-009..013)",
                    "Severity": "High",
                    "Observed_Units": float(sub_c["Units_Sold"].sum()),
                    "Adjusted_Units": float(sub_c["Unconstrained_Demand"].sum()),
                    "Reason": f"Pandemic lockdown affecting {len(sub_c)} rows in {reg_c}",
                    "Action_Taken": "Encoded COVID_Disruption flag; downweighted 0.25x in training",
                }
            )

    if strike_mask.any():
        sub_s = df[strike_mask]
        anomaly_rows.append(
            {
                "Date": "2023-10",
                "SKU_ID": "ACCOUNT-WIDE",
                "Region": "Europe",
                "Product_Family": "All Families",
                "Anomaly_Type": "Customer Labour Strike (EV-038)",
                "Severity": "High",
                "Observed_Units": float(sub_s["Units_Sold"].sum()),
                "Adjusted_Units": float(sub_s["Unconstrained_Demand"].sum()),
                "Reason": f"Nexon Motor Group | Europe strike ({len(sub_s)} rows); used to calibrate EV-062",
                "Action_Taken": "Encoded Cust_Shutdown_Flag; downweighted 0.50x in training",
            }
        )

    if red_sea_mask.any():
        sub_rs = df[red_sea_mask]
        anomaly_rows.append(
            {
                "Date": "2024-01",
                "SKU_ID": "CHANNEL-WIDE",
                "Region": "Export",
                "Product_Family": "All Families",
                "Anomaly_Type": "Logistics Disruption (EV-039)",
                "Severity": "Medium",
                "Observed_Units": float(sub_rs["Units_Sold"].sum()),
                "Adjusted_Units": float(sub_rs["Unconstrained_Demand"].sum()),
                "Reason": f"Red Sea shipping disruption on Export channel ({len(sub_rs)} rows)",
                "Action_Taken": "Encoded Channel_Disruption flag; downweighted 0.75x in training",
            }
        )

    for r in df.loc[erp_gap_mask, ["Date", "SKU_ID", "Region", "Product_Family"]].itertuples(
        index=False
    ):
        anomaly_rows.append(
            {
                "Date": r.Date,
                "SKU_ID": r.SKU_ID,
                "Region": r.Region,
                "Product_Family": r.Product_Family,
                "Anomaly_Type": "Missing Data (ERP Migration EV-033)",
                "Severity": "Medium",
                "Observed_Units": np.nan,
                "Adjusted_Units": np.nan,
                "Reason": "Latin America ERP migration gap (2023-03..2023-05)",
                "Action_Taken": "Preserved as NaN (excluded from training loss; never imputed as zero)",
            }
        )

    # ------------------------------------------------------------------
    # SERIES-LEVEL DETECTIONS: Customer switches, Lost accounts,
    # Cycle shifts, CUSUM breaks, and Positive-Only Rolling Log-MAD Outliers
    # ------------------------------------------------------------------
    df["Is_Outlier"] = False
    for (sku, reg), g in df.groupby(["SKU_ID", "Region"], sort=False):
        dates = g["Date"].values
        y = g["Unconstrained_Demand"].values
        known_ev = g["Is_Known_Event"].values
        fam = g["Product_Family"].iloc[0]

        # 1. Customer switches
        custs = g["Customer_Name"].values
        changes = np.where(custs[1:] != custs[:-1])[0]
        for ci in changes:
            anomaly_rows.append(
                {
                    "Date": dates[ci + 1],
                    "SKU_ID": sku,
                    "Region": reg,
                    "Product_Family": fam,
                    "Anomaly_Type": "Customer Switch",
                    "Severity": "Medium",
                    "Observed_Units": y[ci + 1],
                    "Adjusted_Units": y[ci + 1],
                    "Reason": f"Account transitioned from {custs[ci]} to {custs[ci + 1]}",
                    "Action_Taken": "Encoded Customer_Name & post-switch level via EWM/recency features",
                }
            )

        # 2. Lost accounts (>= 12 trailing zeros through 2026-10)
        valid_pos = np.where((~np.isnan(y)) & (y > 0))[0]
        if len(valid_pos) >= 6 and dates[-1] >= "2026-08":
            last_pos_idx = valid_pos[-1]
            trailing_zeros = len(y) - 1 - last_pos_idx
            if trailing_zeros >= 12:
                anomaly_rows.append(
                    {
                        "Date": dates[last_pos_idx + 1],
                        "SKU_ID": sku,
                        "Region": reg,
                        "Product_Family": fam,
                        "Anomaly_Type": "Lost / Dormant Account",
                        "Severity": "High",
                        "Observed_Units": 0.0,
                        "Adjusted_Units": 0.0,
                        "Reason": f"Zero orders for final {trailing_zeros} months after active sales (last order {dates[last_pos_idx]})",
                        "Action_Taken": "Captured via Trailing_Zero_Months & Hurdle probability zero-lock",
                    }
                )

        # 3. Ordering-cycle shift
        idi_shift = _rolling_idi_regime_shift(y, dates)
        if idi_shift is not None:
            dt_s, m1, m2, sc = idi_shift
            anomaly_rows.append(
                {
                    "Date": dt_s,
                    "SKU_ID": sku,
                    "Region": reg,
                    "Product_Family": fam,
                    "Anomaly_Type": "Ordering-Cycle Shift",
                    "Severity": "Medium",
                    "Observed_Units": np.nan,
                    "Adjusted_Units": np.nan,
                    "Reason": f"Mean inter-demand interval shifted from {m1:.2f} to {m2:.2f} months (z={sc:.1f})",
                    "Action_Taken": "Recent-24m IDI & cycle phase track current ordering cadence",
                }
            )

        # 4. CUSUM structural break
        sb = _cusum_structural_break(y, dates)
        if sb is not None:
            dt_b, m1, m2, rel, t_val = sb
            anomaly_rows.append(
                {
                    "Date": dt_b,
                    "SKU_ID": sku,
                    "Region": reg,
                    "Product_Family": fam,
                    "Anomaly_Type": "Structural Break (Level Shift)",
                    "Severity": "High" if abs(rel) > 0.6 else "Medium",
                    "Observed_Units": m2,
                    "Adjusted_Units": m2,
                    "Reason": f"Persistent mean shift {m1:.1f} -> {m2:.1f} ({rel:+.0%}, t={t_val:.1f})",
                    "Action_Taken": "Retained real shift; EWM level & 18m log-slope adapt to post-break regime",
                }
            )

        # 5. Positive-Only Rolling Log-MAD Outlier Detection
        # Ignores recurring holiday/cycle zeros completely and flags only true >=2.3x one-off shocks
        pos_idx = np.where((~np.isnan(y)) & (y > 0) & (~known_ev))[0]
        if len(pos_idx) >= 18:
            yp = y[pos_idx]
            ly = np.log1p(yp)
            rmed = pd.Series(ly).rolling(13, min_periods=5, center=True).median().values
            diff = ly - rmed
            d_med = np.median(diff)
            mad = max(float(np.median(np.abs(diff - d_med))), 0.12)
            z = np.abs(diff - d_med) / (1.4826 * mad)

            # Flag genuine extreme shocks (z > 5.0 AND >2.3x deviation from local positive median)
            hits = np.where((z > 5.0) & (np.abs(diff) > 0.85))[0]
            for hi in hits:
                orig_pos = pos_idx[hi]
                row_idx = g.index[orig_pos]
                dt_o = dates[orig_pos]
                local_med = float(np.expm1(rmed[hi]))
                ratio = yp[hi] / max(local_med, 1.0)
                df.at[row_idx, "Sample_Weight"] = min(df.at[row_idx, "Sample_Weight"], 0.45)
                df.at[row_idx, "Is_Outlier"] = True

                anomaly_rows.append(
                    {
                        "Date": dt_o,
                        "SKU_ID": sku,
                        "Region": reg,
                        "Product_Family": fam,
                        "Anomaly_Type": "One-Off Demand Spike / Dip",
                        "Severity": "High" if z[hi] > 7.0 else "Medium",
                        "Observed_Units": float(yp[hi]),
                        "Adjusted_Units": float(yp[hi]),
                        "Reason": f"Positive log-MAD shock: {yp[hi]:.0f} vs local median {local_med:.1f} ({ratio:.2f}x, z={z[hi]:.1f})",
                        "Action_Taken": "Downweighted 0.45x in training loss to prevent level distortion",
                    }
                )

        # Log significant EV-030 pre-buying spikes separately
        pb_idx = np.where((~np.isnan(y)) & (y > 0) & (np.isin(dates, ["2022-11", "2022-12"])))[0]
        if len(pb_idx) > 0 and len(pos_idx) >= 12:
            s_med = float(np.median(y[pos_idx]))
            for pi in pb_idx:
                if y[pi] > max(s_med * 1.45, s_med + 15.0):
                    anomaly_rows.append(
                        {
                            "Date": dates[pi],
                            "SKU_ID": sku,
                            "Region": reg,
                            "Product_Family": fam,
                            "Anomaly_Type": "Pre-Buy Spike (EV-030)",
                            "Severity": "Medium",
                            "Observed_Units": float(y[pi]),
                            "Adjusted_Units": float(y[pi]),
                            "Reason": f"Forward buying ahead of Jan 2023 +9% hike ({y[pi]:.0f} vs median {s_med:.1f})",
                            "Action_Taken": "Modeled via Pre_Hike_Signal_Pct; downweighted 0.70x",
                        }
                    )

    # 6. Log Master Data vs Transactions contradiction (BSS-018 early sales)
    df_launch = df.merge(sku_master[["SKU_ID", "Launch_Date"]], on="SKU_ID", how="left")
    early = df_launch[df_launch["Date"] < df_launch["Launch_Date"]]
    for sku, g in early.groupby("SKU_ID"):
        anomaly_rows.append(
            {
                "Date": g["Date"].min(),
                "SKU_ID": sku,
                "Region": "All Affected",
                "Product_Family": g["Product_Family"].iloc[0],
                "Anomaly_Type": "Master vs Transactions Contradiction",
                "Severity": "Medium",
                "Observed_Units": float(g["Units_Sold"].sum()),
                "Adjusted_Units": float(g["Units_Sold"].sum()),
                "Reason": f"{len(g)} valid rows prior to master Launch_Date={g['Launch_Date'].iloc[0]}",
                "Action_Taken": "Kept real transaction history; flagged master Launch_Date for MDM fix",
            }
        )

    df.drop(
        columns=["Prev_Backordered", "ELA_Normal_Med", "Is_Known_Event"],
        inplace=True,
        errors="ignore",
    )

    anomaly_df = pd.DataFrame(anomaly_rows)
    if not anomaly_df.empty:
        anomaly_df = anomaly_df.sort_values(["Date", "Severity", "SKU_ID"]).reset_index(drop=True)
    anomaly_df.to_csv(OUT_DIR / "anomaly_register.csv", index=False)
    df.to_parquet(OUT_DIR / "clean_train.parquet", index=False)

    return df, anomaly_df