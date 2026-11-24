import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

st.set_page_config(
    page_title="Celeris Automobili | Demand Forecasting Control Tower",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_artifacts():
    hist = pd.read_parquet(OUT_DIR / "clean_train.parquet")
    fc = pd.read_parquet(OUT_DIR / "forecast_full.parquet")
    macro = pd.read_parquet(OUT_DIR / "clean_macro.parquet")
    sku_mst = pd.read_parquet(OUT_DIR / "clean_sku_master.parquet")
    bt_metrics = pd.read_csv(OUT_DIR / "backtest_metrics.csv")
    bt_preds = pd.read_parquet(OUT_DIR / "backtest_predictions.parquet")
    anomalies = pd.read_csv(OUT_DIR / "anomaly_register.csv")
    lag_df = pd.read_csv(OUT_DIR / "macro_lead_lag_analysis.csv")
    proc = pd.read_csv(OUT_DIR / "procurement_plan.csv")
    exceptions = pd.read_csv(OUT_DIR / "planner_exceptions.csv")
    st_path = DATA_DIR / "inventory_stockouts.csv"
    if st_path.exists():
        stockouts = pd.read_csv(st_path)
    else:
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
        stockouts = hist.loc[hist["Is_Doc_Stockout"], st_cols].copy()
    with open(OUT_DIR / "data_quality_report.json", "r", encoding="utf-8") as f:
        dq_report = json.load(f)
    return (
        hist,
        fc,
        macro,
        sku_mst,
        bt_metrics,
        bt_preds,
        anomalies,
        lag_df,
        proc,
        exceptions,
        stockouts,
        dq_report,
    )


(
    hist_raw,
    fc_raw,
    macro_df,
    sku_master,
    bt_metrics,
    bt_preds,
    anomalies_raw,
    lag_df,
    proc_df,
    exceptions_df,
    stockouts_raw,
    dq_report,
) = load_artifacts()

# ------------------------------------------------------------------
# SHARED GLOBAL SIDEBAR FILTERS (Apply across BOTH History & Forecast)
# ------------------------------------------------------------------
st.sidebar.title("Celeris Control Tower")
st.sidebar.caption("Unified Filters (History 2019-01..2026-10 & Forecast 2026-11..2028-05)")

all_families = sorted(set(hist_raw["Product_Family"].dropna()) | set(fc_raw["Product_Family"].dropna()))
all_regions = sorted(set(hist_raw["Region"].dropna()) | set(fc_raw["Region"].dropna()))
all_channels = sorted(set(hist_raw["Sales_Channel"].dropna()) | set(fc_raw["Sales_Channel"].dropna()))
all_customers = sorted(set(hist_raw["Customer_Name"].dropna()) | set(fc_raw["Customer_Name"].dropna()))

sel_fam = st.sidebar.multiselect("Product Family", all_families, default=[])
sel_reg = st.sidebar.multiselect("Region", all_regions, default=[])
sel_chan = st.sidebar.multiselect("Sales Channel", all_channels, default=[])
sel_cust = st.sidebar.multiselect("Customer", all_customers, default=[])

h_skus = (
    hist_raw.loc[hist_raw["Product_Family"].isin(sel_fam), "SKU_ID"]
    if sel_fam
    else hist_raw["SKU_ID"]
)
f_skus = (
    fc_raw.loc[fc_raw["Product_Family"].isin(sel_fam), "SKU_ID"]
    if sel_fam
    else fc_raw["SKU_ID"]
)
sku_pool = sorted(set(h_skus.dropna()) | set(f_skus.dropna()))
sel_sku = st.sidebar.multiselect("SKU_ID (optional drill-down)", sku_pool, default=[])
use_unconstrained = st.sidebar.toggle("Use Unconstrained Demand in History", value=True)


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    out = df
    if sel_fam and "Product_Family" in out.columns:
        out = out[out["Product_Family"].isin(sel_fam)]
    if sel_reg and "Region" in out.columns:
        out = out[out["Region"].isin(sel_reg)]
    if sel_chan and "Sales_Channel" in out.columns:
        out = out[out["Sales_Channel"].isin(sel_chan)]
    if sel_cust and "Customer_Name" in out.columns:
        out = out[out["Customer_Name"].isin(sel_cust)]
    if sel_sku and "SKU_ID" in out.columns:
        out = out[out["SKU_ID"].isin(sel_sku)]
    return out


hist = apply_filters(hist_raw)
fc = apply_filters(fc_raw)
anomalies = apply_filters(anomalies_raw)
hist_val_col = "Unconstrained_Demand" if use_unconstrained else "Units_Sold"

st.sidebar.markdown("---")
view = st.sidebar.radio(
    "Navigation (12 Views)",
    [
        "1. Executive Overview (Both)",
        "2. Data Health (Training)",
        "3. Demand Explorer (Training)",
        "4. Anomaly Monitor (Training)",
        "5. Forecast & Accuracy (Prediction)",
        "6. Macro Lead/Lag Heatmap",
        "7. What-If Scenario Simulator",
        "8. Procurement Plan (P90)",
        "9. Lifecycle: EOL & Launches",
        "10. Stockouts & Lost Sales",
        "11. Planner Exception Queue",
        "12. Data Download Center",
    ],
)

# ------------------------------------------------------------------
# VIEW 1: EXECUTIVE OVERVIEW (Continuous Timeline: History -> Forecast)
# ------------------------------------------------------------------
if view.startswith("1."):
    st.title("Executive Overview — History into 19-Month Forecast")

    c1, c2, c3, c4 = st.columns(4)
    fc_u_total = int(fc["Units_Sold_Forecast"].sum()) if not fc.empty else 0
    fc_rev_total = float(fc["Forecast_Revenue_USD"].sum()) if not fc.empty else 0.0
    fc_p90_total = int(fc["P90"].sum()) if not fc.empty else 0
    ens_row = bt_metrics.loc[
        (bt_metrics["Slice_Type"] == "Overall") & (bt_metrics["Model"] == "Ensemble"), "WAPE"
    ]
    ens_wape = float(ens_row.values[0]) if len(ens_row) > 0 else 0.0

    c1.metric("19-Mo Forecast Units (P50)", f"{fc_u_total:,}")
    c2.metric("19-Mo Forecast Revenue", f"${fc_rev_total / 1e6:,.2f}M")
    c3.metric("19-Mo P90 Capacity Envelope", f"{fc_p90_total:,}")
    c4.metric("19-Mo Backtest WAPE (Ensemble)", f"{ens_wape * 100:.2f}%")

    if hist.empty and fc.empty:
        st.warning("No records match the current sidebar filter combination.")
    else:
        h_m = hist.groupby("Date", as_index=False).agg(
            Units=(hist_val_col, "sum"), Revenue=("Revenue_USD", "sum")
        )
        f_m = fc.groupby("Date", as_index=False).agg(
            Units=("Units_Sold_Forecast", "sum"),
            P10=("P10", "sum"),
            P90=("P90", "sum"),
            Revenue=("Forecast_Revenue_USD", "sum"),
        )

        metric_toggle = st.radio("Timeline Metric", ["Units", "Revenue (USD)"], horizontal=True)
        fig = go.Figure()
        if metric_toggle == "Units":
            fig.add_trace(
                go.Scatter(
                    x=h_m["Date"],
                    y=h_m["Units"],
                    name="History (Cleaned)",
                    line=dict(color="#1f77b4", width=2.5),
                )
            )
            fig.add_trace(go.Scatter(x=f_m["Date"], y=f_m["P90"], line=dict(width=0), showlegend=False))
            fig.add_trace(
                go.Scatter(
                    x=f_m["Date"],
                    y=f_m["P10"],
                    fill="tonexty",
                    fillcolor="rgba(255,127,14,0.20)",
                    line=dict(width=0),
                    name="P10–P90 Range",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=f_m["Date"],
                    y=f_m["Units"],
                    name="Forecast (P50)",
                    line=dict(color="#ff7f0e", width=2.5, dash="dash"),
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=h_m["Date"],
                    y=h_m["Revenue"],
                    name="Historical Revenue",
                    line=dict(color="#2ca02c", width=2.5),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=f_m["Date"],
                    y=f_m["Revenue"],
                    name="Forecast Revenue",
                    line=dict(color="#d62728", width=2.5, dash="dash"),
                )
            )

        fig.add_vline(x="2026-10", line_dash="dot", line_color="gray")
        fig.add_annotation(x="2026-11", y=1.02, yref="paper", text="Forecast Start →", showarrow=False)
        fig.update_layout(height=420, margin=dict(l=20, r=20, t=30, b=20), hovermode="x unified")
        st.plotly_chart(fig, width="stretch")

        if not fc.empty:
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                fam_mix = fc.groupby("Product_Family", as_index=False)["Units_Sold_Forecast"].sum()
                st.plotly_chart(
                    px.bar(
                        fam_mix,
                        x="Units_Sold_Forecast",
                        y="Product_Family",
                        orientation="h",
                        title="19-Mo Forecast Mix by Family",
                    ),
                    width="stretch",
                )
            with col_b:
                reg_mix = fc.groupby("Region", as_index=False)["Units_Sold_Forecast"].sum()
                st.plotly_chart(
                    px.pie(
                        reg_mix,
                        values="Units_Sold_Forecast",
                        names="Region",
                        title="19-Mo Forecast Mix by Region",
                        hole=0.45,
                    ),
                    width="stretch",
                )
            with col_c:
                ch_mix = fc.groupby("Sales_Channel", as_index=False)["Units_Sold_Forecast"].sum()
                st.plotly_chart(
                    px.pie(
                        ch_mix,
                        values="Units_Sold_Forecast",
                        names="Sales_Channel",
                        title="19-Mo Forecast Mix by Channel",
                        hole=0.45,
                    ),
                    width="stretch",
                )

# ------------------------------------------------------------------
# VIEW 2: DATA HEALTH (Training Data Audit)
# ------------------------------------------------------------------
elif view.startswith("2."):
    st.title("Data Health & Forensic Cleaning Audit")
    rows = []
    for fname, payload in dq_report.items():
        for k, v in payload.get("issues", {}).items():
            if isinstance(v, (int, float)):
                rows.append({"File": fname, "Issue_Type": k, "Rows_Or_Cells_Affected": int(v)})
    dq_df = pd.DataFrame(rows).sort_values("Rows_Or_Cells_Affected", ascending=False)

    c1, c2, c3 = st.columns(3)
    tr_rep = dq_report["train.csv"]
    c1.metric("train.csv Raw Rows", f"{tr_rep['rows_before']:,}")
    c2.metric(
        "train.csv Clean Unique Rows",
        f"{tr_rep['rows_after']:,}",
        delta=f"{tr_rep['rows_after'] - tr_rep['rows_before']:,} dups removed",
    )
    c3.metric("Total Issues Fixed", f"{dq_df['Rows_Or_Cells_Affected'].sum():,}")

    st.plotly_chart(
        px.bar(
            dq_df.head(15),
            x="Rows_Or_Cells_Affected",
            y="Issue_Type",
            color="File",
            orientation="h",
            title="Top 15 Data Quality Issues Repaired",
        ),
        width="stretch",
    )
    st.dataframe(dq_df, width="stretch")

# ------------------------------------------------------------------
# VIEW 3: DEMAND EXPLORER (Drill-Down, Seasonality, Zeros, Lifecycle)
# ------------------------------------------------------------------
elif view.startswith("3."):
    st.title("Demand Explorer — Drill-Down, Seasonality, Zeros & Lifecycle")
    if hist.empty:
        st.warning("No training history rows match the current filter selection.")
    else:
        h_ts = hist.groupby("Date", as_index=False).agg(
            Observed=("Units_Sold", "sum"), Unconstrained=("Unconstrained_Demand", "sum")
        )
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=h_ts["Date"], y=h_ts["Observed"], name="Shipped (Units_Sold)", line=dict(color="#636efa")
            )
        )
        fig.add_trace(
            go.Scatter(
                x=h_ts["Date"],
                y=h_ts["Unconstrained"],
                name="Unconstrained Demand",
                line=dict(color="#00cc96", dash="dot"),
            )
        )
        fig.update_layout(
            title="Observed Shipments vs Reconstructed Unconstrained Demand", height=360, hovermode="x unified"
        )
        st.plotly_chart(fig, width="stretch")

        c1, c2 = st.columns(2)
        with c1:
            seas = hist.groupby("Month", as_index=False)[hist_val_col].mean()
            st.plotly_chart(
                px.bar(seas, x="Month", y=hist_val_col, title="Average Monthly Seasonality Profile"),
                width="stretch",
            )
        with c2:
            zero_by_fam = (
                hist.assign(Is_Zero=hist[hist_val_col] == 0)
                .groupby("Product_Family", as_index=False)["Is_Zero"]
                .mean()
            )
            zero_by_fam["Zero_Month_Pct"] = (zero_by_fam["Is_Zero"] * 100).round(1)
            st.plotly_chart(
                px.bar(
                    zero_by_fam,
                    x="Zero_Month_Pct",
                    y="Product_Family",
                    orientation="h",
                    title="Zero-Order Month Rate (%) by Family",
                ),
                width="stretch",
            )

        # Lifecycle Mix & EV/ICE Shift + Family -> SKU -> Region -> Customer Drill-Down
        lc_cols = ["Lifecycle_Status", "Powertrain_Application", "Launch_Date", "EOL_Date"]
        h_base = hist.drop(columns=[c for c in lc_cols if c in hist.columns])
        h_lc = h_base.merge(sku_master[["SKU_ID"] + lc_cols], on="SKU_ID", how="left")

        c3, c4 = st.columns(2)
        with c3:
            lc_sum = h_lc.groupby("Lifecycle_Status", as_index=False)[hist_val_col].sum()
            st.plotly_chart(
                px.pie(
                    lc_sum,
                    values=hist_val_col,
                    names="Lifecycle_Status",
                    hole=0.45,
                    title="Historical Demand by Lifecycle Status",
                ),
                width="stretch",
            )
        with c4:
            pt_ts = h_lc.groupby(["Year", "Powertrain_Application"], as_index=False)[hist_val_col].sum()
            st.plotly_chart(
                px.bar(
                    pt_ts,
                    x="Year",
                    y=hist_val_col,
                    color="Powertrain_Application",
                    title="EV vs ICE vs Universal Demand Shift (Annual)",
                ),
                width="stretch",
            )

        st.subheader("Interactive Drill-Down: Family → SKU → Region → Customer")
        drill = (
            h_lc.groupby(
                [
                    "Product_Family",
                    "SKU_ID",
                    "Region",
                    "Customer_Name",
                    "Lifecycle_Status",
                    "Powertrain_Application",
                ],
                as_index=False,
            )
            .agg(
                Total_Units=(hist_val_col, "sum"),
                Avg_Monthly_Units=(hist_val_col, "mean"),
                Zero_Month_Pct=(hist_val_col, lambda s: round(float((s == 0).mean()) * 100, 1)),
                Months_Observed=("Date", "nunique"),
            )
            .sort_values("Total_Units", ascending=False)
        )
        drill["Avg_Monthly_Units"] = drill["Avg_Monthly_Units"].round(1)
        st.dataframe(drill.head(500), width="stretch")

# ------------------------------------------------------------------
# VIEW 4: ANOMALY MONITOR
# ------------------------------------------------------------------
elif view.startswith("4."):
    st.title("Anomaly Monitor — Real Events vs Data Errors")
    atypes = sorted(anomalies["Anomaly_Type"].dropna().unique())
    sel_atype = st.multiselect("Filter Anomaly Type", atypes, default=atypes)
    anom_f = anomalies[anomalies["Anomaly_Type"].isin(sel_atype)]

    if anom_f.empty:
        st.info("No flagged anomalies for the active filters.")
    else:
        c1, c2 = st.columns([1, 2])
        with c1:
            cnt = anom_f["Anomaly_Type"].value_counts().reset_index()
            cnt.columns = ["Anomaly_Type", "Count"]
            st.plotly_chart(
                px.bar(cnt, x="Count", y="Anomaly_Type", orientation="h", title="Anomalies by Category"),
                width="stretch",
            )
        with c2:
            ts = hist.groupby("Date", as_index=False)[hist_val_col].sum()
            anom_dates = anom_f.groupby("Date", as_index=False).size().merge(ts, on="Date", how="inner")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ts["Date"], y=ts[hist_val_col], name="Demand Timeline"))
            fig.add_trace(
                go.Scatter(
                    x=anom_dates["Date"],
                    y=anom_dates[hist_val_col],
                    mode="markers",
                    marker=dict(size=9, color="red"),
                    name="Months with Flagged Anomalies",
                )
            )
            fig.update_layout(title="Flagged Anomalies Overlaid on Demand History", height=380)
            st.plotly_chart(fig, width="stretch")

        st.dataframe(anom_f.head(500), width="stretch")

# ------------------------------------------------------------------
# VIEW 5: FORECAST & ACCURACY (Prediction + Backtest Benchmarks)
# ------------------------------------------------------------------
elif view.startswith("5."):
    st.title("Forecast & 19-Month Rolling Backtest Accuracy")

    if not hist.empty or not fc.empty:
        h_m = hist.groupby("Date", as_index=False)[hist_val_col].sum().rename(columns={hist_val_col: "Units"})
        f_m = fc.groupby("Date", as_index=False).agg(
            Units=("Units_Sold_Forecast", "sum"), P10=("P10", "sum"), P90=("P90", "sum")
        )
        fig_fc = go.Figure()
        fig_fc.add_trace(
            go.Scatter(x=h_m["Date"], y=h_m["Units"], name="History (Unconstrained)", line=dict(color="#1f77b4", width=2))
        )
        fig_fc.add_trace(go.Scatter(x=f_m["Date"], y=f_m["P90"], line=dict(width=0), showlegend=False))
        fig_fc.add_trace(
            go.Scatter(
                x=f_m["Date"],
                y=f_m["P10"],
                fill="tonexty",
                fillcolor="rgba(255,127,14,0.20)",
                line=dict(width=0),
                name="P10–P90 Range",
            )
        )
        fig_fc.add_trace(
            go.Scatter(
                x=f_m["Date"],
                y=f_m["Units"],
                name="Forecast (P50)",
                line=dict(color="#ff7f0e", width=2.5, dash="dash"),
            )
        )
        fig_fc.add_vline(x="2026-10", line_dash="dot", line_color="gray")
        fig_fc.update_layout(title="Forecast vs History (Filtered Scope)", height=350, hovermode="x unified")
        st.plotly_chart(fig_fc, width="stretch")

    ov = bt_metrics[bt_metrics["Slice_Type"].isin(["Overall", "Calibration"])].sort_values(
        ["Slice_Type", "WAPE"]
    )
    st.subheader("Model Benchmark Leaderboard (3-Origin 19-Month Backtest)")
    st.dataframe(ov, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        fam_m = bt_metrics[
            (bt_metrics["Slice_Type"] == "Product_Family")
            & (bt_metrics["Model"].isin(["Seasonal_Naive", "SBA_ETS", "Ensemble"]))
        ]
        st.plotly_chart(
            px.bar(
                fam_m,
                x="Slice_Value",
                y="WAPE",
                color="Model",
                barmode="group",
                title="Backtest WAPE by Product Family",
                labels={"Slice_Value": "Product Family"},
            ),
            width="stretch",
        )
    with c2:
        fam_bias = bt_metrics[
            (bt_metrics["Slice_Type"] == "Product_Family") & (bt_metrics["Model"] == "Ensemble")
        ]
        st.plotly_chart(
            px.bar(
                fam_bias,
                x="Slice_Value",
                y="Bias",
                title="Ensemble Backtest Bias by Product Family",
                labels={"Slice_Value": "Product Family"},
            ),
            width="stretch",
        )

    c3, c4 = st.columns(2)
    with c3:
        hor_m = bt_metrics[
            (bt_metrics["Slice_Type"] == "Horizon")
            & (bt_metrics["Model"].isin(["Seasonal_Naive", "SBA_ETS", "Ensemble"]))
        ]
        st.plotly_chart(
            px.line(
                hor_m,
                x="Slice_Value",
                y="WAPE",
                color="Model",
                markers=True,
                title="WAPE Across Forecast Horizon (H01–H19)",
                labels={"Slice_Value": "Horizon"},
            ),
            width="stretch",
        )
    with c4:
        hor_bias = bt_metrics[
            (bt_metrics["Slice_Type"] == "Horizon") & (bt_metrics["Model"] == "Ensemble")
        ]
        st.plotly_chart(
            px.line(
                hor_bias,
                x="Slice_Value",
                y="Bias",
                markers=True,
                title="Ensemble Bias Across Forecast Horizon (H01–H19)",
                labels={"Slice_Value": "Horizon"},
            ),
            width="stretch",
        )

    reg_m = bt_metrics[(bt_metrics["Slice_Type"] == "Region") & (bt_metrics["Model"] == "Ensemble")]
    st.plotly_chart(
        px.bar(
            reg_m,
            x="Slice_Value",
            y=["WAPE", "Bias"],
            barmode="group",
            title="Ensemble WAPE & Bias by Region",
            labels={"Slice_Value": "Region"},
        ),
        width="stretch",
    )

# ------------------------------------------------------------------
# VIEW 6 (OPTIONAL): MACRO LEAD/LAG HEATMAP & STABILITY
# ------------------------------------------------------------------
elif view.startswith("6."):
    st.title("Macroeconomic Lead/Lag Discovery & Regime Stability")
    best_idx = lag_df.groupby(["Product_Family", "Indicator"])["Corr_Full"].apply(lambda s: s.abs().idxmax())
    best = lag_df.loc[best_idx]
    piv_corr = best.pivot(index="Product_Family", columns="Indicator", values="Corr_Full")
    piv_lag = best.pivot(index="Product_Family", columns="Indicator", values="Lag_Months")

    st.plotly_chart(
        px.imshow(
            piv_corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Peak YoY Cross-Correlation by Family x Macro Indicator",
        ),
        width="stretch",
    )
    st.plotly_chart(
        px.imshow(
            piv_lag,
            text_auto=True,
            color_continuous_scale="Viridis",
            title="Optimal Lead/Lag (Months) by Family x Macro Indicator",
        ),
        width="stretch",
    )
    st.subheader("Regime Stability Audit (2019–2022 vs 2023–2026)")
    st.dataframe(
        best[
            [
                "Product_Family",
                "Indicator",
                "Lag_Months",
                "Corr_Full",
                "Corr_Early_2019_2022",
                "Corr_Late_2023_2026",
                "Is_Stable",
            ]
        ],
        width="stretch",
    )

# ------------------------------------------------------------------
# VIEW 7 (OPTIONAL): WHAT-IF SCENARIO SIMULATOR
# ------------------------------------------------------------------
elif view.startswith("7."):
    st.title("What-If Macro Scenario Simulator")
    st.caption("Adjust macroeconomic assumptions to stress-test 19-month unit & revenue plans.")
    c1, c2, c3 = st.columns(3)
    d_pmi = c1.slider("ISM Manufacturing PMI Shock (pts)", -8.0, 8.0, 0.0, 0.5)
    d_ev = c2.slider("EV Adoption Index Shock (%)", -20.0, 25.0, 0.0, 1.0)
    d_auto = c3.slider("Auto Production YoY Shock (%)", -10.0, 10.0, 0.0, 0.5)

    if fc.empty:
        st.warning("No forecast rows match the current filter selection.")
    else:
        sim = fc.copy()
        pmi_eff = 1.0 + 0.012 * d_pmi
        auto_eff = 1.0 + 0.010 * d_auto
        ev_mult = np.where(
            sim["Powertrain_Application"] == "EV",
            1.0 + 0.015 * d_ev,
            np.where(sim["Powertrain_Application"] == "ICE", 1.0 - 0.008 * d_ev, 1.0 + 0.002 * d_ev),
        )
        sim["Scenario_Units"] = (
            sim["Units_Sold_Forecast"] * pmi_eff * auto_eff * ev_mult
        ).clip(lower=0).round()
        sim["Scenario_Rev"] = sim["Scenario_Units"] * sim["Projected_Price_USD"]

        base_u, scen_u = int(sim["Units_Sold_Forecast"].sum()), int(sim["Scenario_Units"].sum())
        base_r, scen_r = float(sim["Forecast_Revenue_USD"].sum()), float(sim["Scenario_Rev"].sum())

        k1, k2 = st.columns(2)
        k1.metric(
            "Scenario 19-Mo Units",
            f"{scen_u:,}",
            delta=f"{scen_u - base_u:+,} ({(scen_u - base_u) / max(base_u, 1):+.1%})",
        )
        k2.metric(
            "Scenario 19-Mo Revenue",
            f"${scen_r / 1e6:,.2f}M",
            delta=f"${(scen_r - base_r) / 1e6:+,.2f}M",
        )

        s_m = sim.groupby("Date", as_index=False).agg(
            Baseline=("Units_Sold_Forecast", "sum"), Scenario=("Scenario_Units", "sum")
        )
        st.plotly_chart(
            px.line(
                s_m, x="Date", y=["Baseline", "Scenario"], markers=True, title="Baseline vs What-If Monthly Units"
            ),
            width="stretch",
        )

# ------------------------------------------------------------------
# VIEW 8 (OPTIONAL): PROCUREMENT PLAN FROM P90 FORECASTS
# ------------------------------------------------------------------
elif view.startswith("8."):
    st.title("Procurement & Safety-Buffer Plan (P90 Envelope)")
    p_filt = apply_filters(proc_df)
    if p_filt.empty:
        st.warning("No procurement records match the current filter selection.")
    else:
        p_m = p_filt.groupby("Date", as_index=False)[
            ["Forecast_P50_Units", "Safety_Buffer_Units", "Procurement_P90_Budget_USD"]
        ].sum()
        st.plotly_chart(
            px.bar(
                p_m,
                x="Date",
                y=["Forecast_P50_Units", "Safety_Buffer_Units"],
                title="Monthly Cycle Stock (P50) + Safety Buffer (P90 - P50)",
            ),
            width="stretch",
        )
        st.dataframe(p_filt, width="stretch")

# ------------------------------------------------------------------
# VIEW 9 (OPTIONAL): LIFECYCLE — EOL & PLANNED LAUNCHES
# ------------------------------------------------------------------
elif view.startswith("9."):
    st.title("Product Lifecycle — EOL Phase-Outs & Replaces_SKU Launches")
    pairs = sku_master[sku_master["Lifecycle_Status"] == "Planned Launch"][
        ["SKU_ID", "Replaces_SKU", "Product_Family", "Launch_Date"]
    ]
    st.dataframe(pairs, width="stretch")

    sel_pair = st.selectbox(
        "Inspect Replacement Pair",
        [f"{r.Replaces_SKU} → {r.SKU_ID} ({r.Product_Family})" for r in pairs.itertuples(index=False)],
    )
    old_sku, new_sku = sel_pair.split(" → ")[0], sel_pair.split(" → ")[1].split(" ")[0]

    h_old = (
        hist_raw[hist_raw["SKU_ID"] == old_sku]
        .groupby("Date", as_index=False)[hist_val_col]
        .sum()
        .rename(columns={hist_val_col: "Units"})
    )
    h_old["Series"] = f"{old_sku} (History)"
    f_old = (
        fc_raw[fc_raw["SKU_ID"] == old_sku]
        .groupby("Date", as_index=False)["Units_Sold_Forecast"]
        .sum()
        .rename(columns={"Units_Sold_Forecast": "Units"})
    )
    f_old["Series"] = f"{old_sku} (EOL Forecast)"
    f_new = (
        fc_raw[fc_raw["SKU_ID"] == new_sku]
        .groupby("Date", as_index=False)["Units_Sold_Forecast"]
        .sum()
        .rename(columns={"Units_Sold_Forecast": "Units"})
    )
    f_new["Series"] = f"{new_sku} (Launch Forecast)"

    comb = pd.concat([h_old, f_old, f_new], ignore_index=True)
    fig = px.line(comb, x="Date", y="Units", color="Series", title=f"Seamless Demand Handover: {old_sku} → {new_sku}")
    fig.add_vline(x="2026-10", line_dash="dot", line_color="gray")
    st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------
# VIEW 10 (OPTIONAL): STOCKOUTS & LOST SALES
# ------------------------------------------------------------------
elif view.startswith("10."):
    st.title("Stockouts, Backorder Echoes & Lost Sales Impact")
    st_f = stockouts_raw.merge(sku_master[["SKU_ID", "Product_Family"]], on="SKU_ID", how="left")
    st_f = apply_filters(st_f)

    if st_f.empty:
        st.info("No documented stockout records for the active filters.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Documented Stockout Rows", f"{len(st_f):,}")
        c2.metric("Total Lost Units", f"{int(st_f['Lost_Units'].sum()):,}")
        c3.metric("Avg Fill Rate", f"{st_f['Fill_Rate_Pct'].mean():.1f}%")

        st_m = st_f.groupby("Date", as_index=False)[
            ["Requested_Units", "Shipped_Units", "Lost_Units", "Backordered_Units"]
        ].sum()
        st.plotly_chart(
            px.area(
                st_m,
                x="Date",
                y=["Shipped_Units", "Backordered_Units", "Lost_Units"],
                title="Monthly Requested Demand Breakdown during Stockouts",
            ),
            width="stretch",
        )

# ------------------------------------------------------------------
# VIEW 11 (OPTIONAL): PLANNER EXCEPTION QUEUE
# ------------------------------------------------------------------
elif view.startswith("11."):
    st.title("Planner Exception Queue (Priority Review)")
    exc_f = apply_filters(exceptions_df)
    st.metric("Actionable Exceptions", f"{len(exc_f):,}")
    st.dataframe(exc_f, width="stretch")

# ------------------------------------------------------------------
# VIEW 12 (OPTIONAL): DATA DOWNLOAD CENTER
# ------------------------------------------------------------------
elif view.startswith("12."):
    st.title("Data Download Center")
    st.download_button(
        "Download Filtered Forecast (CSV)", fc.to_csv(index=False), "celeris_filtered_forecast.csv", "text/csv"
    )
    st.download_button(
        "Download Official Submission (CSV)",
        pd.read_csv(OUT_DIR / "submission.csv").to_csv(index=False),
        "submission.csv",
        "text/csv",
    )
    st.download_button(
        "Download Anomaly Register (CSV)", anomalies.to_csv(index=False), "anomaly_register.csv", "text/csv"
    )
    st.download_button(
        "Download Procurement Plan (CSV)", proc_df.to_csv(index=False), "procurement_plan.csv", "text/csv"
    )
    st.download_button(
        "Download Backtest Metrics (CSV)", bt_metrics.to_csv(index=False), "backtest_metrics.csv", "text/csv"
    )