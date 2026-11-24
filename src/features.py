import calendar
import re
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.config import DATA_DIR, MACRO_COLS, OUT_DIR


def _ym_to_int(ym_series: pd.Series) -> pd.Series:
    return ym_series.str[:4].astype(int) * 12 + ym_series.str[5:7].astype(int)


def _int_to_ym(m_int: int) -> str:
    y, m = divmod(m_int - 1, 12)
    return f"{y:04d}-{m + 1:02d}"


def analyze_macro_lead_lag(
    train_df: pd.DataFrame, macro_df: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[Tuple[str, str], int]]:
    """
    Workstream 3 (Advanced + Expert):
      - Computes YoY-differenced cross-correlations across lags 0..6 for every
        (Product_Family, Macro_Indicator) pair.
      - Tests lag stability across Early regime (2019-2022) vs Late regime (2023-2026).
      - Returns full analysis DataFrame + optimal lag map per (Family, Indicator).
    """
    fam_m = (
        train_df.groupby(["Product_Family", "Date"])["Unconstrained_Demand"]
        .sum()
        .reset_index()
        .sort_values(["Product_Family", "Date"])
    )
    fam_m["Demand_YoY"] = fam_m.groupby("Product_Family")["Unconstrained_Demand"].pct_change(12)

    mac = macro_df.sort_values("Date").copy()
    for c in MACRO_COLS:
        if c == "Auto_Production_YoY_Pct":
            mac[f"{c}_diff"] = mac[c]
        else:
            mac[f"{c}_diff"] = mac[c].diff(12)

    records = []
    best_lag_map: Dict[Tuple[str, str], int] = {}

    for fam, g in fam_m.groupby("Product_Family"):
        merged = g.merge(mac, on="Date", how="left").sort_values("Date").reset_index(drop=True)
        for ind in MACRO_COLS:
            col = f"{ind}_diff"
            best_lag, best_corr = 0, 0.0
            for lag in range(0, 7):
                x = merged[col].shift(lag)
                y = merged["Demand_YoY"]
                valid = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
                corr_all = float(x[valid].corr(y[valid])) if valid.sum() >= 18 else 0.0

                early = valid & (merged["Date"] <= "2022-12")
                late = valid & (merged["Date"] >= "2023-01")
                corr_early = float(x[early].corr(y[early])) if early.sum() >= 10 else np.nan
                corr_late = float(x[late].corr(y[late])) if late.sum() >= 10 else np.nan

                stable = (
                    pd.notna(corr_early)
                    and pd.notna(corr_late)
                    and (np.sign(corr_early) == np.sign(corr_late))
                    and (abs(corr_early - corr_late) < 0.30)
                )

                records.append(
                    {
                        "Product_Family": fam,
                        "Indicator": ind,
                        "Lag_Months": lag,
                        "Corr_Full": round(corr_all, 4),
                        "Corr_Early_2019_2022": round(corr_early, 4) if pd.notna(corr_early) else np.nan,
                        "Corr_Late_2023_2026": round(corr_late, 4) if pd.notna(corr_late) else np.nan,
                        "Regime_Shift_Delta": (
                            round(corr_late - corr_early, 4)
                            if (pd.notna(corr_early) and pd.notna(corr_late))
                            else np.nan
                        ),
                        "Is_Stable": bool(stable),
                    }
                )
                if abs(corr_all) > abs(best_corr):
                    best_corr = corr_all
                    best_lag = lag
            best_lag_map[(fam, ind)] = best_lag

    res = pd.DataFrame(records)
    res.to_csv(OUT_DIR / "macro_lead_lag_analysis.csv", index=False)
    return res, best_lag_map


def build_event_feature_table(
    dates: list[str],
    regions: list[str],
    families: list[str],
    customers: list[str],
    channels: list[str],
) -> dict:
    """
    Parses events_calendar.csv into clean lookup tables covering both history & forecast window:
      - Fractional-month overlap for shifting holidays (Lunar New Year in APAC, Ramadan in MEA)
      - Summer shutdown flags (NA July, Europe August)
      - Annual list-price hike magnitude & pre-hike announcement window (Nov/Dec)
      - Customer-Region specific disruptions (EV-038 Nexon strike, EV-052 Helios expansion,
        EV-062 Orion Driveline NA retooling shutdown 2027-07..2027-08)
    """
    ev = pd.read_csv(DATA_DIR / "events_calendar.csv")

    def _month_overlap(start_str: str, end_str: str, ym: str) -> float:
        y, m = int(ym[:4]), int(ym[5:7])
        ndays = calendar.monthrange(y, m)[1]
        m_start = pd.Timestamp(year=y, month=m, day=1)
        m_end = pd.Timestamp(year=y, month=m, day=ndays)

        s_raw = str(start_str).strip()
        e_raw = str(end_str).strip() if pd.notna(end_str) else "2029-12-31"

        s_dt = pd.Timestamp(s_raw + "-01") if len(s_raw) == 7 else pd.Timestamp(s_raw)
        if len(e_raw) == 7:
            ey, em = int(e_raw[:4]), int(e_raw[5:7])
            e_dt = pd.Timestamp(year=ey, month=em, day=calendar.monthrange(ey, em)[1])
        else:
            e_dt = pd.Timestamp(e_raw)

        ov_start = max(m_start, s_dt)
        ov_end = min(m_end, e_dt)
        if ov_end < ov_start:
            return 0.0
        return float((ov_end - ov_start).days + 1) / float(ndays)

    price_hike_by_ym = {d: 0.0 for d in dates}
    pre_hike_signal_by_ym = {d: 0.0 for d in dates}
    surcharge_by_ym = {d: 0.0 for d in dates}

    for r in ev[ev["Event_Type"] == "Price Change"].itertuples(index=False):
        m_pct = re.search(r"(\d+(?:\.\d+)?)%", str(r.Description))
        pct = float(m_pct.group(1)) if m_pct else 0.0
        if "surcharge" in str(r.Description).lower():
            for d in dates:
                if _month_overlap(r.Start_Date, r.End_Date, d) > 0:
                    surcharge_by_ym[d] = pct
        else:
            start_ym = str(r.Start_Date)[:7]
            if start_ym in price_hike_by_ym:
                price_hike_by_ym[start_ym] = pct
            s_int = int(start_ym[:4]) * 12 + int(start_ym[5:7])
            for offset in (1, 2):
                prev_ym = _int_to_ym(s_int - offset)
                if prev_ym in pre_hike_signal_by_ym:
                    pre_hike_signal_by_ym[prev_ym] = max(pre_hike_signal_by_ym[prev_ym], pct)

    reg_events = {}
    for r in ev[ev["Scope_Type"] == "Region"].itertuples(index=False):
        reg = r.Scope_Value
        desc = str(r.Description).lower()
        etype = str(r.Event_Type)
        for d in dates:
            ov = _month_overlap(r.Start_Date, r.End_Date, d)
            if ov <= 0:
                continue
            slot = reg_events.setdefault(
                (d, reg),
                {"lny_frac": 0.0, "ramadan_frac": 0.0, "summer_shutdown": 0.0, "covid_disruption": 0.0},
            )
            if "lunar new year" in desc:
                slot["lny_frac"] = max(slot["lny_frac"], ov)
            elif "ramadan" in desc:
                slot["ramadan_frac"] = max(slot["ramadan_frac"], ov)
            elif "retooling" in desc or "summer plant" in desc:
                slot["summer_shutdown"] = max(slot["summer_shutdown"], ov)
            elif etype == "Disruption":
                slot["covid_disruption"] = max(slot["covid_disruption"], ov)

    cust_reg_events = {}
    for r in ev[ev["Scope_Type"] == "Customer"].itertuples(index=False):
        parts = [p.strip() for p in str(r.Scope_Value).split("|")]
        if len(parts) != 2:
            continue
        cust, reg = parts
        desc = str(r.Description).lower()
        for d in dates:
            ov = _month_overlap(r.Start_Date, r.End_Date, d)
            if ov <= 0:
                continue
            slot = cust_reg_events.setdefault(
                (d, cust, reg), {"cust_shutdown": 0.0, "cust_capacity_up": 0.0}
            )
            if "strike" in desc or "shutdown" in desc:
                slot["cust_shutdown"] = 1.0
            elif "capacity" in desc or "second" in desc:
                slot["cust_capacity_up"] = 1.0

    chan_events = {}
    for r in ev[ev["Scope_Type"] == "Channel"].itertuples(index=False):
        ch = r.Scope_Value
        for d in dates:
            if _month_overlap(r.Start_Date, r.End_Date, d) > 0:
                chan_events[(d, ch)] = 1.0

    return {
        "price_hike": price_hike_by_ym,
        "pre_hike": pre_hike_signal_by_ym,
        "surcharge": surcharge_by_ym,
        "reg_events": reg_events,
        "cust_reg_events": cust_reg_events,
        "chan_events": chan_events,
    }


def compute_pooled_seasonality(hist_df: pd.DataFrame) -> dict:
    """
    Computes low-variance pooled seasonality strictly from `hist_df <= cutoff`
    (recent 48 months excluding COVID lockdown window 2020-02..2020-06):
      - reg_seas[(Region, Month)]
      - fam_seas[(Product_Family, Month)]
      - fam_reg_seas[(Product_Family, Region, Month)]
    """
    valid = hist_df[
        hist_df["Unconstrained_Demand"].notna() & (~hist_df["Date"].between("2020-02", "2020-06"))
    ].copy()
    if valid.empty:
        return {"reg": {}, "fam": {}, "fam_reg": {}}

    ym_int = _ym_to_int(valid["Date"])
    cutoff_int = int(ym_int.max())
    sub = valid[ym_int >= (cutoff_int - 47)].copy()
    if sub.empty:
        sub = valid

    reg_mean = sub.groupby("Region")["Unconstrained_Demand"].mean()
    reg_m = (sub.groupby(["Region", "Month"])["Unconstrained_Demand"].mean() / reg_mean).clip(0.3, 1.8)

    fam_mean = sub.groupby("Product_Family")["Unconstrained_Demand"].mean()
    fam_m = (
        sub.groupby(["Product_Family", "Month"])["Unconstrained_Demand"].mean() / fam_mean
    ).clip(0.3, 1.8)

    fr_mean = sub.groupby(["Product_Family", "Region"])["Unconstrained_Demand"].mean()
    fr_m = (
        sub.groupby(["Product_Family", "Region", "Month"])["Unconstrained_Demand"].mean() / fr_mean
    ).clip(0.3, 1.8)

    return {
        "reg": {k: float(v) for k, v in reg_m.dropna().items()},
        "fam": {k: float(v) for k, v in fam_m.dropna().items()},
        "fam_reg": {k: float(v) for k, v in fr_m.dropna().items()},
    }


def compute_series_profile(hist_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes leak-free series profile per (SKU_ID, Region) strictly using data <= cutoff:
      - Syntetos-Boylan ADI & CV^2 -> Intermittency_Class
      - Dominant ordering period P, regularity, & last order month
      - Robust 24m/12m medians (Med_24m, Med_12m) + EWM + exact origin lags (Last_1m..3m)
      - 18-month log-trend slope & strongly shrunk per-series seasonal index + exact monthly zero rates
    """
    rows = []
    for (sku, reg), g in hist_df.groupby(["SKU_ID", "Region"], sort=False):
        g = g.sort_values("Date")
        y = g["Unconstrained_Demand"].values.astype(float)
        ym_int = _ym_to_int(g["Date"]).values
        valid = ~np.isnan(y)
        yv = y[valid]
        tmv = ym_int[valid]

        if len(yv) == 0:
            continue

        nz_mask = yv > 0
        nz_vals = yv[nz_mask]
        n_obs = len(yv)
        n_nz = len(nz_vals)
        zero_rate = float(1.0 - n_nz / max(n_obs, 1))

        # Recent 24 months for current ordering cadence
        recent_mask = tmv >= (tmv[-1] - 23)
        y_rec = yv[recent_mask]
        t_rec = tmv[recent_mask]
        nz_rec_t = t_rec[y_rec > 0]

        if len(nz_rec_t) >= 2:
            gaps = np.diff(nz_rec_t)
            adi = float(np.mean(gaps))
            est_period = int(np.clip(round(np.median(gaps)), 1, 6))
        elif n_nz >= 2:
            gaps = np.diff(tmv[nz_mask])
            adi = float(np.mean(gaps))
            est_period = int(np.clip(round(np.median(gaps)), 1, 6))
        else:
            adi = float(n_obs)
            est_period = 12

        if n_nz >= 2 and nz_vals.mean() > 0:
            cv2 = float((nz_vals.std(ddof=0) / nz_vals.mean()) ** 2)
        else:
            cv2 = 0.0

        if adi < 1.32 and cv2 < 0.49:
            iclass = "Smooth"
        elif adi < 1.32 and cv2 >= 0.49:
            iclass = "Erratic"
        elif adi >= 1.32 and cv2 < 0.49:
            iclass = "Intermittent"
        else:
            iclass = "Lumpy"

        last_order_ym = int(tmv[nz_mask][-1]) if n_nz > 0 else int(tmv[0] - 24)
        cutoff_ym = int(tmv[-1])
        trailing_zeros = int(cutoff_ym - last_order_ym)

        # Rolling means, robust medians & EWM level at cutoff
        mean_6 = float(np.mean(yv[-6:]))
        mean_12 = float(np.mean(yv[-12:]))
        mean_24 = float(np.mean(yv[-24:]))
        med_12 = float(np.median(yv[-12:]))
        med_24 = float(np.median(yv[-24:]))

        ewm = float(yv[0])
        for val in yv[1:]:
            ewm = 0.22 * val + 0.78 * ewm

        # Non-zero order size stats (critical for clockwork bi-monthly/quarterly on-cycle scale)
        y_6_pos = yv[-6:][yv[-6:] > 0]
        y_12_pos = yv[-12:][yv[-12:] > 0]
        y_24_pos = yv[-24:][yv[-24:] > 0]
        fallback_nz = float(nz_vals.mean()) if n_nz > 0 else 0.0
        nz_mean_12 = float(np.mean(y_12_pos)) if len(y_12_pos) > 0 else fallback_nz
        nz_mean_6 = float(np.mean(y_6_pos)) if len(y_6_pos) > 0 else nz_mean_12
        nz_med_24 = float(np.median(y_24_pos)) if len(y_24_pos) > 0 else nz_mean_12
        nz_last_size = float(nz_vals[-1]) if n_nz > 0 else 0.0

        if n_nz > 0:
            nz_ewm = float(nz_vals[0])
            for val in nz_vals[1:]:
                nz_ewm = 0.25 * val + 0.75 * nz_ewm
        else:
            nz_ewm = 0.0

        # Exact origin lags
        last_1m = float(yv[-1])
        last_2m = float(yv[-2]) if len(yv) >= 2 else last_1m
        last_3m = float(yv[-3]) if len(yv) >= 3 else last_2m

        std_12 = float(np.std(yv[-12:])) if len(yv) >= 2 else 0.0
        prev_12 = float(np.mean(yv[-24:-12])) if len(yv) >= 24 else mean_12
        yoy_ratio = float(np.clip((mean_12 + 1.0) / (prev_12 + 1.0), 0.2, 5.0))

        # 18-month log-trend slope (on positive months so cycles don't bias slope)
        slope_mask = (tmv >= cutoff_ym - 17) & (yv > 0)
        if slope_mask.sum() >= 5:
            tx = (tmv[slope_mask] - cutoff_ym).astype(float)
            ly = np.log1p(yv[slope_mask])
            tx_c = tx - tx.mean()
            denom = float(np.dot(tx_c, tx_c))
            log_slope_18m = float(
                np.clip(np.dot(tx_c, ly - ly.mean()) / max(denom, 1e-6), -0.06, 0.06)
            )
        else:
            log_slope_18m = 0.0

        # Cycle regularity score over recent 24m
        if len(nz_rec_t) >= 3:
            reg_score = float(np.mean(np.diff(nz_rec_t) == est_period))
        else:
            reg_score = 0.0

        # Strongly shrunk per-series seasonality (prior=4.0 prevents overfitting small monthly samples)
        # + exact calendar-month zero rate & observation count (for deterministic holiday shutdowns)
        months_v = ((tmv - 1) % 12) + 1
        m_stats = {}
        overall_m = max(mean_24, 1e-3)
        overall_nz = max(nz_med_24, 1e-3)
        is_clockwork = (est_period >= 2) and (reg_score >= 0.65) and (len(nz_rec_t) >= 3)

        for m in range(1, 13):
            vals_m = yv[(months_v == m) & (tmv >= cutoff_ym - 47)]
            cnt_m = len(vals_m)
            m_stats[f"cnt_m{m}"] = cnt_m
            if cnt_m > 0:
                m_stats[f"zrate_m{m}"] = float(np.mean(vals_m == 0))
                if is_clockwork:
                    pos_m = vals_m[vals_m > 0]
                    raw_ratio = (float(np.mean(pos_m)) / overall_nz) if len(pos_m) > 0 else 1.0
                    n_eff = len(pos_m)
                    m_stats[f"seas_m{m}"] = float(
                        np.clip((n_eff * raw_ratio + 4.0) / (n_eff + 4.0), 0.6, 1.6)
                    )
                else:
                    raw_ratio = float(np.mean(vals_m)) / overall_m
                    m_stats[f"seas_m{m}"] = float(
                        np.clip((cnt_m * raw_ratio + 4.0) / (cnt_m + 4.0), 0.3, 1.8)
                    )
            else:
                m_stats[f"zrate_m{m}"] = 0.0
                m_stats[f"seas_m{m}"] = 1.0

        rec = {
            "SKU_ID": sku,
            "Region": reg,
            "ADI": adi,
            "CV2": cv2,
            "Intermittency_Class": iclass,
            "Zero_Rate": zero_rate,
            "Est_Period": est_period,
            "Cycle_Regularity": reg_score,
            "Last_Order_YM": last_order_ym,
            "Trailing_Zero_Months": trailing_zeros,
            "Mean_6m": mean_6,
            "Mean_12m": mean_12,
            "Mean_24m": mean_24,
            "Med_12m": med_12,
            "Med_24m": med_24,
            "EWM_Level": ewm,
            "NZ_EWM_Level": nz_ewm,
            "NZ_Mean_6m": nz_mean_6,
            "NZ_Mean_12m": nz_mean_12,
            "NZ_Med_24m": nz_med_24,
            "NZ_Last_Order_Size": nz_last_size,
            "Last_1m": last_1m,
            "Last_2m": last_2m,
            "Last_3m": last_3m,
            "Log_Slope_18m": log_slope_18m,
            "Std_12m": std_12,
            "YoY_Ratio_12m": yoy_ratio,
            "History_Months": n_obs,
        }
        rec.update(m_stats)
        rows.append(rec)

    return pd.DataFrame(rows)


def attach_macro_and_event_features(
    panel: pd.DataFrame,
    macro_df: pd.DataFrame,
    sku_master: pd.DataFrame,
    event_tables: dict,
    best_lag_map: Dict[Tuple[str, str], int],
    origin_ym_int: int,
    pooled_seas: dict | None = None,
) -> pd.DataFrame:
    """
    Builds leak-free features for a target panel anchored at forecast origin `origin_ym_int`:
      - All 8 Lag-1 macro indicators & 3-month macro momentum
      - Pooled (Region, Month) and (Family, Region, Month) seasonal indexes
      - Denoised Cycle-Aware Baseline Anchor (Med_24m + Mean_24m + EWM * pooled seasonality)
    """
    df = panel.copy()
    df["Target_YM_Int"] = _ym_to_int(df["Date"])
    df["Horizon"] = (df["Target_YM_Int"] - origin_ym_int).astype(int)

    # 1. Calendar & cyclical time features
    df["Month"] = df["Date"].str[5:7].astype(int)
    df["Year"] = df["Date"].str[:4].astype(int)
    df["Quarter"] = ((df["Month"] - 1) // 3) + 1
    df["Month_Sin"] = np.sin(2.0 * np.pi * df["Month"] / 12.0)
    df["Month_Cos"] = np.cos(2.0 * np.pi * df["Month"] / 12.0)
    df["Time_Idx"] = df["Target_YM_Int"] - (2019 * 12 + 1)

    # 2. SKU master metadata
    mst_cols = [
        "SKU_ID",
        "Size_Class",
        "Powertrain_Application",
        "Launch_Date",
        "EOL_Date",
        "Lifecycle_Status",
    ]
    for c in mst_cols[1:]:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)
    df = df.merge(sku_master[mst_cols], on="SKU_ID", how="left")

    df["Size_Numeric"] = (
        df["Size_Class"].astype(str).str.extract(r"(\d+)").astype(float).fillna(0.0)
    )
    launch_int = _ym_to_int(df["Launch_Date"].fillna("2015-01"))
    df["SKU_Age_Months"] = (df["Target_YM_Int"] - launch_int).clip(lower=0)
    eol_int = df["EOL_Date"].apply(
        lambda x: int(x[:4]) * 12 + int(x[5:7]) if pd.notna(x) else 999999
    )
    df["Months_To_EOL"] = (eol_int - df["Target_YM_Int"]).clip(-12, 60)

    # 3. Macro indicators: contemporary + lag 1 + lag 2 + lag 3 + 3m momentum + YoY
    mac = macro_df.sort_values("Date").copy()
    mac_feats = mac[["Date"]].copy()
    for c in MACRO_COLS:
        mac_feats[c] = mac[c]
        mac_feats[f"{c}_lag1"] = mac[c].shift(1).bfill()
        mac_feats[f"{c}_lag2"] = mac[c].shift(2).bfill()
        mac_feats[f"{c}_lag3"] = mac[c].shift(3).bfill()
        mac_feats[f"{c}_mom3"] = (mac[c].shift(1) - mac[c].shift(4)).fillna(0.0)
        mac_feats[f"{c}_yoy"] = mac[c].diff(12).fillna(0.0)
    df = df.merge(mac_feats, on="Date", how="left")

    is_ev = (df["Powertrain_Application"] == "EV").astype(float)
    is_ice = (df["Powertrain_Application"] == "ICE").astype(float)
    ev_centered = df["EV_Adoption_Index"] - 100.0
    df["EV_x_EV_Index"] = is_ev * ev_centered
    df["ICE_x_EV_Index"] = is_ice * ev_centered
    df["EV_x_Time"] = is_ev * df["Time_Idx"]
    df["ICE_x_Time"] = is_ice * df["Time_Idx"]

    # Family-specific optimal lagged PMI & New Vehicle Orders
    pmi_dates = {
        int(k[:4]) * 12 + int(k[5:7]): float(v)
        for k, v in mac.set_index("Date")["ISM_Manufacturing_PMI"].items()
    }
    nvo_dates = {
        int(k[:4]) * 12 + int(k[5:7]): float(v)
        for k, v in mac.set_index("Date")["New_Vehicle_Orders_Index"].items()
    }
    opt_pmi_lags = df["Product_Family"].map(
        lambda f: best_lag_map.get((f, "ISM_Manufacturing_PMI"), 1)
    ).values
    opt_nvo_lags = df["Product_Family"].map(
        lambda f: best_lag_map.get((f, "New_Vehicle_Orders_Index"), 1)
    ).values
    t_ints = df["Target_YM_Int"].values
    df["Opt_Lag_PMI"] = [pmi_dates.get(int(t - l), 50.0) for t, l in zip(t_ints, opt_pmi_lags)]
    df["Opt_Lag_NVO"] = [nvo_dates.get(int(t - l), 100.0) for t, l in zip(t_ints, opt_nvo_lags)]

    # 4. Calendar Events
    df["Price_Hike_Pct"] = df["Date"].map(event_tables["price_hike"]).fillna(0.0)
    df["Pre_Hike_Signal_Pct"] = df["Date"].map(event_tables["pre_hike"]).fillna(0.0)
    df["Surcharge_Pct"] = df["Date"].map(event_tables["surcharge"]).fillna(0.0)

    reg_ev = event_tables["reg_events"]
    lny, ram, summ, cov = [], [], [], []
    for d, r in zip(df["Date"].values, df["Region"].values):
        s = reg_ev.get((d, r))
        if s is None:
            lny.append(0.0); ram.append(0.0); summ.append(0.0); cov.append(0.0)
        else:
            lny.append(s["lny_frac"]); ram.append(s["ramadan_frac"])
            summ.append(s["summer_shutdown"]); cov.append(s["covid_disruption"])
    df["Holiday_LNY_Frac"] = lny
    df["Holiday_Ramadan_Frac"] = ram
    df["Summer_Shutdown"] = summ
    df["COVID_Disruption"] = cov

    cre = event_tables["cust_reg_events"]
    c_shut, c_cap = [], []
    for d, c, r in zip(df["Date"].values, df["Customer_Name"].values, df["Region"].values):
        s = cre.get((d, c, r))
        if s is None:
            c_shut.append(0.0); c_cap.append(0.0)
        else:
            c_shut.append(s["cust_shutdown"]); c_cap.append(s["cust_capacity_up"])
    df["Cust_Shutdown_Flag"] = c_shut
    df["Cust_Capacity_Up_Flag"] = c_cap

    ce = event_tables["chan_events"]
    df["Channel_Disruption"] = [
        ce.get((d, ch), 0.0) for d, ch in zip(df["Date"].values, df["Sales_Channel"].values)
    ]

    # 5. Pooled low-variance seasonality lookup
    if pooled_seas is not None:
        reg_map = pooled_seas.get("reg", {})
        fam_map = pooled_seas.get("fam", {})
        fr_map = pooled_seas.get("fam_reg", {})
        df["Pooled_Reg_Seas"] = [
            reg_map.get((r, int(m)), 1.0)
            for r, m in zip(df["Region"].values, df["Month"].values)
        ]
        df["Pooled_Fam_Seas"] = [
            fam_map.get((f, int(m)), 1.0)
            for f, m in zip(df["Product_Family"].values, df["Month"].values)
        ]
        df["Pooled_FamReg_Seas"] = [
            fr_map.get((f, r, int(m)), reg_map.get((r, int(m)), 1.0))
            for f, r, m in zip(
                df["Product_Family"].values, df["Region"].values, df["Month"].values
            )
        ]
    else:
        df["Pooled_Reg_Seas"] = 1.0
        df["Pooled_Fam_Seas"] = 1.0
        df["Pooled_FamReg_Seas"] = 1.0

    # 6. Deterministic ordering-cycle phase & clockwork flags
    has_hist = df["History_Months"].fillna(0.0) > 0
    if "Last_Order_YM" in df.columns and "Est_Period" in df.columns:
        last_ord = df["Last_Order_YM"].fillna(origin_ym_int).astype(float)
        period = df["Est_Period"].fillna(1.0).clip(lower=1.0)
        reg_sc = df["Cycle_Regularity"].fillna(0.0)

        delta_t = df["Target_YM_Int"].astype(float) - last_ord
        phase_mod = np.mod(delta_t, period)
        df["Cycle_Phase_Mod"] = phase_mod
        df["Cycle_Is_Order_Month"] = (
            has_hist & (period >= 2) & (phase_mod == 0) & (reg_sc >= 0.65)
        ).astype(float)
        df["Is_Clockwork_Off_Cycle"] = (
            has_hist & (period >= 2) & (phase_mod != 0) & (reg_sc >= 0.65)
        ).astype(float)
        df["Cycle_Phase_Sin"] = np.sin(2.0 * np.pi * phase_mod / period)
        df["Cycle_Phase_Cos"] = np.cos(2.0 * np.pi * phase_mod / period)
    else:
        df["Cycle_Phase_Mod"] = 0.0
        df["Cycle_Is_Order_Month"] = 0.0
        df["Is_Clockwork_Off_Cycle"] = 0.0
        df["Cycle_Phase_Sin"] = 0.0
        df["Cycle_Phase_Cos"] = 1.0

    # 7. Denoised Cycle-Aware Baseline Anchor (Med_24m + Mean_24m + EWM * pooled seasonality)
    if "seas_m1" in df.columns:
        seas_arr = np.ones(len(df), dtype=float)
        zrate_arr = np.zeros(len(df), dtype=float)
        cnt_arr = np.zeros(len(df), dtype=float)
        for m in range(1, 13):
            mask_m = df["Month"].values == m
            seas_arr[mask_m] = df.loc[mask_m, f"seas_m{m}"].fillna(1.0).values
            zrate_arr[mask_m] = df.loc[mask_m, f"zrate_m{m}"].fillna(0.0).values
            cnt_arr[mask_m] = df.loc[mask_m, f"cnt_m{m}"].fillna(0.0).values

        df["Target_Month_Seas_Idx"] = seas_arr
        df["Target_Month_Zero_Rate"] = zrate_arr
        df["Target_Month_Obs_Count"] = cnt_arr

        # Denoised seasonal multiplier: 65% Pooled Region + 35% Pooled Family-Region
        pooled_s = 0.65 * df["Pooled_Reg_Seas"].values + 0.35 * df["Pooled_FamReg_Seas"].values

        # Robust level anchor (verified -5.74 pp WAPE vs Mean_12m * per-series seas)
        smooth_level = (
            0.45 * df["Med_24m"].fillna(0.0)
            + 0.35 * df["Mean_24m"].fillna(0.0)
            + 0.20 * df["EWM_Level"].fillna(0.0)
        )
        df["Seas_Adjusted_Base"] = np.maximum(smooth_level * pooled_s, 0.0)

        # On-cycle clockwork series anchor strictly on non-zero order size * pooled seasonality
        nz_level = 0.55 * df["NZ_Med_24m"].fillna(0.0) + 0.45 * df["NZ_EWM_Level"].fillna(0.0)
        is_cw_on = df["Cycle_Is_Order_Month"] > 0.5
        is_cw_off = df["Is_Clockwork_Off_Cycle"] > 0.5

        shifting_holiday_month = (
            ((df["Region"] == "Asia Pacific") & df["Month"].isin([1, 2]))
            | ((df["Region"] == "Middle East & Africa") & df["Month"].isin([1, 2, 3, 4]))
        )
        df["Is_Fixed_Calendar_Zero"] = (
            has_hist
            & (df["Target_Month_Obs_Count"] >= 3)
            & (df["Target_Month_Zero_Rate"] >= 0.999)
            & (~shifting_holiday_month)
        ).astype(float)

        cycle_base = np.where(
            is_cw_off | (df["Is_Fixed_Calendar_Zero"] > 0.5),
            0.0,
            np.where(is_cw_on, nz_level * pooled_s, df["Seas_Adjusted_Base"]),
        )
        df["Cycle_Aware_Base"] = np.maximum(cycle_base, 0.0)
    else:
        df["Target_Month_Seas_Idx"] = 1.0
        df["Target_Month_Zero_Rate"] = 0.0
        df["Target_Month_Obs_Count"] = 0.0
        df["Seas_Adjusted_Base"] = 0.0
        df["Is_Fixed_Calendar_Zero"] = 0.0
        df["Cycle_Aware_Base"] = 0.0

    return df