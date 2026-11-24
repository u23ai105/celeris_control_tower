import warnings
from typing import Dict, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.config import BACKTEST_CUTOFFS, DATA_DIR, FORECAST_HORIZON, OUT_DIR
from src.features import (
    _int_to_ym,
    _ym_to_int,
    attach_macro_and_event_features,
    build_event_feature_table,
    compute_pooled_seasonality,
    compute_series_profile,
)
from src.reconciliation import largest_remainder_round, mint_shrink_reconcile

warnings.filterwarnings("ignore")

CAT_COLS = [
    "SKU_ID",
    "Product_Family",
    "Business_Unit",
    "Region",
    "Customer_Name",
    "Sales_Channel",
    "Powertrain_Application",
    "Intermittency_Class",
]

NUM_FEATURES = [
    "Horizon",
    "Month",
    "Year",
    "Quarter",
    "Month_Sin",
    "Month_Cos",
    "Time_Idx",
    "Size_Numeric",
    "SKU_Age_Months",
    "Months_To_EOL",
    "ADI",
    "CV2",
    "Zero_Rate",
    "Est_Period",
    "Cycle_Regularity",
    "Trailing_Zero_Months",
    "Mean_6m",
    "Mean_12m",
    "Mean_24m",
    "Med_12m",
    "Med_24m",
    "EWM_Level",
    "NZ_EWM_Level",
    "NZ_Mean_6m",
    "NZ_Mean_12m",
    "NZ_Med_24m",
    "NZ_Last_Order_Size",
    "Last_1m",
    "Last_2m",
    "Last_3m",
    "Log_Slope_18m",
    "Std_12m",
    "YoY_Ratio_12m",
    "History_Months",
    "Pooled_Reg_Seas",
    "Pooled_Fam_Seas",
    "Pooled_FamReg_Seas",
    "Target_Month_Seas_Idx",
    "Target_Month_Zero_Rate",
    "Target_Month_Obs_Count",
    "Seas_Adjusted_Base",
    "Cycle_Aware_Base",
    "Cycle_Phase_Mod",
    "Cycle_Is_Order_Month",
    "Is_Clockwork_Off_Cycle",
    "Is_Fixed_Calendar_Zero",
    "Cycle_Phase_Sin",
    "Cycle_Phase_Cos",
    "ISM_Manufacturing_PMI",
    "ISM_Manufacturing_PMI_lag1",
    "ISM_Manufacturing_PMI_lag2",
    "ISM_Manufacturing_PMI_mom3",
    "Auto_Production_YoY_Pct",
    "Auto_Production_YoY_Pct_lag1",
    "Auto_Production_YoY_Pct_lag2",
    "Auto_Production_YoY_Pct_mom3",
    "Vehicle_Inventory_Days",
    "Vehicle_Inventory_Days_lag1",
    "Raw_Material_Cost_Index",
    "Raw_Material_Cost_Index_lag1",
    "New_Vehicle_Orders_Index",
    "New_Vehicle_Orders_Index_lag1",
    "New_Vehicle_Orders_Index_lag2",
    "New_Vehicle_Orders_Index_mom3",
    "Industrial_Util_Rate_Pct",
    "Industrial_Util_Rate_Pct_lag1",
    "Fuel_Price_Index",
    "Fuel_Price_Index_lag1",
    "EV_Adoption_Index",
    "EV_Adoption_Index_lag1",
    "EV_Adoption_Index_yoy",
    "EV_x_EV_Index",
    "ICE_x_EV_Index",
    "EV_x_Time",
    "ICE_x_Time",
    "Opt_Lag_PMI",
    "Opt_Lag_NVO",
    "Price_Hike_Pct",
    "Pre_Hike_Signal_Pct",
    "Surcharge_Pct",
    "Holiday_LNY_Frac",
    "Holiday_Ramadan_Frac",
    "Summer_Shutdown",
    "COVID_Disruption",
    "Cust_Shutdown_Flag",
    "Cust_Capacity_Up_Flag",
    "Channel_Disruption",
]


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.sum(np.abs(y_true)))
    return float(np.sum(np.abs(y_true - y_pred)) / denom) if denom > 0 else 0.0


def bias_pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.sum(np.abs(y_true)))
    return float(np.sum(y_pred - y_true) / denom) if denom > 0 else 0.0


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ------------------------------------------------------------------
# BASELINE MODELS: Seasonal Naive & Syntetos-Boylan (SBA) / ETS
# ------------------------------------------------------------------
def compute_baselines(hist_df: pd.DataFrame, target_panel: pd.DataFrame) -> pd.DataFrame:
    """Computes Seasonal Naive (y_{t-12}) and SBA/ETS baselines for every target row."""
    h = hist_df.dropna(subset=["Unconstrained_Demand"]).sort_values(["SKU_ID", "Region", "Date"])
    last_year_lookup = {}
    sba_lookup = {}

    for (sku, reg), g in h.groupby(["SKU_ID", "Region"], sort=False):
        y = g["Unconstrained_Demand"].values.astype(float)
        months = g["Month"].values
        m_map = {}
        for mo in range(1, 13):
            vals = y[months == mo]
            if len(vals) > 0:
                m_map[mo] = float(vals[-1])
        fallback = float(np.mean(y[-12:])) if len(y) >= 1 else 0.0
        last_year_lookup[(sku, reg)] = (m_map, fallback)

        alpha = 0.15
        nz_idx = np.where(y > 0)[0]
        if len(nz_idx) == 0:
            sba_val = 0.0
        elif (y > 0).mean() < 0.70 and len(nz_idx) >= 2:
            z = float(y[nz_idx[0]])
            p = float(np.mean(np.diff(nz_idx)))
            for k in range(1, len(nz_idx)):
                z = alpha * y[nz_idx[k]] + (1 - alpha) * z
                p = alpha * (nz_idx[k] - nz_idx[k - 1]) + (1 - alpha) * p
            sba_val = (1.0 - alpha / 2.0) * (z / max(p, 1.0))
        else:
            s = float(y[0])
            for val in y[1:]:
                s = 0.20 * val + 0.80 * s
            sba_val = s
        sba_lookup[(sku, reg)] = max(sba_val, 0.0)

    sn_preds = []
    sba_preds = []
    for sku, reg, mo in zip(
        target_panel["SKU_ID"].values,
        target_panel["Region"].values,
        target_panel["Month"].values,
    ):
        m_map, fb = last_year_lookup.get((sku, reg), ({}, 0.0))
        sn_preds.append(m_map.get(int(mo), fb))
        sba_preds.append(sba_lookup.get((sku, reg), 0.0))

    return pd.DataFrame(
        {"Pred_Seasonal_Naive": sn_preds, "Pred_SBA_ETS": sba_preds},
        index=target_panel.index,
    )


# ------------------------------------------------------------------
# DIRECT MULTI-HORIZON TRAINING DATASET BUILDER (Zero Recursive Drift)
# ------------------------------------------------------------------
def build_direct_training_set(
    hist_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    sku_master: pd.DataFrame,
    event_tables: dict,
    best_lag_map: Dict[Tuple[str, str], int],
    cutoff_ym: str,
    n_origins: int = 8,
    stride_months: int = 3,
) -> pd.DataFrame:
    c_int = int(cutoff_ym[:4]) * 12 + int(cutoff_ym[5:7])
    min_ym_int = int(hist_df["Date"].min()[:4]) * 12 + int(hist_df["Date"].min()[5:7])

    origins = []
    for k in range(n_origins):
        o = c_int - 6 - k * stride_months
        if o >= min_ym_int + 18:
            origins.append(o)
    origins = sorted(set(origins))

    frames = []
    hist_ym_int = _ym_to_int(hist_df["Date"])
    for o_int in origins:
        o_str = _int_to_ym(o_int)
        past = hist_df[hist_ym_int <= o_int]
        fut = hist_df[
            (hist_ym_int > o_int)
            & (hist_ym_int <= min(o_int + FORECAST_HORIZON, c_int))
            & hist_df["Unconstrained_Demand"].notna()
        ].copy()
        if past.empty or fut.empty:
            continue

        prof = compute_series_profile(past)
        p_seas = compute_pooled_seasonality(past)
        fut = fut.merge(prof, on=["SKU_ID", "Region"], how="inner")
        fut = attach_macro_and_event_features(
            fut,
            macro_df,
            sku_master,
            event_tables,
            best_lag_map,
            origin_ym_int=o_int,
            pooled_seas=p_seas,
        )
        age_from_cutoff = max(c_int - o_int, 1)
        recency_w = float(np.exp(-0.025 * age_from_cutoff))
        fut["Train_Weight"] = fut["Sample_Weight"].fillna(1.0) * recency_w
        fut["Origin_YM"] = o_str
        frames.append(fut)

    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------------
# MODEL 4 (V2.1): Denoised Log-Ratio Ridge Macro-EV Trend Extrapolator
# ------------------------------------------------------------------
def fit_predict_ridge_extrapolator(
    train_df: pd.DataFrame, pred_df: pd.DataFrame
) -> np.ndarray:
    """
    Fits strongly regularized Ridge on log((y + 1) / (Cycle_Aware_Base + 1))
    anchored on the denoised 24m-median + pooled-seasonality baseline.
    """
    trend_cols = [
        "Horizon",
        "EV_Adoption_Index",
        "EV_Adoption_Index_yoy",
        "ISM_Manufacturing_PMI_lag1",
        "Auto_Production_YoY_Pct_lag1",
        "New_Vehicle_Orders_Index_lag1",
        "Industrial_Util_Rate_Pct_lag1",
        "Log_Slope_18m",
        "Holiday_LNY_Frac",
        "Holiday_Ramadan_Frac",
        "Summer_Shutdown",
        "Pre_Hike_Signal_Pct",
        "Price_Hike_Pct",
    ]
    preds = np.zeros(len(pred_df), dtype=float)

    for (fam, pt), g_pred in pred_df.groupby(
        ["Product_Family", "Powertrain_Application"], sort=False, dropna=False
    ):
        base_te = np.maximum(g_pred["Cycle_Aware_Base"].fillna(0.0).values.astype(float), 0.0)
        g_tr = train_df[
            (train_df["Product_Family"] == fam)
            & (train_df["Powertrain_Application"] == pt)
            & (train_df["Cycle_Aware_Base"] > 2.0)
            & (train_df["Unconstrained_Demand"] > 0.0)
        ]
        if len(g_tr) < 50:
            preds[g_pred.index] = base_te
            continue

        base_tr = g_tr["Cycle_Aware_Base"].values.astype(float)
        y_tr = g_tr["Unconstrained_Demand"].values.astype(float)
        y_ratio_log = np.clip(np.log((y_tr + 1.0) / (base_tr + 1.0)), -1.0, 1.0)
        w_tr = g_tr["Train_Weight"].values * base_tr

        X_tr = g_tr[trend_cols].fillna(0.0).values.astype(float)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        model = Ridge(alpha=300.0)
        model.fit(X_tr_s, y_ratio_log, sample_weight=w_tr)

        X_te = g_pred[trend_cols].fillna(0.0).values.astype(float)
        X_te_s = scaler.transform(X_te)
        mult = np.exp(np.clip(model.predict(X_te_s), -0.35, 0.35))
        preds[g_pred.index] = np.where(base_te > 1e-6, base_te * mult, 0.0)

    return np.maximum(preds, 0.0)


# ------------------------------------------------------------------
# 4-MODEL DIVERSE ENSEMBLE + CONFORMAL QUANTILE MODELS
# ------------------------------------------------------------------
def _prep_features(df: pd.DataFrame, cat_categories: dict | None = None):
    X = df[NUM_FEATURES + CAT_COLS].copy()
    cats_out = {}
    for c in CAT_COLS:
        X[c] = X[c].astype(object).fillna("MISSING").astype(str)
        if cat_categories is None:
            cat_type = pd.CategoricalDtype(categories=sorted(X[c].unique()))
            cats_out[c] = cat_type
        else:
            cat_type = cat_categories[c]
        X[c] = X[c].astype(cat_type)
    for c in NUM_FEATURES:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)
    return X, (cats_out if cat_categories is None else cat_categories)


def fit_and_predict_base_models(
    train_ds: pd.DataFrame,
    pred_ds: pd.DataFrame,
    fit_quantiles: bool = False,
) -> dict:
    """
    Trains 4 diverse direct multi-horizon base learners + optional P10/P50/P90 quantile models:
      1. Direct LightGBM Blend: 50% Tweedie + 50% Regression L1 (bias-balanced, zero ratio blow-up)
      2. Direct CatBoost (MAE on log1p(y)) for high-order categorical interactions
      3. Two-Stage Hurdle Model (P(y > 0) x E[y | y > 0])
      4. Denoised Log-Ratio Ridge Macro-EV Trend Extrapolator
    """
    X_tr, cat_types = _prep_features(train_ds)
    X_te, _ = _prep_features(pred_ds, cat_types)
    y_tr = np.maximum(train_ds["Unconstrained_Demand"].values.astype(float), 0.0)
    w_tr = train_ds["Train_Weight"].values.astype(float)

    # 1. Global Direct LightGBM (50% Tweedie + 50% L1)
    lgb_tweedie = lgb.LGBMRegressor(
        objective="tweedie",
        tweedie_variance_power=1.25,
        n_estimators=750,
        learning_rate=0.035,
        num_leaves=63,
        min_child_samples=40,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=5.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_tweedie.fit(X_tr, y_tr, sample_weight=w_tr)
    p_lgb_tw = np.maximum(lgb_tweedie.predict(X_te), 0.0)

    lgb_l1 = lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=750,
        learning_rate=0.035,
        num_leaves=63,
        min_child_samples=40,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=5.0,
        random_state=102,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_l1.fit(X_tr, y_tr, sample_weight=w_tr)
    p_lgb_l1 = np.maximum(lgb_l1.predict(X_te), 0.0)
    p_lgb = 0.50 * p_lgb_tw + 0.50 * p_lgb_l1

    # 2. Global Direct CatBoost (MAE on log1p(y))
    X_tr_cb = X_tr[NUM_FEATURES].copy()
    X_te_cb = X_te[NUM_FEATURES].copy()
    for c in CAT_COLS:
        X_tr_cb[c] = (
            pd.Series(train_ds[c].values, index=X_tr_cb.index)
            .astype(object)
            .fillna("MISSING")
            .astype(str)
        )
        X_te_cb[c] = (
            pd.Series(pred_ds[c].values, index=X_te_cb.index)
            .astype(object)
            .fillna("MISSING")
            .astype(str)
        )

    cb_model = CatBoostRegressor(
        loss_function="MAE",
        iterations=700,
        learning_rate=0.055,
        depth=7,
        l2_leaf_reg=5.0,
        random_seed=42,
        verbose=0,
        thread_count=-1,
    )
    cb_model.fit(X_tr_cb, np.log1p(y_tr), cat_features=CAT_COLS, sample_weight=w_tr)
    p_catboost = np.maximum(np.expm1(np.clip(cb_model.predict(X_te_cb), 0.0, 11.0)), 0.0)

    # 3. Two-Stage Hurdle Model (P(y > 0) x E[y | y > 0])
    y_bin = (y_tr > 0).astype(int)
    clf = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=450,
        learning_rate=0.04,
        num_leaves=47,
        min_child_samples=35,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    clf.fit(X_tr, y_bin, sample_weight=w_tr)
    prob_pos = clf.predict_proba(X_te)[:, 1]

    pos_mask = y_tr > 0
    reg_pos = lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=550,
        learning_rate=0.04,
        num_leaves=63,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    reg_pos.fit(X_tr[pos_mask], np.log1p(y_tr[pos_mask]), sample_weight=w_tr[pos_mask])
    size_pos = np.maximum(np.expm1(np.clip(reg_pos.predict(X_te), 0.0, 11.0)), 0.0)
    p_hurdle = np.where(prob_pos < 0.22, 0.0, prob_pos * size_pos)

    # 4. Denoised Log-Ratio Ridge Macro-EV Trend Extrapolator
    p_ridge = fit_predict_ridge_extrapolator(train_ds, pred_ds.reset_index(drop=True))

    res = {
        "Pred_LGBM": p_lgb,
        "Pred_CatBoost": p_catboost,
        "Pred_Hurdle": p_hurdle,
        "Pred_Ridge_Trend": p_ridge,
        "Prob_Positive": prob_pos,
    }

    if fit_quantiles:
        for alpha, label in [(0.10, "Raw_P10"), (0.50, "Raw_P50"), (0.90, "Raw_P90")]:
            qm = lgb.LGBMRegressor(
                objective="quantile",
                alpha=alpha,
                n_estimators=500,
                learning_rate=0.04,
                num_leaves=55,
                min_child_samples=40,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
            qm.fit(X_tr, y_tr, sample_weight=w_tr)
            res[label] = np.maximum(qm.predict(X_te), 0.0)

    return res


# ------------------------------------------------------------------
# REGIME-AWARE SLSQP META-STACKER + ZERO-LOCK + BIAS CALIBRATION
# ------------------------------------------------------------------
def fit_meta_weights(oof_df: pd.DataFrame) -> Dict[Tuple[str, str], np.ndarray]:
    """Learns optimal non-negative simplex weights across the 4 base learners minimizing WAPE."""
    model_cols = ["Pred_LGBM", "Pred_CatBoost", "Pred_Hurdle", "Pred_Ridge_Trend"]
    df = oof_df.copy()
    df["I_Grp"] = np.where(
        df["Intermittency_Class"].isin(["Intermittent", "Lumpy"]), "Intermittent", "Smooth"
    )
    df["H_Bucket"] = pd.cut(
        df["Horizon"], bins=[0, 6, 12, 24], labels=["H1_6", "H7_12", "H13_19"]
    ).astype(str)

    weights = {}
    for (ig, hb), g in df.groupby(["I_Grp", "H_Bucket"]):
        P = g[model_cols].values
        y = g["Actual"].values
        if len(y) < 100 or y.sum() <= 0:
            weights[(ig, hb)] = np.array([0.35, 0.35, 0.15, 0.15])
            continue

        def loss(w):
            pred = P @ w
            return np.sum(np.abs(y - pred)) / np.sum(np.abs(y))

        init = np.array([0.35, 0.35, 0.15, 0.15])
        bounds = [(0.0, 1.0)] * 4
        cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        opt = minimize(loss, init, method="SLSQP", bounds=bounds, constraints=cons)
        weights[(ig, hb)] = opt.x if opt.success else init

    return weights


def apply_meta_weights(df: pd.DataFrame, weights: Dict[Tuple[str, str], np.ndarray]) -> np.ndarray:
    model_cols = ["Pred_LGBM", "Pred_CatBoost", "Pred_Hurdle", "Pred_Ridge_Trend"]
    ig_arr = np.where(
        df["Intermittency_Class"].isin(["Intermittent", "Lumpy"]), "Intermittent", "Smooth"
    )
    hb_arr = (
        pd.cut(df["Horizon"], bins=[0, 6, 12, 24], labels=["H1_6", "H7_12", "H13_19"])
        .astype(str)
        .values
    )
    P = df[model_cols].values
    out = np.zeros(len(df), dtype=float)
    default_w = np.array([0.35, 0.35, 0.15, 0.15])
    for i in range(len(df)):
        w = weights.get((ig_arr[i], hb_arr[i]), default_w)
        out[i] = float(P[i] @ w)
    return np.maximum(out, 0.0)


def compute_cycle_zero_lock_mask(df: pd.DataFrame) -> np.ndarray:
    """
    Identifies deterministic zero-demand cells ONLY for series with observed history:
      1. Clockwork off-cycle month (Est_Period >= 2, Regularity >= 0.65 -> 99.01% zero)
      2. Fixed-calendar holiday shutdown (0 in >=3 past years, non-shifting -> 99.27% zero)
      3. Low Hurdle classifier probability (Prob_Positive < 0.22)
    """
    has_hist = df["History_Months"].fillna(0.0).values > 0
    cw_off = df["Is_Clockwork_Off_Cycle"].fillna(0.0).values > 0.5
    cal_zero = df["Is_Fixed_Calendar_Zero"].fillna(0.0).values > 0.5
    low_prob = df["Prob_Positive"].fillna(1.0).values < 0.22
    return has_hist & (cw_off | cal_zero | low_prob)


def fit_bias_calibrators(
    oof_df: pd.DataFrame, pred_col: str = "Pred_Ensemble"
) -> Dict[Tuple[str, str], float]:
    """Shrunk multiplicative bias-correction factors per (Product_Family, Horizon_Bucket)."""
    df = oof_df.copy()
    df["H_Bucket"] = pd.cut(
        df["Horizon"], bins=[0, 6, 12, 24], labels=["H1_6", "H7_12", "H13_19"]
    ).astype(str)
    cal = {}
    for (fam, hb), g in df.groupby(["Product_Family", "H_Bucket"]):
        p_sum = float(g[pred_col].sum())
        a_sum = float(g["Actual"].sum())
        if p_sum > 100.0:
            raw_mult = a_sum / p_sum
            shrunk = 0.70 * raw_mult + 0.30 * 1.0
            cal[(fam, hb)] = float(np.clip(shrunk, 0.96, 1.06))
        else:
            cal[(fam, hb)] = 1.0
    return cal


def apply_bias_calibrators(
    df: pd.DataFrame, pred: np.ndarray, cal: Dict[Tuple[str, str], float]
) -> np.ndarray:
    hb_arr = (
        pd.cut(df["Horizon"], bins=[0, 6, 12, 24], labels=["H1_6", "H7_12", "H13_19"])
        .astype(str)
        .values
    )
    mult = np.array(
        [cal.get((f, h), 1.0) for f, h in zip(df["Product_Family"].values, hb_arr)],
        dtype=float,
    )
    return np.maximum(pred * mult, 0.0)


# ------------------------------------------------------------------
# LIFECYCLE TRANSFER ENGINE (Replaces_SKU Empirical S-Curve + EOL Zero)
# ------------------------------------------------------------------
def learn_replacement_scurve(
    train_df: pd.DataFrame, sku_master: pd.DataFrame
) -> Dict[int, float]:
    pairs = sku_master[
        sku_master["Replaces_SKU"].notna() & (sku_master["Lifecycle_Status"] == "Active")
    ]
    shares_by_dt: Dict[int, list[float]] = {}

    for r in pairs.itertuples(index=False):
        new_sku = r.SKU_ID
        old_sku = r.Replaces_SKU
        l_int = int(r.Launch_Date[:4]) * 12 + int(r.Launch_Date[5:7])

        g_new = train_df[train_df["SKU_ID"] == new_sku].groupby("Date")["Unconstrained_Demand"].sum()
        g_old = train_df[train_df["SKU_ID"] == old_sku].groupby("Date")["Unconstrained_Demand"].sum()
        all_dates = sorted(set(g_new.index) | set(g_old.index))

        for d in all_dates:
            dt_int = int(d[:4]) * 12 + int(d[5:7]) - l_int
            if 0 <= dt_int <= 6:
                v_new = float(g_new.get(d, 0.0))
                v_old = float(g_old.get(d, 0.0))
                if v_new + v_old > 0:
                    shares_by_dt.setdefault(dt_int, []).append(v_new / (v_new + v_old))

    curve = {}
    default_curve = {0: 0.25, 1: 0.55, 2: 0.80, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0}
    for dt_int in range(0, 7):
        vals = shares_by_dt.get(dt_int, [])
        curve[dt_int] = float(np.median(vals)) if len(vals) >= 2 else default_curve[dt_int]
    return curve


def apply_lifecycle_and_future_events(
    pred_df: pd.DataFrame,
    train_df: pd.DataFrame,
    sku_master: pd.DataFrame,
    scurve: Dict[int, float],
) -> pd.DataFrame:
    df = pred_df.copy()

    nex_eu = train_df[
        (train_df["Customer_Name"] == "Nexon Motor Group")
        & (train_df["Region"] == "Europe")
        & train_df["Unconstrained_Demand"].notna()
    ]
    strike_mean = nex_eu[nex_eu["Date"].isin(["2023-10", "2023-11"])]["Unconstrained_Demand"].mean()
    normal_mean = nex_eu[
        nex_eu["Date"].between("2023-04", "2024-04")
        & (~nex_eu["Date"].isin(["2023-10", "2023-11"]))
    ]["Unconstrained_Demand"].mean()
    ev062_factor = (
        float(np.clip(strike_mean / max(normal_mean, 1.0), 0.15, 0.75))
        if pd.notna(strike_mean)
        else 0.35
    )

    planned = sku_master[sku_master["Lifecycle_Status"] == "Planned Launch"]
    eol_map = sku_master.set_index("SKU_ID")["EOL_Date"].to_dict()
    launch_map = sku_master.set_index("SKU_ID")["Launch_Date"].to_dict()

    for r in planned.itertuples(index=False):
        new_sku = r.SKU_ID
        old_sku = r.Replaces_SKU
        if pd.isna(old_sku):
            continue
        l_int = int(r.Launch_Date[:4]) * 12 + int(r.Launch_Date[5:7])
        old_eol = eol_map.get(old_sku)
        eol_int = (
            int(old_eol[:4]) * 12 + int(old_eol[5:7]) if pd.notna(old_eol) else l_int + 2
        )

        old_rows = df[df["SKU_ID"] == old_sku].set_index(["Date", "Region"])
        new_idx = df[df["SKU_ID"] == new_sku].index

        for idx in new_idx:
            dt = df.at[idx, "Date"]
            reg = df.at[idx, "Region"]
            t_int = int(dt[:4]) * 12 + int(dt[5:7])
            delta_t = t_int - l_int

            if delta_t < 0:
                df.at[idx, "Pred_Ensemble"] = 0.0
                df.at[idx, "Raw_P10"] = 0.0
                df.at[idx, "Raw_P50"] = 0.0
                df.at[idx, "Raw_P90"] = 0.0
                continue

            if (dt, reg) in old_rows.index:
                total_p = float(old_rows.loc[(dt, reg), "Pred_Ensemble"])
                total_p10 = float(old_rows.loc[(dt, reg), "Raw_P10"])
                total_p50 = float(old_rows.loc[(dt, reg), "Raw_P50"])
                total_p90 = float(old_rows.loc[(dt, reg), "Raw_P90"])
            else:
                total_p = total_p10 = total_p50 = total_p90 = 0.0

            w_new = 1.0 if t_int > eol_int else scurve.get(delta_t, 0.80)
            df.at[idx, "Pred_Ensemble"] = total_p * w_new
            df.at[idx, "Raw_P10"] = total_p10 * w_new
            df.at[idx, "Raw_P50"] = total_p50 * w_new
            df.at[idx, "Raw_P90"] = total_p90 * w_new

            if t_int <= eol_int:
                old_match = df[
                    (df["SKU_ID"] == old_sku) & (df["Date"] == dt) & (df["Region"] == reg)
                ].index
                for oi in old_match:
                    w_old = 1.0 - w_new
                    df.at[oi, "Pred_Ensemble"] = total_p * w_old
                    df.at[oi, "Raw_P10"] = total_p10 * w_old
                    df.at[oi, "Raw_P50"] = total_p50 * w_old
                    df.at[oi, "Raw_P90"] = total_p90 * w_old

    ev062_mask = (
        (df["Customer_Name"] == "Orion Driveline")
        & (df["Region"] == "North America")
        & df["Date"].isin(["2027-07", "2027-08"])
    )
    for col in ["Pred_Ensemble", "Raw_P10", "Raw_P50", "Raw_P90"]:
        if col in df.columns:
            df.loc[ev062_mask, col] *= ev062_factor

    eol_col = df["SKU_ID"].map(eol_map)
    launch_col = df["SKU_ID"].map(launch_map)
    post_eol = eol_col.notna() & (df["Date"] > eol_col)
    pre_launch = launch_col.notna() & (df["Date"] < launch_col)
    rule5_lock = post_eol | pre_launch
    for col in ["Pred_Ensemble", "Raw_P10", "Raw_P50", "Raw_P90"]:
        if col in df.columns:
            df.loc[rule5_lock, col] = 0.0

    prev_lock = (
        df["Is_Zero_Locked"].values if "Is_Zero_Locked" in df.columns else np.zeros(len(df), dtype=bool)
    )
    df["Is_Zero_Locked"] = prev_lock | rule5_lock.values
    return df


# ------------------------------------------------------------------
# SPLIT CONFORMALIZED QUANTILE CALIBRATION (Guaranteed ~80% [P10, P90])
# ------------------------------------------------------------------
def calibrate_conformal_quantiles(
    backtest_df: pd.DataFrame, forecast_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    bt = backtest_df.copy()
    fc = forecast_df.copy()

    scale_bt = np.maximum(bt["Pred_Ensemble"].values, 5.0)
    score_lo = (bt["Raw_P10"].values - bt["Actual"].values) / scale_bt
    score_hi = (bt["Actual"].values - bt["Raw_P90"].values) / scale_bt
    bt["c_score"] = np.maximum(score_lo, score_hi)

    q_by_h = {}
    for h in range(1, FORECAST_HORIZON + 1):
        sub = bt[bt["Horizon"] == h]["c_score"].values
        q_by_h[h] = float(np.quantile(sub, 0.80)) if len(sub) >= 30 else 0.0

    for frame in (bt, fc):
        q_arr = frame["Horizon"].map(q_by_h).fillna(0.0).values
        anchor = (
            frame["Pred_Reconciled"].values
            if "Pred_Reconciled" in frame.columns
            else frame["Pred_Ensemble"].values
        )
        sc = np.maximum(anchor, 5.0)

        p50 = anchor
        p10 = np.minimum(frame["Raw_P10"].values - q_arr * sc, p50)
        p90 = np.maximum(frame["Raw_P90"].values + q_arr * sc, p50)

        if "Is_Zero_Locked" in frame.columns:
            locked = frame["Is_Zero_Locked"].values
            p10 = np.where(locked, 0.0, np.maximum(p10, 0.0))
            p50 = np.where(locked, 0.0, np.maximum(p50, 0.0))
            p90 = np.where(locked, 0.0, np.maximum(p90, 0.0))
        else:
            p10 = np.maximum(p10, 0.0)
            p50 = np.maximum(p50, 0.0)
            p90 = np.maximum(p90, 0.0)

        frame["P10"] = np.round(p10).astype(int)
        frame["P50"] = np.round(p50).astype(int)
        frame["P90"] = np.round(p90).astype(int)
        frame["P10"] = np.minimum(frame["P10"], frame["P50"])
        frame["P90"] = np.maximum(frame["P90"], frame["P50"])

    return bt, fc


# ------------------------------------------------------------------
# FUTURE PRICE PROJECTION & OPERATIONAL OUTPUTS
# ------------------------------------------------------------------
def attach_future_prices_and_ops(
    forecast_df: pd.DataFrame, train_df: pd.DataFrame, sku_master: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fc = forecast_df.copy()
    p2026 = (
        train_df[train_df["Year"] == 2026]
        .groupby("SKU_ID")["Unit_Price_USD"]
        .median()
        .to_dict()
    )
    p_all = train_df.groupby("SKU_ID")["Unit_Price_USD"].median().to_dict()
    replaces_map = sku_master.set_index("SKU_ID")["Replaces_SKU"].to_dict()

    def _base_2026_price(sku):
        if sku in p2026:
            return p2026[sku]
        pred_sku = replaces_map.get(sku)
        if pd.notna(pred_sku) and pred_sku in p2026:
            return p2026[pred_sku]
        return p_all.get(sku, float(np.median(list(p2026.values()))))

    base_p = fc["SKU_ID"].map(_base_2026_price)
    mult = np.where(fc["Year"] == 2026, 1.0, np.where(fc["Year"] == 2027, 1.045, 1.045 * 1.040))
    fc["Projected_Price_USD"] = (base_p * mult).round(2)
    fc["Forecast_Revenue_USD"] = (fc["Units_Sold_Forecast"] * fc["Projected_Price_USD"]).round(2)
    fc["Forecast_Revenue_P10_USD"] = (fc["P10"] * fc["Projected_Price_USD"]).round(2)
    fc["Forecast_Revenue_P90_USD"] = (fc["P90"] * fc["Projected_Price_USD"]).round(2)

    proc = (
        fc.groupby(["Date", "Business_Unit", "Product_Family", "Region"], as_index=False)
        .agg(
            Forecast_P50_Units=("Units_Sold_Forecast", "sum"),
            Procurement_P90_Units=("P90", "sum"),
            Forecast_Revenue_USD=("Forecast_Revenue_USD", "sum"),
            Procurement_P90_Budget_USD=("Forecast_Revenue_P90_USD", "sum"),
        )
    )
    proc["Safety_Buffer_Units"] = (
        proc["Procurement_P90_Units"] - proc["Forecast_P50_Units"]
    ).clip(lower=0)
    proc.to_csv(OUT_DIR / "procurement_plan.csv", index=False)

    hist_l12 = (
        train_df[train_df["Date"] >= "2025-11"]
        .groupby(["SKU_ID", "Region"])["Unconstrained_Demand"]
        .mean()
        .rename("Hist_L12_Monthly_Mean")
    )
    fc_ser = (
        fc.groupby(
            ["SKU_ID", "Region", "Product_Family", "Customer_Name", "Lifecycle_Status"],
            as_index=False,
        )
        .agg(
            Fc_Monthly_Mean=("Units_Sold_Forecast", "mean"),
            Fc_Total_19m=("Units_Sold_Forecast", "sum"),
            Fc_P90_Total=("P90", "sum"),
            Fc_Rev_19m=("Forecast_Revenue_USD", "sum"),
        )
        .merge(hist_l12, on=["SKU_ID", "Region"], how="left")
    )
    fc_ser["Hist_L12_Monthly_Mean"] = fc_ser["Hist_L12_Monthly_Mean"].fillna(0.0)
    fc_ser["Pct_Change_vs_L12"] = (
        fc_ser["Fc_Monthly_Mean"] - fc_ser["Hist_L12_Monthly_Mean"]
    ) / (fc_ser["Hist_L12_Monthly_Mean"] + 1.0)
    fc_ser["Uncertainty_Ratio"] = (fc_ser["Fc_P90_Total"] - fc_ser["Fc_Total_19m"]) / (
        fc_ser["Fc_Total_19m"] + 1.0
    )

    exc_reasons = []
    for r in fc_ser.itertuples(index=False):
        tags = []
        if r.Lifecycle_Status in ("EOL Announced", "Planned Launch"):
            tags.append(f"Lifecycle: {r.Lifecycle_Status}")
        if r.Pct_Change_vs_L12 >= 0.40 and r.Fc_Total_19m >= 100:
            tags.append(f"Surge {r.Pct_Change_vs_L12:+.0%} vs L12M")
        elif r.Pct_Change_vs_L12 <= -0.40 and r.Hist_L12_Monthly_Mean >= 5:
            tags.append(f"Drop {r.Pct_Change_vs_L12:+.0%} vs L12M")
        if r.Uncertainty_Ratio >= 0.65 and r.Fc_Total_19m >= 100:
            tags.append(f"Wide P90/P50 spread ({r.Uncertainty_Ratio:.2f})")
        exc_reasons.append("; ".join(tags))

    fc_ser["Exception_Reason"] = exc_reasons
    exceptions = (
        fc_ser[fc_ser["Exception_Reason"] != ""]
        .sort_values("Fc_Rev_19m", ascending=False)
        .reset_index(drop=True)
    )
    exceptions.to_csv(OUT_DIR / "planner_exceptions.csv", index=False)

    return fc, proc, exceptions


# ------------------------------------------------------------------
# MASTER WORKSTREAM 4 RUNNER (V2.1)
# ------------------------------------------------------------------
def run_forecasting_pipeline(
    train_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    sku_master: pd.DataFrame,
    best_lag_map: Dict[Tuple[str, str], int],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    all_dates = sorted(set(train_df["Date"].unique()) | set(test_df["Date"].unique()))
    event_tables = build_event_feature_table(
        dates=all_dates,
        regions=sorted(test_df["Region"].unique()),
        families=sorted(test_df["Product_Family"].unique()),
        customers=sorted(
            set(train_df["Customer_Name"].dropna()) | set(test_df["Customer_Name"].dropna())
        ),
        channels=sorted(test_df["Sales_Channel"].unique()),
    )
    scurve = learn_replacement_scurve(train_df, sku_master)

    # ==============================================================
    # 1. 3-ORIGIN 19-MONTH ROLLING-ORIGIN BACKTEST
    # ==============================================================
    bt_frames = []
    train_ym_int = _ym_to_int(train_df["Date"])

    for cutoff in BACKTEST_CUTOFFS:
        print(f"      -> Running backtest origin cutoff={cutoff}...")
        c_int = int(cutoff[:4]) * 12 + int(cutoff[5:7])
        past = train_df[train_ym_int <= c_int].copy()
        val = train_df[
            (train_ym_int > c_int)
            & (train_ym_int <= c_int + FORECAST_HORIZON)
            & train_df["Unconstrained_Demand"].notna()
        ].copy()

        prof = compute_series_profile(past)
        p_seas = compute_pooled_seasonality(past)
        val = val.merge(prof, on=["SKU_ID", "Region"], how="inner")
        val = attach_macro_and_event_features(
            val,
            macro_df,
            sku_master,
            event_tables,
            best_lag_map,
            origin_ym_int=c_int,
            pooled_seas=p_seas,
        )

        base_preds = compute_baselines(past, val)
        val["Pred_Seasonal_Naive"] = base_preds["Pred_Seasonal_Naive"].values
        val["Pred_SBA_ETS"] = base_preds["Pred_SBA_ETS"].values

        tr_ds = build_direct_training_set(
            past, macro_df, sku_master, event_tables, best_lag_map, cutoff_ym=cutoff, n_origins=7
        )
        m_out = fit_and_predict_base_models(tr_ds, val, fit_quantiles=True)
        for k in [
            "Pred_LGBM",
            "Pred_CatBoost",
            "Pred_Hurdle",
            "Pred_Ridge_Trend",
            "Prob_Positive",
            "Raw_P10",
            "Raw_P50",
            "Raw_P90",
        ]:
            val[k] = m_out[k]

        val["Actual"] = val["Unconstrained_Demand"].astype(float)
        val["Cutoff"] = cutoff
        bt_frames.append(val)

    backtest_all = pd.concat(bt_frames, ignore_index=True)

    # Leak-free Leave-One-Cutoff-Out Meta-Stacking + Zero-Lock + Bias Calibration
    bt_ens = np.zeros(len(backtest_all), dtype=float)
    bt_zero_lock = compute_cycle_zero_lock_mask(backtest_all)
    backtest_all["Is_Zero_Locked"] = bt_zero_lock

    for cutoff in BACKTEST_CUTOFFS:
        tr_mask = (backtest_all["Cutoff"] != cutoff).values
        te_mask = (backtest_all["Cutoff"] == cutoff).values
        w_fold = fit_meta_weights(backtest_all[tr_mask])
        p_raw_tr = apply_meta_weights(backtest_all[tr_mask], w_fold)
        p_raw_tr = np.where(bt_zero_lock[tr_mask], 0.0, p_raw_tr)
        tmp_tr = backtest_all[tr_mask].copy()
        tmp_tr["Pred_Raw_Ens"] = p_raw_tr
        cal_fold = fit_bias_calibrators(tmp_tr, pred_col="Pred_Raw_Ens")

        p_te = apply_meta_weights(backtest_all[te_mask], w_fold)
        p_te = np.where(bt_zero_lock[te_mask], 0.0, p_te)
        p_te = apply_bias_calibrators(backtest_all[te_mask], p_te, cal_fold)
        bt_ens[te_mask] = p_te

    backtest_all["Pred_Ensemble"] = bt_ens
    for qcol in ["Raw_P10", "Raw_P50", "Raw_P90"]:
        backtest_all.loc[bt_zero_lock, qcol] = 0.0

    # Fit final production meta-weights & bias calibrators on all 3 OOF cutoffs
    meta_weights = fit_meta_weights(backtest_all)
    p_all_raw = np.where(bt_zero_lock, 0.0, apply_meta_weights(backtest_all, meta_weights))
    tmp_all = backtest_all.copy()
    tmp_all["Pred_Raw_Ens"] = p_all_raw
    bias_cal = fit_bias_calibrators(tmp_all, pred_col="Pred_Raw_Ens")

    eval_models = [
        "Pred_Seasonal_Naive",
        "Pred_SBA_ETS",
        "Pred_LGBM",
        "Pred_CatBoost",
        "Pred_Hurdle",
        "Pred_Ridge_Trend",
        "Pred_Ensemble",
    ]
    metric_rows = []
    for mcol in eval_models:
        metric_rows.append(
            {
                "Slice_Type": "Overall",
                "Slice_Value": "All",
                "Model": mcol.replace("Pred_", ""),
                "WAPE": round(wape(backtest_all["Actual"].values, backtest_all[mcol].values), 4),
                "Bias": round(bias_pct(backtest_all["Actual"].values, backtest_all[mcol].values), 4),
                "RMSE": round(rmse(backtest_all["Actual"].values, backtest_all[mcol].values), 3),
            }
        )
    for col_slice in ["Product_Family", "Region", "Intermittency_Class"]:
        for val_slice, g in backtest_all.groupby(col_slice):
            for mcol in ["Pred_Seasonal_Naive", "Pred_SBA_ETS", "Pred_Ensemble"]:
                metric_rows.append(
                    {
                        "Slice_Type": col_slice,
                        "Slice_Value": str(val_slice),
                        "Model": mcol.replace("Pred_", ""),
                        "WAPE": round(wape(g["Actual"].values, g[mcol].values), 4),
                        "Bias": round(bias_pct(g["Actual"].values, g[mcol].values), 4),
                        "RMSE": round(rmse(g["Actual"].values, g[mcol].values), 3),
                    }
                )
    for h, g in backtest_all.groupby("Horizon"):
        for mcol in ["Pred_Seasonal_Naive", "Pred_SBA_ETS", "Pred_Ensemble"]:
            metric_rows.append(
                {
                    "Slice_Type": "Horizon",
                    "Slice_Value": f"H{int(h):02d}",
                    "Model": mcol.replace("Pred_", ""),
                    "WAPE": round(wape(g["Actual"].values, g[mcol].values), 4),
                    "Bias": round(bias_pct(g["Actual"].values, g[mcol].values), 4),
                    "RMSE": round(rmse(g["Actual"].values, g[mcol].values), 3),
                }
            )
    metrics_df = pd.DataFrame(metric_rows)

    # ==============================================================
    # 2. FINAL FULL-HISTORY TRAINING & 19-MONTH FORECAST (2026-11..2028-05)
    # ==============================================================
    print("      -> Training final full-history ensemble for 2026-11..2028-05...")
    final_cutoff = "2026-10"
    final_c_int = 2026 * 12 + 10
    prof_full = compute_series_profile(train_df)
    p_seas_full = compute_pooled_seasonality(train_df)

    test_panel = test_df.merge(prof_full, on=["SKU_ID", "Region"], how="left")
    test_panel["Intermittency_Class"] = test_panel["Intermittency_Class"].fillna("Smooth")
    test_panel = attach_macro_and_event_features(
        test_panel,
        macro_df,
        sku_master,
        event_tables,
        best_lag_map,
        origin_ym_int=final_c_int,
        pooled_seas=p_seas_full,
    )

    tr_full = build_direct_training_set(
        train_df, macro_df, sku_master, event_tables, best_lag_map, cutoff_ym=final_cutoff, n_origins=9
    )
    final_out = fit_and_predict_base_models(tr_full, test_panel, fit_quantiles=True)
    for k in [
        "Pred_LGBM",
        "Pred_CatBoost",
        "Pred_Hurdle",
        "Pred_Ridge_Trend",
        "Prob_Positive",
        "Raw_P10",
        "Raw_P50",
        "Raw_P90",
    ]:
        test_panel[k] = final_out[k]

    p_ens_raw = apply_meta_weights(test_panel, meta_weights)
    fc_cycle_lock = compute_cycle_zero_lock_mask(test_panel)
    p_ens_raw = np.where(fc_cycle_lock, 0.0, p_ens_raw)
    p_ens_cal = apply_bias_calibrators(test_panel, p_ens_raw, bias_cal)

    test_panel["Pred_Ensemble"] = p_ens_cal
    test_panel["Is_Zero_Locked"] = fc_cycle_lock
    for qcol in ["Raw_P10", "Raw_P50", "Raw_P90"]:
        test_panel.loc[fc_cycle_lock, qcol] = 0.0

    # Apply Lifecycle S-Curve Transfer (Replaces_SKU), EV-062 Shutdown & Hard Rule 5 EOL Zero
    test_panel = apply_lifecycle_and_future_events(test_panel, train_df, sku_master, scurve)

    # MinT-Shrinkage Hierarchical Reconciliation (preserves all zero-locked cells)
    test_panel = mint_shrink_reconcile(
        test_panel, backtest_all, pred_col="Pred_Ensemble", out_col="Pred_Reconciled"
    )

    # Split Conformalized Quantile Calibration (P10 / P50 / P90)
    backtest_all, test_panel = calibrate_conformal_quantiles(backtest_all, test_panel)

    # Exact Largest-Remainder Non-Negative Integer Rounding per month
    final_int = np.zeros(len(test_panel), dtype=int)
    for dt, g in test_panel.groupby("Date", sort=False):
        final_int[g.index.values] = largest_remainder_round(g["Pred_Reconciled"].values)
    test_panel["Units_Sold_Forecast"] = final_int
    test_panel["P50"] = test_panel["Units_Sold_Forecast"]
    test_panel["P10"] = np.minimum(test_panel["P10"], test_panel["P50"])
    test_panel["P90"] = np.maximum(test_panel["P90"], test_panel["P50"])

    cov_80 = float(
        (
            (backtest_all["Actual"] >= backtest_all["P10"])
            & (backtest_all["Actual"] <= backtest_all["P90"])
        ).mean()
    )
    metrics_df = pd.concat(
        [
            metrics_df,
            pd.DataFrame(
                [
                    {
                        "Slice_Type": "Calibration",
                        "Slice_Value": "P10_P90_Coverage",
                        "Model": "Conformal_Ensemble",
                        "WAPE": round(cov_80, 4),
                        "Bias": 0.0,
                        "RMSE": 0.0,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    test_panel, _, _ = attach_future_prices_and_ops(test_panel, train_df, sku_master)

    metrics_df.to_csv(OUT_DIR / "backtest_metrics.csv", index=False)
    backtest_keep = [
        "Cutoff",
        "Date",
        "Horizon",
        "SKU_ID",
        "Region",
        "Product_Family",
        "Business_Unit",
        "Intermittency_Class",
        "Actual",
        "Pred_Seasonal_Naive",
        "Pred_SBA_ETS",
        "Pred_LGBM",
        "Pred_CatBoost",
        "Pred_Hurdle",
        "Pred_Ridge_Trend",
        "Pred_Ensemble",
        "P10",
        "P50",
        "P90",
    ]
    backtest_all[backtest_keep].to_parquet(OUT_DIR / "backtest_predictions.parquet", index=False)
    test_panel.to_parquet(OUT_DIR / "forecast_full.parquet", index=False)

    sub = test_df[["Date", "SKU_ID", "Region"]].merge(
        test_panel[["Date", "SKU_ID", "Region", "Units_Sold_Forecast"]],
        on=["Date", "SKU_ID", "Region"],
        how="left",
        validate="one_to_one",
    )
    assert sub["Units_Sold_Forecast"].isna().sum() == 0, "Null values found in submission!"
    assert (sub["Units_Sold_Forecast"] < 0).sum() == 0, "Negative values found in submission!"
    sub["Units_Sold_Forecast"] = sub["Units_Sold_Forecast"].astype(int)
    sub.to_csv(OUT_DIR / "submission.csv", index=False)

    return backtest_all, metrics_df, test_panel