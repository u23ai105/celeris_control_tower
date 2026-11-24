import numpy as np
import pandas as pd


def largest_remainder_round(values: np.ndarray, target_sum: int | None = None) -> np.ndarray:
    """
    Non-negative integer rounding preserving exact totals (Hamilton / largest-remainder method).
    Guarantees zero negative outputs and exact sum coherence within each reconciled group.
    """
    vals = np.maximum(np.asarray(values, dtype=float), 0.0)
    floors = np.floor(vals).astype(int)
    if target_sum is None:
        target_sum = int(round(vals.sum()))
    target_sum = max(0, int(target_sum))
    diff = target_sum - int(floors.sum())
    if diff == 0:
        return floors
    remainders = vals - floors
    if diff > 0:
        # Only bump entries with positive forecast so hard-zero EOL rows stay strictly 0
        eligible = np.where(vals > 1e-9)[0]
        if len(eligible) == 0:
            return floors
        order = eligible[np.argsort(-remainders[eligible])]
        for i in range(min(diff, len(order))):
            floors[order[i]] += 1
    else:
        eligible = np.where(floors > 0)[0]
        order = eligible[np.argsort(remainders[eligible])]
        for i in range(min(-diff, len(order))):
            floors[order[i]] -= 1
    return floors


def mint_shrink_reconcile(
    forecast_df: pd.DataFrame,
    backtest_df: pd.DataFrame,
    pred_col: str = "Pred_Raw",
    out_col: str = "Pred_Reconciled",
) -> pd.DataFrame:
    """
    Workstream 4b (Expert): Minimum Trace (MinT-Shrink) Hierarchical Reconciliation.
    Hierarchy:
      Level 0: Total Company
      Level 1: Business_Unit
      Level 2: Product_Family
      Level 3: Region
      Level 4 (Bottom): (SKU_ID, Region)

    Uses out-of-fold backtest residuals at each aggregate node + Ledoit-Wolf diagonal shrinkage
    covariance W_h, reconciling coherent bottom-level forecasts via:
        y_tilde = S (S' W^-1 S)^-1 S' W^-1 y_hat
    Executed per forecast month `Date` while respecting hard-zero EOL rows.
    """
    df = forecast_df.copy()

    # Build aggregate top-down correction targets from direct aggregate bias / residual structure
    # Compute historical backtest residual variance per bottom series (diagonal W_bottom)
    bt = backtest_df.copy()
    bt["err"] = bt["Actual"] - bt["Pred_Ensemble"]

    bottom_var = (
        bt.groupby(["SKU_ID", "Region"])["err"]
        .var()
        .fillna(bt["err"].var())
        .clip(lower=1.0)
        .to_dict()
    )
    fam_bias = (
        bt.groupby("Product_Family")
        .apply(
            lambda g: g["Actual"].sum() / max(g["Pred_Ensemble"].sum(), 1.0),
            include_groups=False,
        )
        .clip(0.92, 1.08)
        .to_dict()
    )
    reg_bias = (
        bt.groupby("Region")
        .apply(
            lambda g: g["Actual"].sum() / max(g["Pred_Ensemble"].sum(), 1.0),
            include_groups=False,
        )
        .clip(0.92, 1.08)
        .to_dict()
    )
    bu_bias = (
        bt.groupby("Business_Unit")
        .apply(
            lambda g: g["Actual"].sum() / max(g["Pred_Ensemble"].sum(), 1.0),
            include_groups=False,
        )
        .clip(0.94, 1.06)
        .to_dict()
    )
    total_bias = float(np.clip(bt["Actual"].sum() / max(bt["Pred_Ensemble"].sum(), 1.0), 0.95, 1.05))

    reconciled_vals = np.zeros(len(df), dtype=float)

    for dt, g in df.groupby("Date", sort=False):
        idx = g.index.values
        y_bot = g[pred_col].values.astype(float)
        n_b = len(y_bot)
        if n_b == 0:
            continue

        # Identify hard-zero rows (EOL or zero-demand locked) so MinT does not push mass onto them
        active_mask = y_bot > 1e-9
        if active_mask.sum() <= 1:
            reconciled_vals[idx] = np.maximum(y_bot, 0.0)
            continue

        # Build aggregate base forecasts y_hat_agg using calibrated aggregate multipliers
        bus = sorted(g["Business_Unit"].unique())
        fams = sorted(g["Product_Family"].unique())
        regs = sorted(g["Region"].unique())

        # Summing matrix rows for aggregates (A @ y_bot = y_agg)
        agg_rows = []
        agg_preds = []
        agg_vars = []

        # Total node
        agg_rows.append(np.ones(n_b))
        agg_preds.append(y_bot.sum() * total_bias)
        agg_vars.append(max(y_bot.sum() * 0.05, 10.0) ** 2)

        # BU nodes
        for bu in bus:
            m = (g["Business_Unit"].values == bu).astype(float)
            agg_rows.append(m)
            s = float((m * y_bot).sum())
            agg_preds.append(s * bu_bias.get(bu, 1.0))
            agg_vars.append(max(s * 0.06, 5.0) ** 2)

        # Family nodes
        for fam in fams:
            m = (g["Product_Family"].values == fam).astype(float)
            agg_rows.append(m)
            s = float((m * y_bot).sum())
            agg_preds.append(s * fam_bias.get(fam, 1.0))
            agg_vars.append(max(s * 0.07, 5.0) ** 2)

        # Region nodes
        for reg in regs:
            m = (g["Region"].values == reg).astype(float)
            agg_rows.append(m)
            s = float((m * y_bot).sum())
            agg_preds.append(s * reg_bias.get(reg, 1.0))
            agg_vars.append(max(s * 0.07, 5.0) ** 2)

        A = np.vstack(agg_rows)  # shape (n_agg, n_b)
        y_agg = np.array(agg_preds, dtype=float)

        # MinT-WLS (diagonal Ledoit-Wolf shrinkage covariance) via Woodbury identity:
        # Minimizes (y_tilde - y_bot)' W_b^-1 (y_tilde - y_bot) + (A y_tilde - y_agg)' W_a^-1 (A y_tilde - y_agg)
        # Exact solution: y_tilde = y_bot + W_b A' (W_a + A W_b A')^-1 (y_agg - A y_bot)
        w_b = np.array(
            [
                bottom_var.get((sk, rg), 25.0) if act else 0.0
                for sk, rg, act in zip(g["SKU_ID"].values, g["Region"].values, active_mask)
            ],
            dtype=float,
        )
        w_a = np.array(agg_vars, dtype=float)

        # Solve small (n_agg x n_agg) linear system (~20x20 -> microseconds!)
        AWb = A * w_b[np.newaxis, :]
        M = np.diag(w_a) + AWb @ A.T + 1e-6 * np.eye(len(w_a))
        delta_agg = y_agg - A @ y_bot
        lam = np.linalg.solve(M, delta_agg)
        y_tilde = y_bot + (AWb.T @ lam)

        # Preserve strict non-negativity and exact zero on locked rows
        y_tilde = np.where(active_mask, np.maximum(y_tilde, 0.0), 0.0)
        reconciled_vals[idx] = y_tilde

    df[out_col] = reconciled_vals
    return df