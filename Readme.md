# Celeris Automobili Industrial — Demand Forecasting & Planning Control Tower

End-to-end, reproducible demand forecasting pipeline and 12-view interactive S&OP Control Tower for **Celeris Automobili Industrial**, forecasting **19 months (`2026-11` → `2028-05`)** across **3,068 SKU-Region series (`58,219` rows)** at Expert tier across all 5 workstreams.

---

## 1. Quick Start (One-Command Pipeline & Dashboard)

### Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
*(macOS Apple Silicon: run `brew install libomp` once if LightGBM requires OpenMP.)*

Place the 7 raw CSV files (`train.csv`, `test.csv`, `sku_master.csv`, `macro_indicators.csv`, `inventory_stockouts.csv`, `events_calendar.csv`, `sample_submission.csv`) inside `data/`.

### Run Full Pipeline (Cleaning → Anomalies → Macro/Features → Backtest → Forecast)
```bash
python run_pipeline.py
```

### Run Automated QA & Business-Rule Verification Suite
```bash
pytest -v tests/test_pipeline.py
```

### Launch 12-View Interactive Control Tower
```bash
streamlit run dashboard/app.py
```
Opens locally at `http://localhost:8501`. All views read directly from generated files in `outputs/` with unified sidebar filters shared across history and forecast.

---

## 2. Project Structure

```text
celeris_control_tower/
├── data/                         # 7 raw input CSV files
├── src/
│   ├── config.py                 # Canonical mappings, constants, 3 rolling backtest cutoffs
│   ├── cleaning.py               # Workstream 1: Deterministic cleaning + DQ JSON report
│   ├── anomalies.py              # Workstream 2: 3-layer unconstrained demand + anomaly register
│   ├── features.py               # Workstream 3: Macro lead/lag stability, pooled seasonality, cycles, events
│   ├── reconciliation.py         # Workstream 4b: MinT-Shrinkage reconciliation + largest-remainder rounding
│   └── forecasting.py            # Workstream 4a: Baselines, 4-model ensemble, LOCO stacker, CQR, lifecycle
├── dashboard/app.py              # Workstream 5: 12-view Streamlit + Plotly Control Tower
├── tests/test_pipeline.py        # Automated pytest suite verifying all 7 business rules
├── outputs/                      # Generated artifacts (submission.csv, reports, parquet tables)
├── run_pipeline.py               # One-command orchestrator
├── requirements.txt
└── README.md
```

---

## 3. Forensic Audit: Data Errors Fixed vs. Real Events Modeled

### Workstream 1 — Data Errors Fixed (`outputs/data_quality_report.json`)
| File | Key Repairs Verified |
|---|---|
| `train.csv` | Deduplicated `248,248 → 244,566` rows (`3,682` repeated `Record_ID`s resolved by latest `Load_Timestamp`; `0` remaining `(Date, SKU_ID, Region)` duplicates). Fixed `30` transposed `2062-MM → 2026-MM` dates, `9,821` non-standard date strings, `246` `Year/Month` mismatches, `3,683` malformed `SKU_ID`s, `5,912` Region variants, `3,683` channel variants, `2,455` customer typos, and recovered `942/942` missing Regions (`916` via unique slot, `26` via context; `0` unresolved). Fixed `491` negative sign flips, `498` EUR currency errors, `385` India 10x price errors, `614` cents 100x errors, `295` zero prices, `293`/`75` 10x/100x UOM unit errors, and `2,406` revenue reconciliations. Recovered `3,164` missing `Units_Sold` deterministically from `Revenue_USD / Canon_Price` while preserving `380` dual-null ERP migration rows as `NaN` (never forced to `0`). |
| `macro_indicators.csv` | Deduplicated `114 → 113` rows (kept revised `2022-06`), multiplied `12` 2019 fractional `Auto_Production_YoY_Pct` rows by `100`, fixed `ISM_Manufacturing_PMI` `2023-05` decimal shift (`4.79 → 47.9`), and interpolated `5` missing cells. |
| `sku_master.csv` | Fixed `PMP-004` Business Unit contradiction (`Power & Drive Systems → Advanced Manufacturing`) and logged `BSS-018` early-launch contradiction (`16` valid rows before `Launch_Date = 2025-06`) without deleting real history. |

### Workstream 2 — Real Events & Unconstrained Demand (`outputs/anomaly_register.csv`: `10,496` records)
1. **3-Layer Unconstrained Demand Reconstruction:**
   - *Layer 1 (`5,804` rows):* Replaced constrained `Units_Sold` with `Requested_Units` on documented stockouts.
   - *Layer 2 (`1,041` rows):* Subtracted prior-month `Backordered_Units` carry-in echo from the subsequent non-stockout month.
   - *Layer 3 (`940` rows):* Uplifted undocumented `Electrical Assemblies` semiconductor-shortage rows (`EV-019`, `2021-03..2022-09`) using the monthly empirical fill-rate curve from documented ELA stockouts.
2. **Real Structural & Account Events Separated from Errors:** Logged `714` pre-buying spikes (`EV-030` ahead of the Jan 2023 +9% price hike), `622` CUSUM structural breaks (including `EV-052` Helios India capacity expansion), `379` `EV-033` Latin America ERP migration gaps, `27` customer switches, `26` ordering-cycle shifts, `11` lost/dormant accounts, `924` positive-only rolling log-MAD one-off shocks (without misflagging recurring holiday zeros), and explicit disruption records (`EV-009..013` COVID, `EV-038` Nexon strike, `EV-039` Red Sea logistics).

---

## 4. Macro Lead/Lag Stability & EV Shift (`outputs/macro_lead_lag_analysis.csv`)

Evaluated `392` `(Product_Family, Macro_Indicator, Lag 0..6)` pairs using YoY-differenced cross-correlations across Early (`2019–2022`) vs. Late (`2023–2026`) regimes:
* **Strongest Leading Drivers (Lag = 1 month):** `New_Vehicle_Orders_Index` leads `Hydraulic Systems` (`r = 0.904`, stable), `Electrical Assemblies` (`r = 0.899`, stable), `Powertrain Components` (`r = 0.838`), and `Precision Machined Parts` (`r = 0.829`). `ISM_Manufacturing_PMI` stably leads `Hydraulic Systems` (lag 1, `r = 0.816`) and `Electrical Assemblies` (lag 2, `r = 0.787`).
* **Regime Stability Finding:** Only `19/56` peak family-indicator pairs remain stable across 2019–2022 vs. 2023–2026; e.g., `Brake & Safety Systems` vs. `New_Vehicle_Orders_Index` flipped from `+0.81` to `-0.52`, justifying recency weighting and direct multi-horizon modeling.
* **Secular EV Shift:** Modeled explicitly via `Powertrain_Application × EV_Adoption_Index` and `Time_Idx` interactions. Forecasted average monthly demand reflects continued `ICE` contraction (`18,122` in late 2026 → `17,612` in 2027 → `16,662` in 2028) alongside sustained `EV` adoption (`16,030 → 16,562 → 16,048`).

---

## 5. Forecasting Architecture & 19-Month Backtest Results (`outputs/backtest_metrics.csv`)

### Architecture
* **Validation:** 3-origin rolling-origin backtest across the full **19-month horizon** (cutoffs `2024-05`, `2024-11`, `2025-03`) with strict Leave-One-Cutoff-Out (LOCO) meta-stacking and bias calibration.
* **4 Direct Multi-Horizon Base Learners (Zero Recursive Drift):**
  1. Direct LightGBM (`50%` Tweedie + `50%` L1)
  2. Direct CatBoost (`MAE` on `log1p(y)`)
  3. Two-Stage Hurdle Model (`LGBMClassifier × Positive LGBMRegressor`)
  4. Denoised Log-Ratio Ridge Macro-EV Extrapolator anchored on 24-month median + pooled `(Region, Month)` / `(Family, Region, Month)` seasonality
* **Clockwork Cycle & Fixed-Calendar Zero-Locking:** Verified >99% accurate on 2026 holdout; eliminates off-cycle bleed before reconciliation (`15.20%` submission zero rate vs. `14.66%` historical + 156 EOL zeros).
* **Lifecycle & Future Events:** Hard-zeros all `156` post-EOL rows (`PMP-003`, `PTC-005`, `THM-001`), transfers cold-start demand via empirical `Replaces_SKU` S-curve (`PMP-020`: 1,562; `PTC-029`: 1,055; `THM-012`: 831 units), and models `EV-062` (`Orion Driveline | North America` Jul–Aug 2027 shutdown calibrated from `EV-038`), Jan 2027 (+4.5%) / Jan 2028 (+4.0%) price increases, and shifting Lunar New Year / Ramadan windows.
* **Hierarchical Coherence & Uncertainty:** Minimum Trace (`MinT-Shrink`) reconciliation (`Total → BU → Family → Region → SKU-Region`), Split Conformalized Quantile Regression (`P10/P50/P90`), and exact largest-remainder non-negative integer rounding.

### 3-Origin × 19-Month Backtest Leaderboard
| Model | WAPE | Bias | RMSE | Notes |
|---|---:|---:|---:|---|
| `Seasonal_Naive` (Baseline 1) | `36.24%` | `-0.66%` | `31.070` | Last year's calendar month |
| `SBA_ETS` (Baseline 2) | `50.15%` | `-1.30%` | `40.619` | Flat Croston/SBA smears deterministic bi-monthly/quarterly cycles |
| `Ridge_Trend` | `33.19%` | `-8.38%` | `28.179` | Linear macro/EV extrapolator |
| `LGBM` (Tweedie + L1) | `28.66%` | `+1.57%` | `24.136` | Direct global GBDT |
| `Hurdle` (2-Stage) | `28.57%` | `+0.54%` | `24.206` | Specialized intermittent cycle learner |
| `CatBoost` (MAE log1p) | `27.62%` | `-1.65%` | `23.946` | Strongest single WAPE learner |
| **`Ensemble` (Strict LOCO)** | **`27.88%`** | **`+0.50%`** | **`23.832`** | **Best RMSE, near-zero bias, `-23.1%` WAPE vs. Seasonal Naive** |
| **Conformal `[P10, P90]` Coverage** | **`81.27%`** | — | — | Target nominal coverage: `80.0%` |

---

## 6. Headline 19-Month Forecast & Key Deliverables

* **Official Submission (`outputs/submission.csv`):** `58,219` rows, `0` nulls/negatives, `2,803,696` total P50 forecast units (`$2,497.60M` projected revenue).
* **Procurement Plan (`outputs/procurement_plan.csv`):** `3,914,484` P90 units (`1,110,788` safety-buffer units; `$3,529.14M` P90 capacity budget).
* **Planner Exception Queue (`outputs/planner_exceptions.csv`):** `309` revenue-prioritized SKU-Region alerts (lifecycle transitions, ≥40% surges/drops vs. L12M, wide P90/P50 spread).
* **12-View Control Tower (`dashboard/app.py`):** All 5 required views (`Executive Overview`, `Data Health`, `Demand Explorer`, `Anomaly Monitor`, `Forecast & Accuracy`) + all 7 optional Expert views (`Macro Lead/Lag Heatmap`, `What-If Scenario Simulator`, `Procurement Plan`, `Lifecycle`, `Stockouts & Lost Sales`, `Planner Exception Queue`, `Data Download Center`).