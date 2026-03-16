# FX Valuation — Reproducible Stage 1 + Stage 2 Pipeline

This repository implements an end-to-end quantitative FX fair-value workflow across G10 currencies:

- **Stage 1**: rolling driver discovery (univariate OLS, significance filtering, diversified top-driver selection)
- **Stage 2**: walk-forward machine-learning fair-value models (OLS, Ridge, Lasso, ElasticNet, SGD, XGBoost, LightGBM, stacked ensemble)
- **Audit layer**: signal quality, forward-return validation, strategy equity curves, model comparison, and cross-currency summary outputs

The instructions below are designed so a new user can run the project on their own machine and reproduce the outputs in `data/processed/` and `data/audits/`.

---

## 1) Repository layout

```text
fxvaluation_d200_am3483/
├── config.py
├── requirements.txt
├── setup_env.ps1
├── data/
│   ├── rawdata.xlsx                     # Primary source dataset
│   ├── processed/                       # Master training CSVs per currency
│   ├── audits/                          # Main Stage 2 audit outputs
│   └── audits_test/                     # Test/sandbox audit outputs
├── notebooks/
│   ├── explore_dataframes.ipynb
│   ├── ultimate_dataframes.ipynb
│   ├── ols_regressions.ipynb
│   └── fairvalue_ols.ipynb
└── src/
    ├── data/                            # Data loading + transformation pipeline
    ├── rolling_univariate_ols.py        # Stage 1 rolling single-factor regressions
    ├── diversified_top_drivers_history.py
    ├── stage2_ml_models.py              # Stage 2 model runner (CLI)
    ├── stage2_ml_performance_audit.py   # Stage 2 audit runner (CLI)
    ├── stage2_fair_value_runner.py      # Convenience fair-value runner
    └── stage2_policy_agent.py           # Prototype RL threshold policy agent
```

---

## 2) System requirements

- **Python**: 3.10+ recommended (3.8+ should work per dependency bounds)
- **OS**: Linux/macOS/Windows
- **RAM**: 8 GB minimum, 16 GB recommended for full multi-model + multi-currency runs
- **Disk**: 2+ GB free space for outputs/plots

Python dependencies are pinned by lower bounds in `requirements.txt` and include:
`pandas`, `numpy`, `statsmodels`, `scikit-learn`, `xgboost`, `lightgbm`, `matplotlib`, `seaborn`, `plotly`, `openpyxl`, `jupyter`.

---

## 3) Environment setup

### Option A (Windows PowerShell, automated)

```powershell
# From repo root
.\setup_env.ps1
```

### Option B (cross-platform, manual)

```bash
# From repo root
python -m venv .venv

# Activate
# Linux/macOS
source .venv/bin/activate
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Verify installation

```bash
python -c "import pandas, numpy, sklearn, statsmodels, xgboost, lightgbm, plotly; print('env-ok')"
```

---

## 4) Data expectations and reproducibility assumptions

1. `data/rawdata.xlsx` must exist and be readable (it is the raw source used by the loaders).
2. The project builds many objects at import time; always run commands from **repo root**.
3. Some model classes are deterministic by construction, while tree/ensemble methods may show minor run-to-run drift from backend threading/library differences.
4. To maximize reproducibility across machines, keep:
    - same Python/dependency versions,
    - same input Excel file,
    - same command parameters,
    - same timezone/locale defaults where possible.

---

## 5) Full runbook to replicate results

## Step 0 — Sanity-check the source file

```bash
python -c "from pathlib import Path; p=Path('data/rawdata.xlsx'); print(p.resolve(), p.exists())"
```

Expected: path printed with `True`.

## Step 1 — Build master training CSVs (Stage 0/1 preparation)

This creates one file per currency in `data/processed/` (e.g., `eur_master.csv`, `jpy_master.csv`).

```bash
python -m src.data.build_model_ready_data
```

Optional custom parameters:

```bash
python - <<'PY'
from src.data.build_model_ready_data import save_master_training_csvs
save_master_training_csvs(window=250, min_significance=95.0, top_n=3)
print('master-csvs-built')
PY
```

## Step 2 — Run Stage 2 model suite for one currency

Generates per-model output CSVs in `data/model_outputs/<currency>/`.

```bash
python -m src.stage2_ml_models eur \
    --models ols ridge lasso elasticnet sgd xgb lgbm stacked \
    --window 250 \
    --error-sum-window 10 \
    --recenter-window 60 \
    --cv-splits 4 \
    --retune-frequency 60
```

Repeat for all currencies as needed:
`eur gbp aud nzd cad jpy chf nok sek`.

## Step 3 — Generate Stage 2 audit for one currency

### 3A) Baseline single-model audit (OLS-focused)

```bash
python -m src.stage2_ml_performance_audit eur \
    --window 50 \
    --error-sum-window 10 \
    --recenter-window 60 \
    --forward-days 10 \
    --threshold 2.0
```

### 3B) Multi-model comparison audit

```bash
python -m src.stage2_ml_performance_audit eur \
    --compare-models \
    --models ols ridge lasso elasticnet sgd xgb lgbm stacked \
    --window 50 \
    --error-sum-window 10 \
    --recenter-window 60
```

Outputs are saved under `data/audits/eur/` (same structure for each currency).

## Step 4 — Replicate full G10 comparison package

This runs the multi-model comparison across all currencies and writes aggregate ranking and SHAP summaries.

```bash
python - <<'PY'
from src.stage2_ml_performance_audit import save_stage2_g10_master_comparison

saved = save_stage2_g10_master_comparison(
    currencies=("eur","gbp","aud","nzd","cad","jpy","chf","nok","sek"),
    models=("ols","ridge","lasso","elasticnet","sgd","xgb","lgbm","stacked"),
    output_dir="data/audits",
    window=50,
    error_sum_window=10,
    recenter_window=60,
    forward_days=10,
    threshold=2.0,
)

for k, v in saved.items():
    print(f"{k}: {v}")
PY
```

Key expected aggregate files include:

- `data/audits/stage2_g10_master_model_ranking.csv`
- `data/audits/stage2_final_advanced_ml_report.csv` (if generated)
- `data/audits/stage2_g10_shap_driver_summary.csv` (if generated)
- `data/audits/stage2_g10_shap_theme_summary.csv` (if generated)
- `data/audits/stage2_policy_vs_stacked_equity_curve.csv`

---

## 6) How to run each major script/module

## Data / feature engineering modules

- `python -m src.data.load_excel_sheets` — parse workbook sheets and print shape/date ranges.
- `python -m src.data.standardize_rolling_drivers` — build standardized rolling driver map.
- `python -m src.data.build_model_ready_data` — generate and save currency master CSVs.

## Stage 1 driver discovery

- `python -m src.rolling_univariate_ols` — rolling beta/significance maps.
- `python -m src.top_drivers_history` — top drivers by significance.
- `python -m src.diversified_top_drivers_history` — category-diversified top drivers.

## Stage 2 fair value + model outputs

- `python -m src.stage2_ml_models <currency> [options]` — primary model runner CLI.
- `python -m src.stage2_fair_value_runner` — convenience OLS fair-value run + plot.

## Stage 2 auditing/reporting

- `python -m src.stage2_ml_performance_audit <currency> [options]` — baseline or multi-model audit.

## Notebook workflow (optional)

After environment activation:

```bash
jupyter notebook notebooks/
```

Suggested order:
1. `explore_dataframes.ipynb`
2. `ultimate_dataframes.ipynb`
3. `ols_regressions.ipynb`
4. `fairvalue_ols.ipynb`

---

## 7) Validation checks after a full run

Use these checks to confirm a successful replication:

```bash
# 1) Processed masters exist
python -c "from pathlib import Path; req=['eur','gbp','aud','nzd','cad','jpy','chf','nok','sek']; print(all((Path('data/processed')/f'{c}_master.csv').exists() for c in req))"

# 2) One currency audit summary exists
python -c "from pathlib import Path; print((Path('data/audits/eur/stage2_model_comparison_summary.csv')).exists())"

# 3) G10 aggregate ranking exists
python -c "from pathlib import Path; print((Path('data/audits/stage2_g10_master_model_ranking.csv')).exists())"
```

---

## 8) Operational best practices (industry standard)

- Run inside a dedicated virtual environment and lock package versions for production replication.
- Keep raw data immutable (`data/rawdata.xlsx`) and version output artifacts separately.
- Parameterize all runs (window, threshold, retune frequency) and log them with output paths.
- Treat `data/audits/` as generated artifacts; avoid manual edits.
- For team reproducibility, store command history (or a shell script) used to produce each results bundle.

---

## 9) Troubleshooting

- **`FileNotFoundError: Could not locate project root containing 'src'`**
    - Run commands from repo root or use `python -m ...` module form.

- **`rawdata.xlsx does not exist`**
    - Ensure `data/rawdata.xlsx` is present and readable.

- **LightGBM / XGBoost import issues**
    - Reinstall dependencies in a clean environment:
        `pip install --upgrade pip && pip install -r requirements.txt`

- **No signals generated in audit output**
    - Lower/adjust threshold (`--threshold`), increase sample window, or validate master CSV content.

---

## License

This project is licensed under the terms in `LICENSE`.


