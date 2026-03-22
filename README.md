# FX Valuation: Reproducible G10 Fair-Value and ML Audit Pipeline

This repository implements a full G10 FX research pipeline:

- Stage 1 driver discovery with rolling OLS and diversified top-driver selection
- Stage 2 machine learning fair-value models across OLS, regularized linear models, gradient boosting, and a stacked ensemble
- An audit layer that scores signal quality, plots strategy equity curves, aggregates G10 model rankings, and compares the stacked ensemble against the reinforcement learning policy agent

The workflow below is written for a clean machine and is the recommended way to reproduce the final output file:

`data/audits/stage2_final_advanced_ml_report.csv`

## Repository Layout

```text
fxvaluation_d200_am3483/
|-- config.py
|-- requirements.txt
|-- setup_env.ps1
|-- data/
|   |-- rawdata.xlsx
|   |-- processed/
|   |-- audits/
|   `-- audits_test/
|-- notebooks/
`-- src/
    |-- data/
    |-- stage2_ml_models.py
    |-- stage2_ml_performance_audit.py
    |-- stage2_policy_agent.py
    `-- ...
```

## Supported Environment

- Python 3.10 or newer
- Windows, Linux, or macOS
- 8 GB RAM minimum, 16 GB recommended for full multi-currency runs

The repository is pinned to the package versions currently used in the production environment:

```text
pandas==3.0.1
numpy==2.4.2
scikit-learn==1.8.0
matplotlib==3.10.8
seaborn==0.13.2
plotly==6.6.0
statsmodels==0.14.6
xgboost==3.2.0
lightgbm==4.6.0
openpyxl==3.1.5
ipykernel==7.2.0
jupyter==1.1.1
```

## 1. Clone The Repository

```bash
gh repo clone alberto2002mp-2/fxvaluation_d200_am3483
cd fxvaluation_d200_am3483
```

## 2. Create And Activate A Virtual Environment


### Windows PowerShell

```powershell
.\setup_env.ps1
.\venv\Scripts\Activate.ps1
```

### Linux or macOS

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

## 3. Confirm The Source Data Exists

The pipeline expects the Excel workbook at `data/rawdata.xlsx`.

```bash
python -c "from pathlib import Path; p = Path('data/rawdata.xlsx'); print(p.resolve()); print('exists=', p.exists())"
```

Expected result: the resolved path prints and `exists= True`.

## 4. Build The Master Training CSVs

This produces one processed master table per currency in `data/processed/`.

```bash
python -m src.data.build_model_ready_data
```

Expected files include:

- `data/processed/eur_master.csv`
- `data/processed/gbp_master.csv`
- `data/processed/aud_master.csv`
- `data/processed/nzd_master.csv`
- `data/processed/cad_master.csv`
- `data/processed/jpy_master.csv`
- `data/processed/chf_master.csv`
- `data/processed/nok_master.csv`
- `data/processed/sek_master.csv`

## 5. Run The Stage 2 Model Suite For One Currency

This generates walk-forward model outputs in `data/model_outputs/<currency>/`.

```bash
python -m src.stage2_ml_models eur --models ols ridge lasso elasticnet sgd xgb lgbm stacked --window 250 --error-sum-window 10 --recenter-window 60 --cv-splits 4 --retune-frequency 60
```

## 6. Run The Full Stage 2 Audit For One Currency

This produces the audit datasets, comparison tables, and charts in `data/audits/<currency>/`.

```bash
python -m src.stage2_ml_performance_audit eur --compare-models --models ols ridge lasso elasticnet sgd xgb lgbm stacked --window 250 --error-sum-window 10 --recenter-window 60 --forward-days 10 --threshold 2.0
```

Key files created for one currency include:

- `data/audits/eur/stage2_model_comparison_summary.csv`
- `data/audits/eur/stage2_stacked_audit_dataset.csv`
- `data/audits/eur/stage2_policy_agent_dataset.csv` after the G10 comparison step

## 7. Run The Full G10 Package

This is the main end-to-end command for reproducing the final report and the G10 comparison package.

```powershell
@'
from src.stage2_ml_performance_audit import save_stage2_g10_master_comparison

saved = save_stage2_g10_master_comparison(
    currencies=("eur", "gbp", "aud", "nzd", "cad", "jpy", "chf", "nok", "sek"),
    models=("ols", "ridge", "lasso", "elasticnet", "sgd", "xgb", "lgbm", "stacked"),
    output_dir="data/audits",
    window=250,
    error_sum_window=10,
    recenter_window=60,
    forward_days=10,
    threshold=2.0,
    cv_splits=4,
    retune_frequency=60,
    early_stopping_rounds=25,
)

for label, path in saved.items():
    print(f"{label}: {path}")
'@ | .\venv\Scripts\python.exe -
```

The command above will:

1. Build per-currency Stage 2 comparison audits for the full G10 universe.
2. Save the G10 master ranking table.
3. Generate deterministic policy-agent outputs from the stacked ensemble audit datasets.
4. Build the final advanced ML report after the policy-agent comparison exists.
5. Save G10-average equity-curve and SHAP summary artifacts.

## 8. Final Outputs To Verify

After a successful full run, confirm these files exist:

- `data/audits/stage2_g10_master_model_ranking.csv`
- `data/audits/stage2_final_advanced_ml_report.csv`
- `data/audits/stage2_policy_vs_stacked_equity_curve.csv`
- `data/audits/stage2_g10_model_equity_curves.csv`
- `data/audits/stage2_g10_shap_driver_summary.csv`
- `data/audits/stage2_g10_shap_theme_summary.csv`

Quick validation:

```bash
python -c "from pathlib import Path; required = [Path('data/audits/stage2_g10_master_model_ranking.csv'), Path('data/audits/stage2_final_advanced_ml_report.csv'), Path('data/audits/stage2_policy_vs_stacked_equity_curve.csv')]; print(all(path.exists() for path in required))"
```

Expected result: `True`

## Optional Notebook Workflow

The notebooks are intended for narrative inspection of the same pipeline stages:

1. `notebooks/explore_dataframes.ipynb`
2. `notebooks/ultimate_dataframes.ipynb`
3. `notebooks/ols_regressions.ipynb`
4. `notebooks/fairvalue_ols.ipynb`

Launch with:

```bash
jupyter notebook notebooks/
```

## Troubleshooting

- `FileNotFoundError` for a master CSV usually means `python -m src.data.build_model_ready_data` has not been run yet.
- `rawdata.xlsx` errors mean `data/rawdata.xlsx` is missing or unreadable.
- `lightgbm` or `xgboost` import errors usually indicate the active environment is not the project virtual environment.
- The recommended invocation style is `python -m ...` from repository root. The scripts now use `pathlib` and script-relative path resolution, but running from repo root remains the cleanest operational pattern.

## License

This project is licensed under the Apache-2.0 License. See `LICENSE` for the full text.
