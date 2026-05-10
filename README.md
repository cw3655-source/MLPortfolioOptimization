# ML Portfolio Optimizer

Machine-learning-driven portfolio optimization on the Russell 1000 cross-section.

Following the methodology of Gu, Kelly & Xiu (2020), *Empirical Asset Pricing via Machine Learning*.

---

## What it does

Forecasts next-month excess returns for individual stocks using firm characteristics + industry membership, then constructs a closed-form mean-variance tangency portfolio. The argument arc is **CAPM (1 factor) → Fama-French 5 (5 factors) → ML (~91 features, nonlinear interactions)** — each step adds capacity beyond hand-picked priced characteristics.

**End-to-end pipeline:**

1. Pull monthly returns (CRSP) + fundamentals (Compustat) via WRDS, 2000–present
2. Build 17 firm characteristics (momentum, value, profitability, etc.) + ~74 SIC2 industry dummies, all rank-transformed cross-sectionally to [-1, 1]
3. Train ElasticNet / Ridge / XGBoost; pick best via walk-forward backtest (annual refit, 10-year rolling window)
4. Forecast next-month excess return for any user-supplied portfolio
5. Estimate Σ from realized covariance of daily returns (yfinance, last 60 trading days)
6. Compute closed-form tangency weights: w* ∝ Σ⁻¹μ̂
7. Backtest strategy vs equal-weighted benchmark and SPY buy-and-hold

---

## Files

| File | Purpose |
|---|---|
| `pipeline.py` | Single-source-of-truth library: WRDS pulls, features, models, optimizer |
| `explore.ipynb` | End-to-end notebook — 11 sections from raw data to deployed artifacts |
| `app.py` | Streamlit web app with 6 tabs (Overview, Data, Models, Forecast, Optimize, Backtest) |
| `feature_dictionary.csv` | Exported feature reference table for the report |
| `cache/app_artifacts.pkl` | Trained production model + latest feature snapshot (no raw WRDS rows) |
| `requirements.txt` | Python dependencies |
| `hhaa009.pdf` | Reference paper (GKX 2020) |

---

## Local setup

```bash
python3 -m venv ~/venvs/aml
source ~/venvs/aml/bin/activate
pip install -r requirements.txt

# Optional — only needed if you want to re-pull WRDS data and retrain
pip install wrds psycopg2-binary

streamlit run app.py
```

The app loads pre-computed artifacts from `cache/app_artifacts.pkl` and serves an interactive portfolio-optimization interface at `http://localhost:8501`.

---

## Regenerating artifacts

To re-pull WRDS data, retrain models, and regenerate `cache/app_artifacts.pkl`:

1. Open `explore.ipynb` in Jupyter / VS Code with the `aml` venv as kernel
2. Run §1 → §3 (WRDS pull) → §4 → §5 → §6 (models + walk-forward) → §7 (production train) → §8 (current weights) → §9 (backtest) → §11 (export artifacts)
3. Re-run the Streamlit app

---

## Deployment

Hosted on [Streamlit Community Cloud](https://share.streamlit.io). Auto-deploys on push to `main`.

---

## Data

- **CRSP MSF + Compustat funda + CCM linktable**: monthly stock returns and annual firm fundamentals, via WRDS (Columbia subscription)
- **Ken French 5-factor returns**: free, Dartmouth (`mba.tuck.dartmouth.edu`)
- **Daily prices**: yfinance (free, real-time)

The repo does **not** include raw WRDS data files (academic licensing restrictions). It only includes the derived production artifact: trained model parameters + a cross-sectional snapshot of features at the latest date.

---

## Reference

Gu, S., B. Kelly, and D. Xiu (2020). *Empirical Asset Pricing via Machine Learning*. The Review of Financial Studies 33(5), 2223–2274.
