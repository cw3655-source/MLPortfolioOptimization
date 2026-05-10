# AML Final Project — Notes

## Project framing
ML for portfolio optimization. Argument arc:
1. CAPM is intuitive but underfits — single market factor misses many priced characteristics.
2. Fama-French (3F, 5F) adds factors motivated by empirical anomalies (size, value, profitability, investment), but is still a low-dimensional linear model with hand-picked features.
3. ML methods (Ridge, Lasso, Random Forest, possibly gradient boosting) can use many more characteristics, capture nonlinearities and interactions, and may produce better expected-return forecasts.
4. Better expected returns → better mean-variance (tangency) portfolios on the back end.

## Design decisions (locked)
- **Universe:** Russell 1000 (with survivorship-bias mitigation — historical constituents via WRDS if accessible, otherwise fixed point-in-time list with explicit caveat).
- **Prediction target:** individual stock returns, monthly horizon.
- **Model evaluation:** out-of-sample MSFE (and OOS R² vs. historical-mean benchmark) used to select the best model first.
- **Optimization step:** mean-variance / CAPM tangency portfolio in the final step, applied to a 5-stock subset for illustration.
- **Pedagogical move:** train ML models on the full Russell 1000 cross-section to get statistical power, then demonstrate the optimization pipeline cleanly on a 5-stock example portfolio.

## Open methodology questions
- Predict raw returns, excess returns over Rf, or returns in excess of cross-sectional mean each month? (Leaning toward last — strips out market-wide moves, isolates "alpha" signal that matters for portfolio choice.)
- Training window: rolling vs. expanding; window length (10 yr rolling is a reasonable default).
- Refit cadence: annual is standard.
- Covariance matrix Σ for the optimization step — sample covariance is unusable with 1000 stocks; use Ledoit-Wolf shrinkage or factor-model-implied covariance.

## Data sources
- **Stock prices:** CRSP (via WRDS / Columbia) preferred; yfinance as fallback.
- **Factor returns (Mkt-RF, SMB, HML, RMW, CMA, MOM):** Ken French data library — free, daily + monthly available. https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
- **Firm characteristics for ML features:** Compustat via WRDS preferred. yfinance/SEC EDGAR as fallback (smaller feature set).

## Reference paper
Gu, Kelly, Xiu (2020), "Empirical Asset Pricing via Machine Learning," *Review of Financial Studies*. Effectively the methodological template — ~94 firm characteristics + 8 macro vars, expanding-window training, OOS R² vs. historical mean.

## Open logistical questions
- WRDS access status (Columbia subscribes — Lehman Library).
- Project deadline / time budget — preliminary deck due Tue 2026-04-28; full project due ~2026-05-05.

## Session state (where we left off)
- Outline + 5-stock plan locked. AAPL / JPM / XOM / JNJ / PG.
- `run_analysis.py` is in this folder — pulls FF5 from Ken French, prices from yfinance, runs CAPM + FF5 per stock, writes `results.json` and 4 PNG charts. Ready to run as soon as network egress allows `mba.tuck.dartmouth.edu` and Yahoo Finance.
- Slide deck not yet built — waiting on real numbers from `run_analysis.py`.
- Next steps: run analysis → generate charts → build 8–10 slide deck (clean academic style: navy/terracotta on white, mixed real CAPM/FF5 results + ML placeholders).

