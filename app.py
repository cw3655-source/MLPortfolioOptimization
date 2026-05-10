"""
AML Final Project — Streamlit web app.

Single-file dashboard that loads pre-computed model artifacts from
`cache/app_artifacts.pkl` and serves an interactive portfolio-optimization
interface with six tabs.

Run:
    streamlit run app.py

Generate the artifacts file first by running §11 of explore.ipynb.
"""

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import (
    FEATURE_COLS,
    FEATURE_DICT,
    feature_dictionary_df,
    pull_yfinance_daily,
    realized_covariance,
    tangency_portfolio,
)

# =============================================================================
# Page config
# =============================================================================

st.set_page_config(
    page_title="ML Portfolio Optimizer",
    page_icon="📈",
    layout="wide",
)

# =============================================================================
# Artifact loading
# =============================================================================

ART_PATH = Path("cache/app_artifacts.pkl")


@st.cache_resource
def load_artifacts():
    if not ART_PATH.exists():
        st.error(
            f"Missing artifacts at `{ART_PATH}`. "
            "Run §11 of `explore.ipynb` to generate it."
        )
        st.stop()
    return joblib.load(ART_PATH)


art = load_artifacts()

chosen_name     = art["chosen_name"]
model_prod      = art["model_prod"]
latest_features = art["latest_features"]
ALL_COLS        = art["all_cols"]
DATA_START      = art["data_start"]
DATA_END        = art["data_end"]

last_panel_date = pd.to_datetime(latest_features["date"]).max().strftime("%Y-%m-%d")

# =============================================================================
# Header
# =============================================================================

st.title("ML Portfolio Optimizer")
st.caption(
    f"Russell 1000 monthly return forecasting → tangency portfolio · "
    f"Production model: **{chosen_name}** · "
    f"Features as-of **{last_panel_date}**"
)

# =============================================================================
# Tabs
# =============================================================================

tabs = st.tabs([
    "📋 Overview",
    "💾 Data",
    "🤖 Models",
    "🔮 Forecast",
    "⚖️ Optimize",
    "📈 Backtest",
])

# -----------------------------------------------------------------------------
# Overview
# -----------------------------------------------------------------------------
with tabs[0]:
    st.header("Machine Learning for Portfolio Optimization")
    st.markdown(
        "*An end-to-end research pipeline that compares traditional asset-pricing models with "
        "machine learning for stock-return forecasting and tangency-portfolio construction.*"
    )

    cols = st.columns(4)
    cols[0].metric("Universe", "Russell 1000")
    cols[1].metric("Total features", f"{len(ALL_COLS)}")
    cols[2].metric("Production model", chosen_name)
    years = (pd.Timestamp(DATA_END) - pd.Timestamp(DATA_START)).days // 365
    cols[3].metric("Sample period", f"{DATA_START[:4]}–{DATA_END[:4]}", f"{years}+ years")

    st.divider()

    # -------------------------------------------------------------
    st.subheader("What this project does")
    st.markdown(
        """
        Better expected-return estimates lead to better portfolio decisions — *if* those forecasts
        are combined with disciplined risk modeling and transparent out-of-sample evaluation. This
        app operationalizes that idea on the Russell 1000 cross-section:

        1. Pull monthly returns and fundamentals from CRSP/Compustat via WRDS (2000–present).
        2. Construct 17 firm characteristics + ~74 SIC2 industry dummies, all rank-transformed
           cross-sectionally to [-1, 1] each month.
        3. Train ElasticNet, Ridge, and XGBoost; pick the best via walk-forward backtest.
        4. Use the chosen model to forecast next-month excess returns for any user-supplied portfolio.
        5. Estimate Σ from realized covariance of daily returns (yfinance, last 60 trading days).
        6. Compute closed-form tangency weights $w^\\star \\propto \\hat\\Sigma^{-1} \\hat\\mu$.
        """
    )

    # -------------------------------------------------------------
    st.subheader("Why CAPM, Fama-French, and ML are all included")
    st.markdown(
        """
        Factor models and ML play **different roles**, not competing ones. CAPM provides the
        equilibrium benchmark and isolates broad market exposure. Fama-French 5-factor adds size,
        value, profitability, and investment exposures — a clearer risk decomposition. Both are
        low-dimensional and largely linear; their strength is *explanation, attribution, and risk
        modeling*, not flexible forecasting.

        ML enters as a forecasting upgrade. Following **Gu, Kelly & Xiu (2020)**, *Empirical Asset
        Pricing via Machine Learning*, we treat next-month return prediction as a high-dimensional
        forecasting problem. ML can use many firm characteristics, industry membership, and
        nonlinear interactions that hand-picked low-dim factor models can't represent. GKX find that
        on a 60-year sample of ~30,000 stocks, tree-based methods and neural nets substantially beat
        linear ML — and the source of the gain is *interactions* between characteristics.

        In this app, all three layers feed the same downstream portfolio problem. The argument arc
        is **CAPM → FF5 → ML**: each step adds capacity, the ML layer measures the *conditional
        mean* of returns more flexibly, then the tangency optimizer turns those forecasts into
        weights.
        """
    )

    # -------------------------------------------------------------
    st.subheader("Data and sample design")
    sample_years = (pd.Timestamp(DATA_END) - pd.Timestamp(DATA_START)).days // 365
    st.markdown(
        f"""
        - **Universe**: top 1,000 stocks by lagged market cap each month — a
          survivorship-bias-free academic stand-in for the Russell 1000.
        - **Sample**: {DATA_START} → {DATA_END} ({sample_years}+ years, multiple regimes
          including dotcom, GFC, COVID, ZIRP, and post-COVID rate cycle).
        - **Monthly returns**: CRSP MSF, with Shumway (1997) delisting-return adjustment.
        - **Fundamentals**: Compustat *funda*, merged via the CCM linktable, lagged 6 months to
          avoid look-ahead bias.
        - **Factor returns**: Ken French 5-factor data library (free).
        - **Daily returns** (for realized covariance + backtest): yfinance, free, near-real-time.

        The 5-stock demo portfolio used in the Backtest tab (AAPL, JPM, XOM, JNJ, PG) is a
        *pedagogical device*. The model itself is trained on the full Russell 1000 cross-section for
        statistical power; the portfolio step is demonstrated on a small subset where weights stay
        interpretable.
        """
    )

    # -------------------------------------------------------------
    st.subheader("Model evaluation logic")
    st.markdown(
        r"""
        Models are judged by **out-of-sample forecast accuracy**, not in-sample fit. Two
        complementary metrics, both following GKX convention:

        - **OOS R² with zero benchmark** —
          $R^2_{\text{oos}} = 1 - \sum(y - \hat y)^2 / \sum y^2$.
          Denominator is sum of squared excess returns *without* demeaning. For individual-stock
          prediction this is the honest benchmark — beating "predict zero" is harder than beating
          the noisy rolling historical mean. Reported numbers are small (~0.4% per month for the
          best models in GKX) but economically meaningful.
        - **OOS R² vs historical mean** — the textbook version. Included for reference; tends to
          inflate by ~3 percentage points relative to the zero-benchmark version because
          individual stocks' rolling historical means are very noisy.

        **Evaluation modes** used in this project:

        - **Single split** — train 2000–2017, validate 2018–2019, test 2020 onward. Fast, but one
          realization is luck-prone.
        - **Walk-forward backtest** — refit at the start of each test year, using the prior 10
          years as training and 1 year as validation. Yields per-year R² across multiple regimes
          plus a pooled R² as the headline. More credible than a single split.
        - **Diebold-Mariano test** — pairwise statistical test on squared-error differences,
          modified for cross-sectional dependence per GKX. Asks whether a model's accuracy
          advantage is significant.
        """
    )

    # -------------------------------------------------------------
    st.subheader("Portfolio construction")
    st.markdown(
        r"""
        The optimization layer turns forecasts into weights. Closed-form **tangency portfolio**:

        $$
        w^\star \;\propto\; \hat\Sigma^{-1} \big(\hat\mu - r_f\,\mathbf{1}\big),
        \qquad \sum_i w_i^\star = 1.
        $$

        Because the ML model predicts return *in excess of the risk-free rate* directly, $\hat\mu$
        feeds the optimizer without further adjustment.

        **Covariance estimator**: realized covariance from the past 60 trading days of daily
        returns (Andersen-Bollerslev-Diebold-Labys 2003). Non-parametric, exploits high-frequency
        information that monthly-frequency estimators discard, no GARCH assumptions, adapts to
        recent regime via the rolling window. For 5-asset portfolios this is competitive with more
        complex parametric alternatives at a fraction of the implementation effort.

        Negative weights = short positions; the closed-form tangency allows them. For a long-only
        constraint you'd swap to a `scipy.optimize`-based mean-variance solve.
        """
    )

    # -------------------------------------------------------------
    st.subheader("Limitations and interpretation")
    st.markdown(
        """
        This is a research framework, not a production trading system. A few caveats users should
        keep in mind:

        - Better OOS R² does **not** equal a guaranteed profitable strategy. Transaction costs,
          market impact, and capacity constraints are not modeled.
        - The 5-stock backtest is illustrative. Real deployments would use the full 1,000-stock
          universe, monthly rebalancing, and proper transaction-cost accounting.
        - ML improves *measurement* of expected returns; it doesn't establish economic *mechanism*.
          Factor attribution (CAPM/FF5) remains part of the interpretive workflow even when ML
          drives the allocation.
        - Hyperparameters are tuned on a validation set; overfitting risk grows when the test
          window is short. Walk-forward evaluation mitigates this but doesn't eliminate it.
        - Survivorship bias is partly mitigated by using point-in-time CRSP membership and
          delisting-return handling. But features that require long history (long-term reversal,
          rolling beta) drop observations for younger firms.
        """
    )

    st.divider()

    # -------------------------------------------------------------
    # USER MANUAL
    # -------------------------------------------------------------
    st.subheader("📖 User Manual")
    st.markdown(
        f"""
        The app has six tabs along the top. Walk through them in the natural order below.

        ##### 1. Overview *(this page)*
        Orientation only. Read once, then tab over.

        ##### 2. 💾 Data
        The **feature dictionary** documents each firm characteristic: its name, formula, source,
        economic intuition, expected sign on the predicted return, and canonical citation. Browse
        this before forecasting to know what the model is seeing.

        The **Latest feature snapshot** panel shows the cross-sectional features being fed to the
        production model right now, for every stock in the universe.

        ##### 3. 🤖 Models
        Model-comparison evidence. Tables of OOS R² (single split and walk-forward) and a per-year
        R² chart across ElasticNet, Ridge, and XGBoost. The production model used for live
        forecasting (**{chosen_name}**) was selected from these comparisons.

        ##### 4. 🔮 Forecast — *enter your portfolio here*
        1. Type tickers separated by commas (e.g., `NVDA, MSFT, JPM, KO, XOM`).
        2. Click **🔮 Forecast**.
        3. The model outputs **μ̂** for each ticker — the predicted next-month excess return over
           the risk-free rate. Positive = expected to beat T-bills, negative = expected to
           underperform.
        4. Magnitudes will be small (tens of basis points). That's normal for monthly forecasts;
           what matters for portfolio choice is the **relative ranking** across stocks, not the
           absolute level.

        ##### 5. ⚖️ Optimize — *compute weights for the portfolio you just forecasted*
        1. Click **⚖️ Optimize** (after running Forecast — it reuses your μ̂).
        2. The app pulls 6 months of daily returns from Yahoo Finance, computes a 60-day realized
           covariance Σ̂, and feeds (μ̂, Σ̂) into the closed-form tangency solver.
        3. Output is a **weight per ticker** (sums to 1), plus portfolio-level expected return,
           volatility, and Sharpe ratio (monthly and annualized).
        4. Negative weights = short positions — the unconstrained tangency allows them. For
           long-only, edit `app.py` to swap in a constrained optimizer.

        ##### 6. 📈 Backtest
        Historical performance of the strategy on the demo 5-stock portfolio (AAPL, JPM, XOM, JNJ,
        PG):

        - **Cumulative wealth chart** — {chosen_name}-driven tangency strategy vs equal-weighted
          buy-and-hold vs SPY (S&P 500).
        - **Performance summary** — total return, annualized return / volatility, Sharpe ratio, and
          max drawdown for each strategy.
        - **Drawdown chart** — how deep losses got and how long the recovery took.

        ---

        **Quick-start workflow:**

        1. Forecast tab → enter your 5–10 tickers → click Forecast
        2. Optimize tab → click Optimize → read off the weights
        3. (Optional) Models tab → confirm the production model has positive OOS R²
        4. (Optional) Backtest tab → see how the strategy performed historically

        **What this app is not:**

        - Not a brokerage. It produces weights; you'd execute trades elsewhere.
        - Not financial advice. Educational and research-oriented.
        - Not real-time. The model's features are as of the last WRDS update (typically ~2 weeks
          behind live). Forecasts are made monthly, not intraday.
        """
    )

# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
with tabs[1]:
    st.header("Feature dictionary")
    st.markdown(
        "Each firm characteristic with its formula, source, economic intuition, "
        "expected sign on the predicted return, and canonical citation."
    )
    st.dataframe(
        feature_dictionary_df(),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.subheader(f"Latest feature snapshot ({last_panel_date})")
    st.dataframe(
        latest_features.sort_values("ticker")[["ticker", "comnam", "date"] + FEATURE_COLS].head(50),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        f"Showing first 50 of {len(latest_features):,} stocks in the universe. "
        "These features feed the production model for next-month forecasting."
    )

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
with tabs[2]:
    st.header("Model comparison")
    st.markdown(
        f"Three ML methods, two evaluation modes. Final production model: **{chosen_name}**."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Single-split test")
        st.dataframe(art["comparison_single"], use_container_width=True)
        st.caption("Train: 2000–2017 · Val: 2018–2019 · Test: 2020 onward")
    with col2:
        st.subheader("Walk-forward backtest")
        st.dataframe(art["rolling_summary"], use_container_width=True)
        st.caption(
            "Annual refit, 10-year rolling window. **Pooled R²** is the "
            "GKX-style headline; **hit rate** measures robustness."
        )

    st.subheader("Per-year OOS R²")
    yearly = art["yearly_r2"]
    fig, ax = plt.subplots(figsize=(9, 4))
    yearly.plot(marker="o", ax=ax, linewidth=1.6)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("OOS R² (%)")
    ax.set_xlabel("Test year")
    ax.set_title("Walk-forward OOS R² by year, by model")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    st.pyplot(fig)

# -----------------------------------------------------------------------------
# Forecast
# -----------------------------------------------------------------------------
with tabs[3]:
    st.header("Forecast next-month excess return")
    st.markdown(
        f"Enter any portfolio of tickers. The **{chosen_name}** production model "
        f"will predict each stock's excess return over the risk-free rate for next month, "
        f"using its most recent feature snapshot ({last_panel_date})."
    )

    tickers_input = st.text_input(
        "Tickers (comma-separated):",
        value="AAPL, JPM, XOM, JNJ, PG",
        key="ticker_input",
    )

    if st.button("🔮 Forecast", type="primary", key="forecast_btn"):
        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

        sub = latest_features[latest_features["ticker"].isin(tickers)]
        present = sub["ticker"].tolist()
        missing = [t for t in tickers if t not in present]

        if missing:
            st.warning(
                f"Not in WRDS universe (or not in latest snapshot): {', '.join(missing)}"
            )

        if not present:
            st.error("None of the entered tickers were found.")
        else:
            sub_ordered = sub.set_index("ticker").reindex(present)
            X = sub_ordered[ALL_COLS].astype("float64").fillna(0).to_numpy()
            mu_hat = model_prod.predict(X)

            result = pd.DataFrame({
                "company": sub_ordered["comnam"].values,
                "as_of":   pd.to_datetime(sub_ordered["date"]).dt.strftime("%Y-%m-%d"),
                "mu_hat":  mu_hat,
            }, index=present)
            result.index.name = "ticker"

            # Stash for the Optimize tab
            st.session_state["fcast_tickers"] = present
            st.session_state["fcast_mu"]      = mu_hat
            st.session_state["fcast_table"]   = result

            st.subheader("Forecast")
            st.dataframe(
                result.style.format({"mu_hat": "{:+.4%}"}),
                use_container_width=True,
            )

            # Bar chart of μ̂
            fig, ax = plt.subplots(figsize=(9, 3.5))
            colors = ["#1E2761" if x >= 0 else "#B85042" for x in mu_hat]
            ax.bar(present, mu_hat * 100, color=colors)
            ax.axhline(0, color="black", linewidth=0.5)
            ax.set_ylabel("μ̂ (% monthly excess)")
            ax.set_title(f"Next-month forecast — {chosen_name} (excess of risk-free rate)")
            ax.grid(alpha=0.2)
            st.pyplot(fig)

            st.info(
                "Positive μ̂ → model expects to beat T-bills. Negative → expects underperformance. "
                "Magnitudes are small (tens of bps) — normal for monthly forecasts."
            )

# -----------------------------------------------------------------------------
# Optimize
# -----------------------------------------------------------------------------
with tabs[4]:
    st.header("Tangency portfolio weights")
    st.markdown(
        r"""
        Closed-form mean-variance optimal:

        $$
        w^\star \;\propto\; \hat\Sigma^{-1} \, \hat\mu, \qquad \sum_i w_i^\star = 1.
        $$

        $\hat\mu$ comes from the Forecast tab. $\hat\Sigma$ is the realized
        covariance over the past 60 trading days, pulled **live** from Yahoo Finance.
        """
    )

    if "fcast_mu" not in st.session_state:
        st.info("👈 Run the **Forecast** tab first to generate μ̂.")
    else:
        tickers = st.session_state["fcast_tickers"]
        mu_hat  = st.session_state["fcast_mu"]

        if st.button("⚖️ Optimize", type="primary", key="opt_btn"):
            with st.spinner("Pulling daily returns from Yahoo Finance..."):
                today = pd.Timestamp.today()
                start = (today - pd.DateOffset(months=6)).strftime("%Y-%m-%d")
                end   = today.strftime("%Y-%m-%d")
                try:
                    daily = pull_yfinance_daily(tickers, start, end)
                except Exception as e:
                    st.error(f"yfinance pull failed: {e}")
                    daily = pd.DataFrame()

            daily = daily.dropna(how="all")
            if len(daily) < 30:
                st.error(
                    f"Not enough daily observations ({len(daily)}). "
                    "Need at least 30 to estimate Σ."
                )
            else:
                Sigma = realized_covariance(daily, window_days=60, horizon="monthly")
                Sigma_arr = Sigma.loc[tickers, tickers].astype("float64").to_numpy()
                w = tangency_portfolio(mu_hat, Sigma_arr, rf=0.0)

                exp_ret  = float(w @ mu_hat)
                port_var = float(w @ Sigma_arr @ w)
                port_vol = float(np.sqrt(port_var))
                sharpe_m = exp_ret / port_vol if port_vol > 0 else np.nan

                col1, col2 = st.columns([3, 2])

                with col1:
                    result = pd.DataFrame({
                        "company": st.session_state["fcast_table"]["company"].values,
                        "mu_hat":  mu_hat,
                        "weight":  w,
                    }, index=tickers)
                    result.index.name = "ticker"
                    st.dataframe(
                        result.style.format({"mu_hat": "{:+.4%}", "weight": "{:+.2%}"}),
                        use_container_width=True,
                    )

                    fig, ax = plt.subplots(figsize=(8, 3.5))
                    colors = ["#1E2761" if x >= 0 else "#B85042" for x in w]
                    ax.bar(tickers, w * 100, color=colors)
                    ax.axhline(0, color="black", linewidth=0.5)
                    ax.set_ylabel("Weight (%)")
                    ax.set_title("Tangency portfolio weights")
                    ax.grid(alpha=0.2)
                    st.pyplot(fig)

                with col2:
                    st.subheader("Portfolio metrics")
                    st.metric("Expected monthly excess return", f"{exp_ret*100:+.3f}%")
                    st.metric("Expected monthly volatility",    f"{port_vol*100:.3f}%")
                    st.metric("Sharpe (monthly)",               f"{sharpe_m:.3f}")
                    st.metric("Sharpe (annualized × √12)",      f"{sharpe_m * np.sqrt(12):.3f}")
                    st.divider()
                    st.metric("Long exposure",  f"{w[w > 0].sum():+.2%}")
                    st.metric("Short exposure", f"{w[w < 0].sum():+.2%}")
                    st.caption(
                        "Negative weights = short positions (closed-form tangency allows shorts). "
                        "For long-only, swap to a constrained optimizer."
                    )

# -----------------------------------------------------------------------------
# Backtest
# -----------------------------------------------------------------------------
with tabs[5]:
    st.header("Strategy backtest vs benchmarks")

    demo = art.get("demo_portfolio", {})
    if demo:
        demo_str = ", ".join(demo.values())
        st.markdown(
            f"Walk-forward backtest of the **{chosen_name}**-driven tangency strategy "
            f"on the demo portfolio (`{demo_str}`), vs equal-weighted and SPY buy-and-hold."
        )

    bt = art.get("backtest")
    bt_stats = art.get("backtest_stats")

    if bt is None or len(bt) == 0:
        st.error("No backtest results in artifacts.")
    else:
        # Stats table
        st.subheader("Performance summary")
        if bt_stats is not None:
            st.dataframe(
                bt_stats.style.format({
                    "total_return": "{:+.2%}",
                    "ann_return":   "{:+.2%}",
                    "ann_vol":      "{:.2%}",
                    "sharpe":       "{:+.3f}",
                    "max_drawdown": "{:.2%}",
                }),
                use_container_width=True,
            )

        # Cumulative wealth
        st.subheader("Cumulative wealth ($1 starting)")
        fig, ax = plt.subplots(figsize=(10, 4.5))
        for col, label, color, ls in [
            ("strategy_total", f"Strategy ({chosen_name})", "#1E2761", "-"),
            ("equal_total",    "Equal-weighted (5 stocks)", "#2C5F2D", "--"),
            ("spy_total",      "SPY (S&P 500)",             "#B85042", "-"),
        ]:
            if col in bt.columns:
                wealth = (1 + bt[col].fillna(0)).cumprod()
                ax.plot(wealth.index, wealth.values, label=label,
                        linewidth=2, color=color, linestyle=ls)
        ax.set_ylabel("Wealth ($1)")
        ax.set_title("Walk-forward backtest cumulative wealth")
        ax.legend(frameon=False)
        ax.grid(alpha=0.3)
        st.pyplot(fig)

        # Drawdown
        st.subheader("Drawdowns")
        fig, ax = plt.subplots(figsize=(10, 3.5))
        for col, label, color, ls in [
            ("strategy_total", f"Strategy ({chosen_name})", "#1E2761", "-"),
            ("equal_total",    "Equal-weighted",            "#2C5F2D", "--"),
            ("spy_total",      "SPY",                       "#B85042", "-"),
        ]:
            if col in bt.columns:
                wealth = (1 + bt[col].fillna(0)).cumprod()
                dd = wealth / wealth.cummax() - 1
                ax.plot(dd.index, dd.values * 100, label=label,
                        linewidth=1.6, color=color, linestyle=ls)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_ylabel("Drawdown (%)")
        ax.set_title("Drawdown by month")
        ax.legend(frameon=False)
        ax.grid(alpha=0.3)
        st.pyplot(fig)
