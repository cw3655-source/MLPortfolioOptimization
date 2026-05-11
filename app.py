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
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
    yearly_long = (
        yearly.reset_index()
              .melt(id_vars=yearly.index.name or "test_year", var_name="model", value_name="oos_r2")
    )
    fig = px.line(
        yearly_long,
        x=yearly.index.name or "test_year", y="oos_r2",
        color="model", markers=True,
        title="Walk-forward OOS R² by year, by model",
        labels={"oos_r2": "OOS R² (%)", "test_year": "Test year"},
    )
    fig.add_hline(y=0, line_color="black", line_width=0.5)
    fig.update_layout(hovermode="x unified", legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)

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

            # Interactive bar chart of μ̂
            colors = ["#1E2761" if x >= 0 else "#B85042" for x in mu_hat]
            fig = go.Figure(
                go.Bar(
                    x=present,
                    y=(mu_hat * 100).tolist(),
                    marker_color=colors,
                    text=[f"{v*100:+.3f}%" for v in mu_hat],
                    hovertemplate="<b>%{x}</b><br>μ̂ = %{y:+.4f}%<extra></extra>",
                )
            )
            fig.add_hline(y=0, line_color="black", line_width=0.5)
            fig.update_layout(
                yaxis_title="μ̂ (% monthly excess)",
                title=f"Next-month forecast — {chosen_name} (excess of risk-free rate)",
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

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

        col_constraint, col_btn = st.columns([2, 1])
        with col_constraint:
            constraint = st.radio(
                "Portfolio constraint",
                options=["Long-short (allows negative weights)", "Long-only (no shorts)"],
                horizontal=True,
                key="opt_constraint",
            )
        opt_long_only = constraint.startswith("Long-only")
        # Persist for the Backtest tab
        st.session_state["opt_long_only"] = opt_long_only

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
                w = tangency_portfolio(mu_hat, Sigma_arr, rf=0.0, long_only=opt_long_only)

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

                    colors = ["#1E2761" if x >= 0 else "#B85042" for x in w]
                    fig = go.Figure(
                        go.Bar(
                            x=tickers,
                            y=(w * 100).tolist(),
                            marker_color=colors,
                            text=[f"{v*100:+.2f}%" for v in w],
                            hovertemplate="<b>%{x}</b><br>weight = %{y:+.2f}%<extra></extra>",
                        )
                    )
                    fig.add_hline(y=0, line_color="black", line_width=0.5)
                    fig.update_layout(
                        yaxis_title="Weight (%)",
                        title="Tangency portfolio weights",
                        showlegend=False,
                    )
                    st.plotly_chart(fig, use_container_width=True)

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
    st.header("Backtest your portfolio")
    st.markdown(
        "Walk-forward backtest of the ML-driven tangency strategy on **your portfolio** "
        "(from the Forecast tab), compared against equal-weighted buy-and-hold and SPY. "
        "Each month: ML predicts μ̂, daily returns from yfinance estimate Σ, tangency optimizer "
        "produces weights, weights are applied to realized next-month returns."
    )

    if "fcast_tickers" not in st.session_state:
        st.info("👈 Run the **Forecast** tab first to set your portfolio.")
    else:
        bt_tickers = st.session_state["fcast_tickers"]
        st.markdown(f"**Portfolio:** {', '.join(bt_tickers)} ({len(bt_tickers)} stocks)")

        # Long-only vs long-short toggle — defaults to whatever the user picked in Optimize
        col_constraint, col_btn = st.columns([2, 1])
        with col_constraint:
            default_idx = 1 if st.session_state.get("opt_long_only", False) else 0
            bt_constraint = st.radio(
                "Portfolio constraint",
                options=["Long-short", "Long-only"],
                horizontal=True,
                index=default_idx,
                key="bt_constraint",
            )
        bt_long_only = bt_constraint == "Long-only"

        with st.expander("📘 Methodology — what each benchmark measures", expanded=False):
            st.markdown(
                """
                The backtest compares your **ML tangency strategy** against four other approaches
                run on the *same portfolio*. All use the same realized covariance Σ̂ (60-day window
                of daily returns) and the same optimizer; what differs is the expected-return input.

                | Strategy | $\\hat\\mu$ at each rebalance | What it tests |
                |---|---|---|
                | **ML tangency** *(yours)* | Production model forecast | The protagonist |
                | **Historical-mean MVO** | Sample mean of past **36 monthly returns** | Does ML beat the textbook "use past returns as forecast"? |
                | **Equal-weighted** | None — $w_i = 1/N$ | Baseline: does *any* optimization help vs naive 1/N? |
                | **SPY (S&P 500)** | Passive market portfolio | Has the strategy at least beaten the index? |

                **Historical-mean window = 36 months (3 years).** Standard academic choice — long
                enough for the sample mean to be reasonably stable, short enough to adapt to regime
                change. Same window Goyal-Welch use for their equity-premium predictability tests
                and what most CFA-curriculum mean-variance examples assume. With <12 months of
                history available the row is skipped.

                **The most informative comparison: ML tangency vs Historical-mean MVO.** Identical
                infrastructure, identical Σ̂, identical constraint set — only $\\hat\\mu$ differs.
                The wedge between those two lines isolates the economic value of the ML forecast.
                """
            )

        run_bt = st.button("📈 Run backtest", type="primary", key="bt_btn")

        # Show pre-computed demo backtest when nothing's been run yet
        if not run_bt:
            demo = art.get("demo_portfolio", {})
            demo_bt = art.get("backtest")
            demo_stats = art.get("backtest_stats")
            if demo and demo_bt is not None:
                with st.expander(
                    f"📊 Pre-computed reference backtest — demo portfolio "
                    f"({', '.join(demo.values())})",
                    expanded=False,
                ):
                    st.markdown(
                        "This is the static reference backtest from the notebook — long-short, "
                        "5-stock demo. Click **Run backtest** above to backtest *your* portfolio instead."
                    )
                    if demo_stats is not None:
                        st.dataframe(
                            demo_stats.style.format({
                                "total_return": "{:+.2%}",
                                "ann_return":   "{:+.2%}",
                                "ann_vol":      "{:.2%}",
                                "sharpe":       "{:+.3f}",
                                "max_drawdown": "{:.2%}",
                            }),
                            use_container_width=True,
                        )
        else:
            # ---- Dynamic backtest on user portfolio ----
            preds = art.get("rolling_predictions")
            if preds is None:
                st.error(
                    "Rolling predictions not found in artifacts. "
                    "Re-run §11 of the notebook to regenerate."
                )
                st.stop()

            user_preds = preds[preds["ticker"].isin(bt_tickers)].copy()
            available = sorted(user_preds["ticker"].unique())
            missing = [t for t in bt_tickers if t not in available]
            if missing:
                st.warning(
                    f"No historical predictions for: {', '.join(missing)} "
                    "(not in WRDS universe during test window). Continuing with the rest."
                )
                bt_tickers = available

            if len(bt_tickers) < 2:
                st.error("Need at least 2 stocks with historical predictions to backtest.")
                st.stop()

            # Pull daily total returns for tickers + SPY
            first_date = pd.Timestamp(user_preds["date"].min())
            last_date  = pd.Timestamp(user_preds["date"].max())
            # 42-month buffer covers both the 60-day Σ window and the 36-month historical-mean window
            buf_start  = (first_date - pd.DateOffset(months=42)).strftime("%Y-%m-%d")
            buf_end    = (last_date + pd.DateOffset(months=2)).strftime("%Y-%m-%d")

            with st.spinner(
                f"Pulling daily returns for {len(bt_tickers)+1} tickers from yfinance "
                f"({buf_start} → {buf_end})..."
            ):
                try:
                    daily = pull_yfinance_daily(bt_tickers + ["SPY"], buf_start, buf_end)
                except Exception as e:
                    st.error(f"yfinance pull failed: {e}")
                    st.stop()

            daily = daily.dropna(how="all")
            monthly = (1 + daily).resample("ME").prod() - 1

            # Walk-forward loop
            records = []
            weights_history = []
            eq_weight = np.ones(len(bt_tickers)) / len(bt_tickers)

            progress = st.progress(0.0, text="Running backtest...")
            rebalance_dates = sorted(user_preds["date"].unique())
            for i, rebalance_dt in enumerate(rebalance_dates):
                progress.progress((i + 1) / len(rebalance_dates), text=f"Month {i+1}/{len(rebalance_dates)}")

                sub = user_preds[user_preds["date"] == rebalance_dt].set_index("ticker")
                if not all(t in sub.index for t in bt_tickers):
                    continue
                mu_ml = sub.loc[bt_tickers, "y_pred"].astype("float64").to_numpy()

                rebalance_me = pd.Timestamp(rebalance_dt) + pd.offsets.MonthEnd(0)

                # Σ̂ from past 60 trading days of daily returns
                dw = daily.loc[:rebalance_me, bt_tickers].dropna().tail(60)
                if len(dw) < 30:
                    continue
                Sigma_d = dw.cov().to_numpy() * 21

                # μ̂ for Historical-mean MVO: past 36 monthly returns (sample mean)
                hist_window = monthly.loc[:rebalance_me, bt_tickers].dropna().tail(36)
                mu_hist = hist_window.mean().to_numpy() if len(hist_window) >= 12 else None

                try:
                    w_ml   = tangency_portfolio(mu_ml, Sigma_d, rf=0.0, long_only=bt_long_only)
                    w_hist = (tangency_portfolio(mu_hist, Sigma_d, rf=0.0, long_only=bt_long_only)
                              if mu_hist is not None else None)
                except Exception:
                    continue

                next_me = rebalance_me + pd.offsets.MonthEnd(1)
                if next_me not in monthly.index:
                    continue
                rt     = monthly.loc[next_me, bt_tickers].astype("float64").to_numpy()
                spy_rt = monthly.loc[next_me, "SPY"]
                if np.any(pd.isna(rt)) or pd.isna(spy_rt):
                    continue

                records.append({
                    "month":          next_me,
                    "strategy_total": float(w_ml   @ rt),
                    "histmvo_total":  float(w_hist @ rt) if w_hist is not None else np.nan,
                    "equal_total":    float(eq_weight @ rt),
                    "spy_total":      float(spy_rt),
                })
                weights_history.append(pd.Series(w_ml, index=bt_tickers, name=rebalance_dt))

            progress.empty()

            if not records:
                st.error("Backtest produced no valid months — check ticker availability.")
                st.stop()

            bt = pd.DataFrame(records).set_index("month").sort_index()
            weights = pd.DataFrame(weights_history)

            # Performance stats
            def perf_stats(r, label):
                r = r.dropna()
                n = len(r)
                if n < 2:
                    return {"label": label, "n_months": n,
                            "total_return": np.nan, "ann_return": np.nan,
                            "ann_vol": np.nan, "sharpe": np.nan, "max_drawdown": np.nan}
                total_ret = (1 + r).prod() - 1
                ann_ret   = (1 + r).prod() ** (12 / n) - 1
                ann_vol   = r.std() * np.sqrt(12)
                sharpe    = ann_ret / ann_vol if ann_vol > 0 else np.nan
                cum       = (1 + r).cumprod()
                max_dd    = (cum / cum.cummax() - 1).min()
                return {"label": label, "n_months": n,
                        "total_return": total_ret, "ann_return": ann_ret,
                        "ann_vol": ann_vol, "sharpe": sharpe, "max_drawdown": max_dd}

            stats_tbl = pd.DataFrame([
                perf_stats(bt["strategy_total"], f"{chosen_name} tangency ({bt_constraint})"),
                perf_stats(bt["histmvo_total"],  f"Historical-mean MVO ({bt_constraint})"),
                perf_stats(bt["equal_total"],    "Equal-weighted"),
                perf_stats(bt["spy_total"],      "SPY"),
            ]).set_index("label")

            st.subheader("Performance summary")
            st.dataframe(
                stats_tbl.style.format({
                    "total_return": "{:+.2%}",
                    "ann_return":   "{:+.2%}",
                    "ann_vol":      "{:.2%}",
                    "sharpe":       "{:+.3f}",
                    "max_drawdown": "{:.2%}",
                }),
                use_container_width=True,
            )

            # Cumulative wealth (4 lines: ML strategy, Historical-mean MVO, equal, SPY)
            st.subheader("Cumulative wealth ($1 starting)")
            strat_w = (1 + bt["strategy_total"]).cumprod()
            hist_w  = (1 + bt["histmvo_total"].fillna(0)).cumprod()
            eq_w    = (1 + bt["equal_total"]).cumprod()
            spy_w   = (1 + bt["spy_total"]).cumprod()

            series_list = [
                (strat_w, f"Strategy ({chosen_name}, {bt_constraint})", "#1E2761", "solid"),
                (hist_w,  f"Historical-mean MVO ({bt_constraint})",     "#E8833D", "solid"),
                (eq_w,    "Equal-weighted",                             "#2C5F2D", "dash"),
                (spy_w,   "SPY (S&P 500)",                              "#B85042", "solid"),
            ]

            fig = go.Figure()
            for series, label, color, dash in series_list:
                fig.add_trace(go.Scatter(
                    x=series.index, y=series.values,
                    name=label, mode="lines",
                    line=dict(color=color, dash=dash, width=2),
                    hovertemplate="<b>" + label + "</b><br>%{x|%Y-%m}: $%{y:.3f}<extra></extra>",
                ))
            fig.update_layout(
                title=f"Walk-forward backtest — {len(bt_tickers)}-stock portfolio",
                yaxis_title="Wealth ($1)",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
            st.plotly_chart(fig, use_container_width=True)

            st.caption(
                "**The wedge between the navy (ML) and orange (Historical-mean MVO) lines is the "
                "economic value of the ML forecast** — both use the same Σ̂ and same constraint, "
                "only μ̂ differs."
            )

            # Drawdown
            st.subheader("Drawdowns")
            fig = go.Figure()
            for series, label, color, dash in series_list:
                dd = (series / series.cummax() - 1) * 100
                fig.add_trace(go.Scatter(
                    x=dd.index, y=dd.values,
                    name=label, mode="lines",
                    line=dict(color=color, dash=dash, width=1.6),
                    hovertemplate="<b>" + label + "</b><br>%{x|%Y-%m}: %{y:.2f}%<extra></extra>",
                ))
            fig.add_hline(y=0, line_color="black", line_width=0.5)
            fig.update_layout(
                title="Drawdown by month",
                yaxis_title="Drawdown (%)",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Weights drift
            st.subheader("Tangency weights over time")
            weights_long = (
                weights.reset_index()
                       .rename(columns={"index": "rebalance"})
                       .melt(id_vars="rebalance", var_name="ticker", value_name="weight")
            )
            fig = px.line(
                weights_long,
                x="rebalance", y="weight", color="ticker",
                title=f"Weights at each rebalance ({bt_constraint})",
                labels={"weight": "Weight", "rebalance": "Rebalance date"},
            )
            fig.add_hline(y=0, line_color="black", line_width=0.5)
            fig.update_layout(hovermode="x unified", legend_title_text="")
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                f"Weights summary (across {len(weights)} rebalances): "
                f"max long = {weights.max().max():.2%}, "
                f"max short = {weights.min().min():.2%}, "
                f"mean turnover ≈ {weights.diff().abs().sum(axis=1).mean():.2%}/month"
            )
