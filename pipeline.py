"""
AML Final Project — ML for Portfolio Optimization
==================================================

Master pipeline. Single file for now; will modularize as it grows.

Pipeline stages:
    1. Data ingestion         (WRDS: CRSP + Compustat + CCM; Ken French factors)
    2. Universe construction  (Russell 1000 / top-1000 by lagged market cap)
    3. Feature engineering    (firm characteristics, cross-sectional ranks)
    4. Model training         (CAPM, FF5, Ridge, Lasso, Random Forest)
    5. OOS evaluation         (MSFE, R² vs historical-mean benchmark)
    6. Portfolio construction (mean-variance tangency on a 5-stock subset)
    7. Reporting              (charts + results.json)

Reference: Gu, Kelly, Xiu (2020), "Empirical Asset Pricing via Machine Learning,"
Review of Financial Studies 33(5).

Run:
    python pipeline.py                # full pipeline (uses cached data if present)
    python pipeline.py --no-wrds      # demo path with yfinance + FF5 only
    python pipeline.py --refresh      # ignore cache and re-pull from WRDS
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import urllib.request
import warnings
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    # ---- Sample period ----
    start_date: str = "2000-01-01"
    end_date:   str = "2024-12-31"

    # ---- Universe ----
    universe:  str  = "top1000_mcap"   # "russell1000" | "top1000_mcap"
    use_wrds:  bool = True              # False -> yfinance-only demo path

    # ---- Illustrative 5-stock subset for the portfolio step ----
    demo_tickers: list = field(default_factory=lambda: [
        "AAPL", "JPM", "XOM", "JNJ", "PG"
    ])

    # ---- Modeling ----
    # Prediction target. GKX uses excess-of-Rf; cross-sectional demean strips
    # market-wide moves and focuses on the relative-ranking signal that drives
    # portfolio choice.
    target:             str = "excess_xs_demean"   # "raw" | "excess_rf" | "excess_xs_demean"
    train_window_years: int = 10
    refit_frequency:    str = "annual"             # "annual" | "monthly"

    # ---- Paths ----
    output_dir: Path = Path(os.environ.get("OUT_DIR", "./outputs"))
    cache_dir:  Path = Path(os.environ.get("CACHE_DIR", "./cache"))

    # ---- Misc ----
    log_level: str = "INFO"
    seed:      int = 42

    # ---- WRDS auth ----
    wrds_username: str = os.environ.get("WRDS_USER", "cw3655")


CONFIG = Config()


# =============================================================================
# LOGGING
# =============================================================================

def setup_logging(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("aml_pipeline")
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        ))
        logger.addHandler(h)
    logger.setLevel(level)
    return logger


log = setup_logging(CONFIG.log_level)


# =============================================================================
# DATA LAYER — WRDS (CRSP + Compustat + CCM)
# =============================================================================

def wrds_connect(username: Optional[str] = None):
    """
    Open a WRDS connection.

    Username defaults to CONFIG.wrds_username (override via WRDS_USER env var).
    Password is read from ~/.pgpass (must be mode 0600). If ~/.pgpass is
    missing or unreadable, you'll be prompted interactively the first time
    and offered the chance to write it. Columbia accounts may require 2FA
    on the first authentication.
    """
    import wrds
    user = username or CONFIG.wrds_username
    log.info(f"Connecting to WRDS as {user}...")
    return wrds.Connection(wrds_username=user)


def verify_wrds() -> bool:
    """
    Smoke-test the WRDS connection. Confirms:
      1. The `wrds` package is installed.
      2. Authentication works (.pgpass or interactive).
      3. The CRSP and Compustat schemas are actually accessible to this account.
      4. A trivial query returns rows.

    Run this once before the first full pipeline run.
    """
    log.info("=" * 70)
    log.info("WRDS connection verification")
    log.info("=" * 70)

    # 1. Import check
    try:
        import wrds  # noqa: F401
        log.info("[ok] `wrds` package importable")
    except ImportError:
        log.error("[fail] `wrds` not installed. Run: pip install wrds psycopg2-binary")
        return False

    # 2. Connect
    try:
        conn = wrds_connect()
        log.info("[ok] connected to WRDS Cloud")
    except Exception as e:
        log.error(f"[fail] could not connect: {e}")
        log.error("       check ~/.pgpass and that your Columbia WRDS account is active")
        return False

    ok = True

    # 3. CRSP smoke test
    try:
        df = conn.raw_sql(
            "SELECT permno, date, ret "
            "FROM crsp.msf "
            "WHERE date = '2020-01-31' "
            "LIMIT 5"
        )
        log.info(f"[ok] crsp.msf accessible — sample row: {df.iloc[0].to_dict()}")
    except Exception as e:
        log.error(f"[fail] crsp.msf query: {e}")
        ok = False

    # 4. Compustat smoke test
    try:
        df = conn.raw_sql(
            "SELECT gvkey, datadate, at "
            "FROM comp.funda "
            "WHERE datadate = '2020-12-31' "
            "  AND indfmt = 'INDL' AND datafmt = 'STD' "
            "  AND popsrc = 'D' AND consol = 'C' "
            "LIMIT 5"
        )
        log.info(f"[ok] comp.funda accessible — sample row: {df.iloc[0].to_dict()}")
    except Exception as e:
        log.error(f"[fail] comp.funda query: {e}")
        ok = False

    # 5. CCM link table smoke test
    try:
        df = conn.raw_sql(
            "SELECT gvkey, lpermno, linkdt, linkenddt "
            "FROM crsp.ccmxpf_linktable "
            "LIMIT 5"
        )
        log.info(f"[ok] crsp.ccmxpf_linktable accessible — {len(df)} rows returned")
    except Exception as e:
        log.error(f"[fail] crsp.ccmxpf_linktable query: {e}")
        ok = False

    conn.close()
    log.info("=" * 70)
    log.info("WRDS verification: " + ("PASS" if ok else "FAIL — see errors above"))
    log.info("=" * 70)
    return ok


def pull_crsp_msf(conn, start: str, end: str) -> pd.DataFrame:
    """
    Pull CRSP monthly stock file, filtered to common stock (shrcd 10/11) on
    NYSE/AMEX/NASDAQ (exchcd 1/2/3). Joins delisting returns and applies the
    standard -30% replacement for performance-related delistings with missing
    dlret (Shumway 1997).
    """
    sql = f"""
        SELECT a.permno, a.date, a.ret, a.retx, a.prc, a.shrout, a.vol,
               b.shrcd, b.exchcd, b.ticker, b.comnam, b.siccd,
               c.dlret, c.dlstcd
          FROM crsp.msf  a
          JOIN crsp.msenames b
            ON a.permno = b.permno
           AND b.namedt <= a.date AND a.date <= b.nameendt
          LEFT JOIN crsp.msedelist c
            ON a.permno = c.permno
           AND date_trunc('month', a.date) = date_trunc('month', c.dlstdt)
         WHERE a.date BETWEEN '{start}' AND '{end}'
           AND b.shrcd IN (10, 11)
           AND b.exchcd IN (1, 2, 3)
    """
    log.info("Pulling CRSP MSF (this can take a few minutes)...")
    df = conn.raw_sql(sql, date_cols=["date"])

    # Delisting-adjusted return
    df["ret_adj"] = df["ret"]
    perf_missing = df["dlstcd"].between(500, 599) & df["dlret"].isna()
    df.loc[perf_missing, "ret_adj"] = -0.30

    has_dlret = df["dlret"].notna()
    df.loc[has_dlret, "ret_adj"] = (
        (1 + df.loc[has_dlret, "ret"].fillna(0))
        * (1 + df.loc[has_dlret, "dlret"]) - 1
    )

    # Market cap (shrout is in thousands, prc can be negative when bid/ask midpoint)
    df["mcap"] = df["prc"].abs() * df["shrout"] * 1_000

    log.info(
        f"  CRSP MSF: {len(df):,} rows | {df['permno'].nunique():,} permnos "
        f"| {df['date'].min().date()} -> {df['date'].max().date()}"
    )
    return df


def pull_crsp_dsf(
    conn,
    permnos: list,
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Pull daily returns from CRSP daily stock file (`crsp.dsf`) for a specific
    set of permnos over a date range.

    Used by the realized-covariance step in the tangency-portfolio optimizer.
    Daily returns include CRSP's standard adjustments. Delisting returns can
    be merged from `crsp.dsedelist` separately if needed; for short windows
    around current data this is usually unnecessary.

    Parameters
    ----------
    conn    : open WRDS connection
    permnos : list of integer permnos
    start, end : 'YYYY-MM-DD' bounds, inclusive

    Returns
    -------
    DataFrame with columns: permno, date, ret, prc, vol, shrout
    """
    if not permnos:
        raise ValueError("pull_crsp_dsf: permnos list is empty")
    permno_str = ",".join(str(int(p)) for p in permnos)
    sql = f"""
        SELECT permno, date, ret, prc, vol, shrout
          FROM crsp.dsf
         WHERE date BETWEEN '{start}' AND '{end}'
           AND permno IN ({permno_str})
    """
    log.info(f"Pulling CRSP DSF for {len(permnos)} permnos ({start} -> {end})...")
    df = conn.raw_sql(sql, date_cols=["date"])
    log.info(f"  CRSP DSF: {len(df):,} rows")
    return df


def pull_yfinance_daily(
    tickers: list,
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Pull daily simple returns from Yahoo Finance via the `yfinance` package.

    Convenient alternative to `pull_crsp_dsf` for the realized-covariance
    step in §8 / strategy backtest in §9 — no WRDS connection required,
    free, and fast for small ticker lists. Auto-adjusted prices include
    splits and dividends.

    Parameters
    ----------
    tickers : list[str] or single str
    start, end : 'YYYY-MM-DD' bounds, inclusive

    Returns
    -------
    DataFrame indexed by date (DatetimeIndex), columns are tickers,
    values are daily simple returns. Suitable input for
    `realized_covariance(...)`.
    """
    try:
        import yfinance as yf
    except ImportError as e:
        raise ImportError("yfinance not installed. Run: pip install yfinance") from e

    if isinstance(tickers, str):
        tickers = [tickers]
    log.info(f"Pulling yfinance daily for {len(tickers)} tickers ({start} -> {end})...")

    px = yf.download(
        tickers, start=start, end=end,
        interval="1d", auto_adjust=True, progress=False, threads=False,
    )

    # Extract close prices into a wide DataFrame: dates × tickers
    if isinstance(px.columns, pd.MultiIndex):
        close = px["Close"].copy()
    else:
        # Single ticker — flat column index
        close = px[["Close"]].copy()
        close.columns = [tickers[0]]

    close = close.reindex(columns=tickers)  # ensure correct column order
    rets = close.pct_change().dropna(how="all")
    rets.index.name = "date"
    log.info(f"  yfinance: {rets.shape[0]} dates × {rets.shape[1]} tickers")
    return rets


def pull_compustat_funda(conn, start: str, end: str) -> pd.DataFrame:
    """
    Annual Compustat fundamentals. Standard academic filters:
        indfmt='INDL' AND datafmt='STD' AND popsrc='D' AND consol='C'.
    """
    sql = f"""
        SELECT gvkey, datadate, fyear,
               at, lt, ceq, seq, ib, ni, oibdp, sale, revt, cogs, xsga,
               capx, dp, dvp, dvc, txdb, itcb, pstkrv, pstkl, pstk
          FROM comp.funda
         WHERE datadate BETWEEN '{start}' AND '{end}'
           AND indfmt = 'INDL' AND datafmt = 'STD'
           AND popsrc = 'D'    AND consol  = 'C'
    """
    log.info("Pulling Compustat funda...")
    df = conn.raw_sql(sql, date_cols=["datadate"])
    log.info(f"  Compustat funda: {len(df):,} rows | {df['gvkey'].nunique():,} gvkeys")
    return df


def pull_ccm_link(conn) -> pd.DataFrame:
    """CCM link table for permno <-> gvkey, primary links only."""
    sql = """
        SELECT gvkey, lpermno AS permno, linkdt, linkenddt, linktype, linkprim
          FROM crsp.ccmxpf_linktable
         WHERE linktype IN ('LU', 'LC')
           AND linkprim IN ('P',  'C')
    """
    log.info("Pulling CCM link table...")
    df = conn.raw_sql(sql, date_cols=["linkdt", "linkenddt"])
    df["linkenddt"] = df["linkenddt"].fillna(pd.Timestamp.today().normalize())
    return df


def merge_crsp_compustat(
    crsp: pd.DataFrame, comp: pd.DataFrame, link: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge CRSP returns with Compustat fundamentals via CCM. Fundamentals are
    lagged 6 months (datadate + 6mo -> 'available_at') to avoid look-ahead bias.
    """
    comp = comp.copy()
    comp["available_date"] = comp["datadate"] + pd.DateOffset(months=6)

    # Map permno -> gvkey via CCM (date-bounded)
    crsp_link = crsp.merge(link, on="permno", how="left")
    crsp_link = crsp_link[
        (crsp_link["date"] >= crsp_link["linkdt"]) &
        (crsp_link["date"] <= crsp_link["linkenddt"])
    ]

    # As-of join: latest fundamentals available at each return date.
    # NOTE: pandas.merge_asof with `by=` still requires the `on` column to
    # be globally monotonic, not just sorted within each by-group. Sort by
    # the on-key alone; pandas handles the gvkey grouping internally.
    crsp_link = crsp_link.sort_values("date").reset_index(drop=True)
    comp      = comp.sort_values("available_date").reset_index(drop=True)

    # gvkey must be the same dtype on both sides for `by` matching to work.
    crsp_link["gvkey"] = crsp_link["gvkey"].astype("string")
    comp["gvkey"]      = comp["gvkey"].astype("string")

    merged = pd.merge_asof(
        crsp_link, comp,
        left_on="date", right_on="available_date",
        by="gvkey", direction="backward",
    )
    log.info(f"  Merged panel: {len(merged):,} rows")
    return merged


# =============================================================================
# DATA LAYER — KEN FRENCH FACTORS
# =============================================================================

FF5_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_CSV.zip"
)
MOM_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Momentum_Factor_CSV.zip"
)


def _ssl_context():
    """SSL context backed by certifi's CA bundle.

    macOS python.org installs don't link system CAs by default, so urllib
    raises CERTIFICATE_VERIFY_FAILED on every HTTPS request unless we hand
    it a CA bundle explicitly.
    """
    import ssl
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def fetch_ff5() -> pd.DataFrame:
    """Fetch monthly Fama-French 5 factors + risk-free rate. Returns are decimals."""
    req = urllib.request.Request(FF5_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60, context=_ssl_context()) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
        raw = z.read(z.namelist()[0]).decode("latin-1")

    lines = raw.splitlines()
    start = end = None
    for i, ln in enumerate(lines):
        s = ln.strip()
        if start is None and s.startswith("Mkt-RF"):
            start = i + 1
        elif start is not None and (s == "" or "Annual" in s):
            end = i
            break

    rows = []
    for ln in lines[start:end or len(lines)]:
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) < 7:
            continue
        try:
            d = parts[0]
            if len(d) != 6:
                continue
            rows.append([d] + [float(x) for x in parts[1:7]])
        except ValueError:
            continue

    df = pd.DataFrame(rows, columns=["YYYYMM", "Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"])
    df["Date"] = pd.to_datetime(df["YYYYMM"], format="%Y%m") + pd.offsets.MonthEnd(0)
    return df.set_index("Date").drop(columns=["YYYYMM"]) / 100.0


def fetch_momentum() -> pd.Series:
    """Carhart momentum factor (UMD). Monthly, decimals."""
    raise NotImplementedError("TODO: parse F-F_Momentum_Factor_CSV.zip")


def merge_with_factors(
    df: pd.DataFrame,
    ff5: pd.DataFrame,
    factors: Optional[list] = None,
) -> pd.DataFrame:
    """
    Merge Ken French factor returns onto a panel that has a `date` column.

    Handles the CRSP-vs-FF5 month-end mismatch: CRSP dates are trading-day
    end-of-month (e.g., 2024-03-28 when 3/29 is Good Friday); FF5 dates are
    calendar end-of-month (2024-03-31). We roll CRSP dates forward to the
    calendar end via `pd.offsets.MonthEnd(0)` for the join.

    Parameters
    ----------
    df      : DataFrame with a `date` column (datetime64).
    ff5     : DataFrame with the FF5 factors, indexed by `Date` (calendar month-end).
    factors : list of factor column names to merge (default: all 6 FF5 fields).

    Returns
    -------
    df with the requested factor columns appended.
    """
    if factors is None:
        factors = [c for c in ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"] if c in ff5.columns]
    rhs = ff5[factors].reset_index().rename(columns={"Date": "_date_me"})
    out = df.copy()
    out["_date_me"] = out["date"] + pd.offsets.MonthEnd(0)
    out = out.merge(rhs, on="_date_me", how="left")
    return out.drop(columns=["_date_me"])


# =============================================================================
# UNIVERSE CONSTRUCTION
# =============================================================================

def build_universe(crsp: pd.DataFrame, kind: str = "top1000_mcap") -> pd.DataFrame:
    """
    Add a boolean column `in_universe` to the CRSP panel.

    'top1000_mcap': monthly top-1000 by lagged (t-1) market cap. Standard
                    academic stand-in for Russell 1000 — survivorship-bias-free.
    'russell1000':  use WRDS Russell index constituents (TODO).
    """
    if kind == "top1000_mcap":
        crsp = crsp.sort_values(["permno", "date"]).copy()
        crsp["mcap_lag1"] = crsp.groupby("permno")["mcap"].shift(1)
        crsp["mcap_rank"] = crsp.groupby("date")["mcap_lag1"].rank(
            ascending=False, method="first"
        )
        crsp["in_universe"] = crsp["mcap_rank"] <= 1000
        return crsp
    if kind == "russell1000":
        raise NotImplementedError("TODO: pull Russell historical constituents from WRDS")
    raise ValueError(f"Unknown universe: {kind}")


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

# =============================================================================
# FEATURE DICTIONARY
# =============================================================================
# Each entry documents a feature: human-readable name, formula, source,
# economic interpretation, expected coefficient sign, and canonical citation.
# Used for (a) report tables, (b) coefficient interpretation, (c) consistency
# checks (dict ordering = column ordering in the design matrix).

FEATURE_DICT: dict = {
    # ---- Returns / price-based ----
    "size": {
        "name": "Market capitalization (log)",
        "category": "price",
        "formula": "log(|prc| × shrout)",
        "source": "CRSP",
        "intuition": "Smaller firms tend to earn higher average returns (size premium).",
        "expected_sign": "−",
        "reference": "Banz (1981); Fama-French SMB (1993)",
    },
    "mom_1": {
        "name": "Short-term reversal (1-month)",
        "category": "price",
        "formula": "ret_{t}",
        "source": "CRSP",
        "intuition": "Last month's return reverses next month — liquidity/microstructure-driven.",
        "expected_sign": "−",
        "reference": "Jegadeesh (1990)",
    },
    "mom_12_2": {
        "name": "Momentum (12-2, skip-month)",
        "category": "price",
        "formula": "cumret(ret_{t-12}, …, ret_{t-2})",
        "source": "CRSP",
        "intuition": "Stocks that outperformed over past year (excluding most recent month) keep outperforming.",
        "expected_sign": "+",
        "reference": "Jegadeesh-Titman (1993); Carhart (1997)",
    },
    "mom_36_13": {
        "name": "Long-term reversal (36-13)",
        "category": "price",
        "formula": "cumret(ret_{t-36}, …, ret_{t-13})",
        "source": "CRSP",
        "intuition": "Past long-horizon winners tend to underperform — multi-year reversal.",
        "expected_sign": "−",
        "reference": "DeBondt-Thaler (1985)",
    },
    "vol_12": {
        "name": "Return volatility (12-month)",
        "category": "price",
        "formula": "std(ret_{t-12}, …, ret_{t-1})",
        "source": "CRSP",
        "intuition": "Low-volatility stocks earn higher risk-adjusted returns (low-vol anomaly).",
        "expected_sign": "−",
        "reference": "Ang, Hodrick, Xing, Zhang (2006)",
    },
    "turnover": {
        "name": "Share turnover",
        "category": "price",
        "formula": "vol_{t} / shrout_{t}",
        "source": "CRSP",
        "intuition": "Liquidity / attention proxy. Higher turnover associated with lower future returns.",
        "expected_sign": "−",
        "reference": "Datar, Naik, Radcliffe (1998)",
    },
    "mom_6_2": {
        "name": "Medium-term momentum (6-2)",
        "category": "price",
        "formula": "cumret(ret_{t-6}, …, ret_{t-2})",
        "source": "CRSP",
        "intuition": "Shorter-horizon momentum complements 12-2 momentum.",
        "expected_sign": "+",
        "reference": "Jegadeesh-Titman (1993)",
    },
    "maxret": {
        "name": "Maximum monthly return (12-mo)",
        "category": "price",
        "formula": "max(ret_{t-12}, …, ret_{t-1})",
        "source": "CRSP",
        "intuition": "Lottery-like stocks (high max return) underperform — preference-for-skewness anomaly.",
        "expected_sign": "−",
        "reference": "Bali, Cakici, Whitelaw (2011)",
    },
    "nsi": {
        "name": "Net share issuance (12-mo)",
        "category": "price",
        "formula": "log(shrout_{t} / shrout_{t-12})",
        "source": "CRSP",
        "intuition": "Firms that issue shares (positive nsi) tend to underperform — issuance anomaly.",
        "expected_sign": "−",
        "reference": "Pontiff-Woodgate (2008); Daniel-Titman (2006)",
    },
    "beta": {
        "name": "CAPM beta (36-mo rolling)",
        "category": "price",
        "formula": "Cov(r_i, Mkt-RF) / Var(Mkt-RF), 36-month rolling window",
        "source": "CRSP + Ken French",
        "intuition": "High-beta stocks have lower risk-adjusted returns than CAPM implies (low-beta anomaly).",
        "expected_sign": "−",
        "reference": "Frazzini-Pedersen (2014); Ang-Hodrick-Xing-Zhang (2006)",
    },
    "age": {
        "name": "Firm age (years on CRSP)",
        "category": "price",
        "formula": "(date − first CRSP appearance) in years",
        "source": "CRSP",
        "intuition": "Younger firms are riskier with higher expected returns; mature firms have lower returns.",
        "expected_sign": "−",
        "reference": "Pastor-Veronesi (2003)",
    },
    # ---- Fundamentals (lagged 6 months via merge_asof) ----
    "bm": {
        "name": "Book-to-market",
        "category": "fundamental",
        "formula": "ceq / mcap",
        "source": "Compustat (ceq) + CRSP (mcap)",
        "intuition": "Value premium — high B/M (value) stocks outperform low B/M (growth).",
        "expected_sign": "+",
        "reference": "Rosenberg-Reid-Lanstein (1985); Fama-French HML (1993)",
    },
    "ep": {
        "name": "Earnings yield",
        "category": "fundamental",
        "formula": "ib / mcap",
        "source": "Compustat (ib) + CRSP (mcap)",
        "intuition": "Cheap-on-earnings stocks outperform expensive ones.",
        "expected_sign": "+",
        "reference": "Basu (1977)",
    },
    "profit": {
        "name": "Operating profitability",
        "category": "fundamental",
        "formula": "oibdp / at",
        "source": "Compustat",
        "intuition": "More profitable firms earn higher returns, controlling for valuation.",
        "expected_sign": "+",
        "reference": "Novy-Marx (2013); Fama-French RMW (2015)",
    },
    "leverage": {
        "name": "Financial leverage",
        "category": "fundamental",
        "formula": "lt / at",
        "source": "Compustat",
        "intuition": "Capital structure measure. Empirical relationship to returns is mixed/weak.",
        "expected_sign": "?",
        "reference": "Bhandari (1988)",
    },
    "invest": {
        "name": "Asset growth (year-over-year)",
        "category": "fundamental",
        "formula": "Δat / at_{prev FY}",
        "source": "Compustat",
        "intuition": "High-investment firms earn lower future returns (investment factor).",
        "expected_sign": "−",
        "reference": "Cooper-Gulen-Schill (2008); Fama-French CMA (2015)",
    },
    # ---- Industry ----
    "ind_mom": {
        "name": "Industry momentum (SIC2)",
        "category": "industry",
        "formula": "Mean of mom_12_2 across firms in same SIC2 industry",
        "source": "CRSP (siccd) + derived",
        "intuition": "Industries with strong recent returns continue to outperform.",
        "expected_sign": "+",
        "reference": "Moskowitz-Grinblatt (1999)",
    },
}

# Column ordering for the design matrix — derived from dict insertion order
FEATURE_COLS = list(FEATURE_DICT.keys())


def feature_dictionary_df() -> pd.DataFrame:
    """Return FEATURE_DICT as a DataFrame for display / report tables.

    Use `.to_markdown(index=False)` for direct paste into a report,
    `.to_csv(...)` for an appendix table, or just print for inspection.
    """
    df = pd.DataFrame(FEATURE_DICT).T
    df.index.name = "feature"
    return df.reset_index()


def cross_sectional_rank(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Rank each column cross-sectionally per month and map to [-1, 1].

    Uses pct rank (0, 1], then 2x - 1 → (-1, 1]. NaNs stay NaN; the caller
    is responsible for filling them (with 0, the cross-sectional median).
    """
    out = df.copy()
    for c in cols:
        out[c] = df.groupby("date")[c].rank(pct=True) * 2 - 1
    return out


def build_features(
    panel: pd.DataFrame,
    ff5: Optional[pd.DataFrame] = None,
    with_industry: bool = True,
) -> pd.DataFrame:
    """
    Construct firm-month features from the merged CRSP/Compustat panel
    and cross-sectionally rank-transform them per GKX (2020).

    Parameters
    ----------
    panel : merged CRSP/Compustat panel from merge_crsp_compustat + build_universe.
    ff5   : Ken French 5-factor DataFrame (with "Mkt-RF" column). Required for
            the rolling-beta feature; if None, beta is set to NaN.
    with_industry : if True, append SIC2 one-hot industry dummies (columns
            named `sic_##`). These are NOT rank-transformed; they stay 0/1.

    Returns
    -------
    A copy of the panel with feature columns appended:
      - 17 firm characteristics (in FEATURE_COLS), rank-transformed to [-1, 1]
        with missing values filled with 0 (cross-sectional median).
      - ~74 SIC2 industry dummies (if with_industry=True), each in {0, 1}.
    """
    df = panel.sort_values(["permno", "date"]).copy()
    by_p = df.groupby("permno", group_keys=False)

    # ---- Price/return-based features ----
    df["size"]  = np.log(df["mcap"].replace(0, np.nan))
    df["mom_1"] = df["ret_adj"]

    # Compound-return windows (work on log returns so windowed sums compose).
    df["_log_r"] = np.log1p(df["ret_adj"].fillna(0))

    df["mom_12_2"] = np.expm1(
        by_p["_log_r"].transform(lambda s: s.shift(2).rolling(11, min_periods=11).sum())
    )
    df["mom_6_2"] = np.expm1(
        by_p["_log_r"].transform(lambda s: s.shift(2).rolling(5, min_periods=5).sum())
    )
    df["mom_36_13"] = np.expm1(
        by_p["_log_r"].transform(lambda s: s.shift(13).rolling(24, min_periods=24).sum())
    )

    df["vol_12"] = by_p["ret_adj"].transform(
        lambda s: s.rolling(12, min_periods=12).std()
    )
    df["maxret"] = by_p["ret_adj"].transform(
        lambda s: s.rolling(12, min_periods=12).max()
    )
    df["turnover"] = df["vol"] / df["shrout"].replace(0, np.nan)

    # Net share issuance: log change in shrout over 12 months
    df["_shrout_lag12"] = by_p["shrout"].transform(lambda s: s.shift(12))
    df["nsi"] = np.log(df["shrout"] / df["_shrout_lag12"].replace(0, np.nan))

    # Firm age: years since first CRSP appearance (per permno)
    first_date = by_p["date"].transform("min")
    df["age"]  = (df["date"] - first_date).dt.days / 365.25

    df = df.drop(columns=["_log_r", "_shrout_lag12"])

    # ---- Rolling 36-month CAPM beta (requires Mkt-RF from FF5) ----
    if ff5 is not None and "Mkt-RF" in ff5.columns:
        # Align CRSP trading-day end-of-month to FF5's calendar end-of-month
        ff5_to_merge = ff5[["Mkt-RF"]].reset_index().rename(columns={"Date": "_date_me"})
        df["_date_me"] = df["date"] + pd.offsets.MonthEnd(0)
        df = df.merge(ff5_to_merge, on="_date_me", how="left")
        df = df.drop(columns=["_date_me"])

        def _rolling_beta(g, window=36, min_periods=12):
            cov = g["ret_adj"].rolling(window, min_periods=min_periods).cov(g["Mkt-RF"])
            var = g["Mkt-RF"].rolling(window, min_periods=min_periods).var()
            return cov / var

        df["beta"] = (
            df.sort_values(["permno", "date"])
              .groupby("permno", group_keys=False)
              .apply(_rolling_beta)
              .reindex(df.index)
        )
        df = df.drop(columns=["Mkt-RF"])
    else:
        df["beta"] = np.nan
        log.info("  build_features: no FF5 supplied — beta will be NaN -> zero after impute")

    # ---- Fundamentals (already lagged 6mo via merge_asof's available_date) ----
    df["bm"]       = df["ceq"]   / df["mcap"]
    df["ep"]       = df["ib"]    / df["mcap"]
    df["profit"]   = df["oibdp"] / df["at"]
    df["leverage"] = df["lt"]    / df["at"]

    # ---- Asset growth (year-over-year change in `at` per gvkey) ----
    if {"gvkey", "datadate", "at"}.issubset(df.columns):
        fy = (
            df[["gvkey", "datadate", "at"]]
              .dropna()
              .drop_duplicates(["gvkey", "datadate"])
              .sort_values(["gvkey", "datadate"])
        )
        fy["invest"] = fy.groupby("gvkey")["at"].pct_change()
        df = df.merge(
            fy[["gvkey", "datadate", "invest"]],
            on=["gvkey", "datadate"], how="left",
        )

    # ---- Industry momentum (mean of mom_12_2 within SIC2 each month) ----
    if "siccd" in df.columns:
        df["sic2"] = (df["siccd"] // 100).astype("Int64")
        df["ind_mom"] = df.groupby(["date", "sic2"])["mom_12_2"].transform("mean")

    # ---- Cross-sectional rank → [-1, 1], median imputation ----
    feat = [c for c in FEATURE_COLS if c in df.columns]
    df = cross_sectional_rank(df, feat)
    df[feat] = df[feat].fillna(0.0)

    # ---- SIC2 industry one-hot dummies (NOT rank-transformed; binary 0/1) ----
    if with_industry and "sic2" in df.columns:
        dummies = pd.get_dummies(df["sic2"], prefix="sic", dtype=int)
        df = pd.concat([df, dummies], axis=1)
        log.info(f"  build_features: added {dummies.shape[1]} SIC2 industry dummies")

    log.info(f"Features built: {len(feat)} firm chars × {len(df):,} rows")
    log.info(f"  Firm-char columns: {feat}")
    return df


def get_all_feature_cols(features_df: pd.DataFrame) -> list:
    """
    Return the full design-matrix column list: firm characteristics from
    FEATURE_COLS plus any SIC2 industry dummies (columns starting with `sic_`).

    Use this anywhere you need to slice X = features_df[cols] for modeling.
    """
    cols = [c for c in FEATURE_COLS if c in features_df.columns]
    cols += sorted([c for c in features_df.columns if c.startswith("sic_")])
    return cols


# =============================================================================
# TARGET CONSTRUCTION
# =============================================================================

def make_target(
    panel: pd.DataFrame,
    kind: str = "excess_xs_demean",
    rf: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Add a `y` column to the panel: NEXT-month return in some excess form.

    kind:
      - "raw"              : next-month total return
      - "excess_rf"        : next-month return minus next-month risk-free rate
                             (requires `rf` indexed by month-end dates, e.g. ff5["RF"])
      - "excess_xs_demean" : next-month return minus its cross-sectional mean
                             (default; isolates relative-ranking signal)
    """
    df = panel.sort_values(["permno", "date"]).copy()
    df["y"] = df.groupby("permno")["ret_adj"].shift(-1)

    if kind == "raw":
        pass
    elif kind == "excess_rf":
        if rf is None:
            raise ValueError("excess_rf target requires `rf` (e.g. ff5['RF']).")
        # Normalize next-month date to CALENDAR month-end before mapping rf.
        # CRSP uses trading-day end-of-month (e.g., 2024-03-28 if 3/29 is a
        # Good Friday); FF5 uses calendar end-of-month (2024-03-31). Without
        # this alignment, .map() returns NaN for ~4 months/year, killing y.
        next_date = df.groupby("permno")["date"].shift(-1)
        next_date_me = next_date + pd.offsets.MonthEnd(0)
        df["y"] = df["y"] - next_date_me.map(rf)
    elif kind == "excess_xs_demean":
        df["y"] = df["y"] - df.groupby("date")["y"].transform("mean")
    else:
        raise ValueError(f"Unknown target kind: {kind}")
    return df


# =============================================================================
# TRAIN / VAL / TEST SPLIT
# =============================================================================

def train_val_test_split(
    df: pd.DataFrame,
    train_end: str,
    val_end: str,
    test_end: str,
    feature_cols: Optional[list] = None,
    universe_only: bool = True,
):
    """
    Split a feature+target panel into train / validation / test by date.

    All bounds are inclusive on the start side. Example:
        train_end="2018-12-31", val_end="2020-12-31", test_end="2024-12-31"
    yields:
        train: dates ≤ 2018-12-31
        val:   2019-01-01 .. 2020-12-31
        test:  2021-01-01 .. 2024-12-31

    Rows must be sorted by date for downstream TimeSeriesSplit CV to be valid.
    Returns (X_train, y_train, X_val, y_val, X_test, y_test) as numpy arrays.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    df = df.dropna(subset=["y"]).sort_values(["date", "permno"])
    if universe_only and "in_universe" in df.columns:
        df = df[df["in_universe"]]

    train = df[df["date"] <= train_end]
    val   = df[(df["date"] > train_end) & (df["date"] <= val_end)]
    test  = df[(df["date"] > val_end)   & (df["date"] <= test_end)]

    log.info(
        f"Split sizes: train={len(train):,}, val={len(val):,}, test={len(test):,} "
        f"(features={len(feature_cols)})"
    )

    return (
        train[feature_cols].values, train["y"].values,
        val[feature_cols].values,   val["y"].values,
        test[feature_cols].values,  test["y"].values,
    )


# =============================================================================
# MODELS
# =============================================================================

def fit_capm(y: pd.Series, mkt_rf: pd.Series) -> dict:
    """Single-factor CAPM regression. y and mkt_rf must be aligned and in excess form."""
    import statsmodels.api as sm
    X = sm.add_constant(mkt_rf)
    res = sm.OLS(y, X).fit()
    return {
        "alpha":     float(res.params.iloc[0]),
        "beta":      float(res.params.iloc[1]),
        "alpha_t":   float(res.tvalues.iloc[0]),
        "beta_t":    float(res.tvalues.iloc[1]),
        "r2":        float(res.rsquared),
        "r2_adj":    float(res.rsquared_adj),
        "rmse":      float(np.sqrt(res.mse_resid)),
        "n":         int(res.nobs),
    }


def fit_ff5(y: pd.Series, factors: pd.DataFrame) -> dict:
    """5-factor Fama-French regression. y in excess form."""
    import statsmodels.api as sm
    X = sm.add_constant(factors[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]])
    res = sm.OLS(y, X).fit()
    return {
        "alpha":  float(res.params.iloc[0]),
        "betas":  {k: float(v) for k, v in res.params.iloc[1:].items()},
        "r2":     float(res.rsquared),
        "r2_adj": float(res.rsquared_adj),
        "rmse":   float(np.sqrt(res.mse_resid)),
        "n":      int(res.nobs),
    }


def fit_elastic_net(
    X_train,
    y_train,
    l1_ratios=(0.1, 0.3, 0.5, 0.7, 0.9, 0.95),
    n_alphas: int = 50,
    n_splits: int = 3,
    max_iter: int = 20_000,
):
    """
    ElasticNet with TimeSeriesSplit-cross-validated alpha and l1_ratio.

    Wraps sklearn.linear_model.ElasticNetCV. The penalty is

        λ [ (1-ρ) ‖θ‖₁ + ½ ρ ‖θ‖²₂ ]

    with α=λ and l1_ratio=ρ (note sklearn uses the opposite convention from
    GKX for ρ — here, l1_ratio=1.0 is Lasso, l1_ratio→0 is Ridge).

    Inputs MUST be sorted by date (rows in chronological order). Otherwise
    TimeSeriesSplit's index-based folds won't respect time order and the
    chosen hyperparameters will be silently look-ahead biased.

    Returns the fitted ElasticNetCV. Inspect:
      .alpha_     — chosen λ
      .l1_ratio_  — chosen ρ
      .coef_      — learned weights (one per feature)
      .intercept_
    """
    from sklearn.linear_model import ElasticNetCV
    from sklearn.model_selection import TimeSeriesSplit

    tscv = TimeSeriesSplit(n_splits=n_splits)
    model = ElasticNetCV(
        l1_ratio=list(l1_ratios),
        n_alphas=n_alphas,
        cv=tscv,
        max_iter=max_iter,
        random_state=CONFIG.seed,
        n_jobs=-1,
    )
    log.info(
        f"Fitting ElasticNet on {len(X_train):,} obs × {X_train.shape[1]} features..."
    )
    model.fit(X_train, y_train)
    log.info(
        f"  Chosen: alpha={model.alpha_:.4g}, l1_ratio={model.l1_ratio_:.2f}, "
        f"nonzero coefs={int((model.coef_ != 0).sum())}"
    )
    return model


def fit_ridge(
    X_train,
    y_train,
    alphas: Optional[list] = None,
    n_splits: int = 3,
):
    """
    Ridge regression with TimeSeriesSplit-cross-validated alpha selection.

    Use this when ElasticNet's L1 component zeros out too many coefficients
    in small-sample, low-signal regimes. Ridge has no L1 thresholding —
    coefficients are shrunken proportionally toward zero but never set to
    exactly zero, so all features stay active.

    Equivalent to ElasticNetCV with l1_ratio=0 (which sklearn doesn't allow
    in ElasticNetCV directly).

    Parameters
    ----------
    X_train, y_train : training arrays (must be sorted by date for TimeSeriesSplit)
    alphas : grid of regularization strengths. Defaults to logspace(-4, 2, 20).
    n_splits : number of TimeSeriesSplit folds.

    Returns
    -------
    Fitted RidgeCV. Inspect:
      .alpha_  — chosen λ
      .coef_   — learned weights (one per feature; none will be zero)
      .intercept_
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import TimeSeriesSplit
    if alphas is None:
        alphas = np.logspace(-4, 2, 20)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    model = RidgeCV(alphas=alphas, cv=tscv)
    log.info(f"Fitting Ridge on {len(X_train):,} obs × {X_train.shape[1]} features...")
    model.fit(X_train, y_train)
    log.info(
        f"  Chosen alpha={model.alpha_:.4g}, "
        f"|coef|_max={np.abs(model.coef_).max():.4g}, "
        f"|coef|_min nonzero={np.abs(model.coef_[model.coef_ != 0]).min():.4g}"
    )
    return model


def fit_xgboost(
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    n_estimators: int = 500,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    early_stopping_rounds: int = 20,
):
    """
    Gradient-boosted regression trees (XGBoost) for return prediction.

    Best general-purpose ML method for tabular data with mixed feature scales.
    Captures nonlinear interactions that linear models (Ridge / Lasso /
    ElasticNet) miss. GKX (2020) finds tree ensembles substantially beat
    linear ML for return prediction; the source of the gain is interactions.

    With (X_val, y_val) supplied: uses early stopping based on validation
    RMSE (stops when val loss hasn't improved for `early_stopping_rounds`
    consecutive rounds). Without: trains for the full `n_estimators` rounds.

    Key hyperparameters:
      max_depth         — tree depth, controls interaction order. 3-6 typical.
      learning_rate     — shrinkage per tree. Smaller → needs more rounds but
                          generalizes better. 0.01-0.1 typical.
      n_estimators      — max number of trees. With early stopping, set high.
      subsample,
      colsample_bytree  — stochastic gradient (per-tree row/column subsampling).
                          0.5-1.0 typical; <1 is regularization.

    Inputs MUST be sorted by date (chronological) for early stopping on the
    validation set to be a fair forecast of test performance.

    Returns
    -------
    Fitted xgboost.XGBRegressor. Inspect:
      .feature_importances_  — gain-based per-feature importance
      .best_iteration        — chosen # of trees (if early-stopped)
    """
    try:
        import xgboost as xgb
    except ImportError as e:
        raise ImportError(
            "xgboost not installed. Run: pip install xgboost"
        ) from e

    kwargs = dict(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=CONFIG.seed,
        n_jobs=-1,
        tree_method="hist",
    )
    if X_val is not None and y_val is not None:
        kwargs["early_stopping_rounds"] = early_stopping_rounds
        kwargs["eval_metric"] = "rmse"
        model = xgb.XGBRegressor(**kwargs)
        log.info(
            f"Fitting XGBoost on {len(X_train):,} obs × {X_train.shape[1]} features "
            f"(max_estimators={n_estimators}, max_depth={max_depth}, "
            f"early_stopping={early_stopping_rounds})..."
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        log.info(f"  Best iteration: {model.best_iteration} of {n_estimators}")
    else:
        model = xgb.XGBRegressor(**kwargs)
        log.info(
            f"Fitting XGBoost on {len(X_train):,} obs × {X_train.shape[1]} features "
            f"(estimators={n_estimators}, no early stopping)..."
        )
        model.fit(X_train, y_train)
    return model


def fit_random_forest(X_train, y_train, **kwargs):
    """Random forest regressor for monthly return prediction."""
    raise NotImplementedError("TODO: next model — sklearn RandomForestRegressor")


# =============================================================================
# EVALUATION
# =============================================================================

def oos_r2_zero_benchmark(y_true, y_pred) -> float:
    """
    GKX (2020) out-of-sample R², equation (19):

        R²_oos = 1 − Σ(y - ŷ)² / Σ y²       (no demeaning of denominator)

    The denominator is sum of squared excess returns — NOT variance around
    a historical mean. GKX argue this is the right benchmark for individual
    stocks because the rolling historical mean is so noisy that beating it
    is too easy. Reported numbers are smaller (~0.4% per month) but
    economically meaningful. This is the headline metric for our project.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sse = ((y_true - y_pred) ** 2).sum()
    sst = (y_true ** 2).sum()
    return float(1 - sse / sst)


def oos_r2_vs_historical_mean(
    y_true: pd.Series, y_pred: pd.Series, y_hist_mean: pd.Series
) -> float:
    """
    Textbook OOS R²: 1 - SSE_model / SSE_historical_mean.

    Negative ⇒ model is worse than predicting the (rolling) historical mean.
    For individual stocks this benchmark is artificially easy — switching
    from this to oos_r2_zero_benchmark typically subtracts ~3pp from R² in
    the GKX setting. Use both, report the zero-benchmark number as primary.
    """
    sse_model = ((y_true - y_pred) ** 2).sum()
    sse_hist  = ((y_true - y_hist_mean) ** 2).sum()
    return 1 - sse_model / sse_hist


def rolling_oos_backtest(
    fwt: pd.DataFrame,
    feature_cols: list,
    fit_fn,
    train_years: int = 3,
    val_years: int = 1,
    test_start: Optional[str] = None,
    test_end: Optional[str] = None,
    universe_only: bool = True,
    refit: str = "annual",
    **fit_kwargs,
) -> pd.DataFrame:
    """
    Walk-forward annual-refit OOS backtest.

    For each test year y in [test_start, test_end]:
        train: dates spanning the `train_years` years up to year y-(val_years+1)
        val:   most recent `val_years` years before y (used for hyperparameter
               tuning / early stopping if fit_fn supports it)
        test:  all months in calendar year y

    The model is refit FROM SCRATCH at the start of each test year (annual
    refit). The fit function is called with X_train and y_train (and X_val,
    y_val if its signature accepts them — XGBoost uses these for early
    stopping; Ridge/ElasticNet ignore them).

    Parameters
    ----------
    fwt           : panel with `date`, `permno`, `ticker`, `y`, feature cols,
                    and `in_universe`. Output of make_target.
    feature_cols  : columns of fwt to use as model inputs.
    fit_fn        : a callable like fit_elastic_net / fit_ridge / fit_xgboost.
    train_years   : length of training window in calendar years.
    val_years     : length of validation window in calendar years (within or
                    adjacent to training).
    test_start    : first test year (e.g. "2023-01-01"). Default: earliest
                    feasible year given train_years + val_years.
    test_end      : last test year. Default: last year in fwt.
    refit         : "annual" supported (one refit per test year).
    universe_only : restrict to in_universe rows (default True).
    **fit_kwargs  : extra keyword args passed to fit_fn.

    Returns
    -------
    DataFrame with one row per OOS prediction:
      date | permno | ticker | y | y_pred | test_year
    Suitable for `summarize_rolling` and `diebold_mariano`.
    """
    import inspect

    if refit != "annual":
        raise NotImplementedError(f"refit={refit!r} not supported (only 'annual' for now)")

    df = fwt.dropna(subset=["y"]).copy()
    if universe_only and "in_universe" in df.columns:
        df = df[df["in_universe"]]
    df = df.sort_values(["date", "permno"]).reset_index(drop=True)

    min_year = int(df["date"].dt.year.min())
    max_year = int(df["date"].dt.year.max())
    earliest_test = min_year + train_years + val_years
    test_start = pd.Timestamp(test_start) if test_start else pd.Timestamp(f"{earliest_test}-01-01")
    test_end   = pd.Timestamp(test_end)   if test_end   else pd.Timestamp(f"{max_year}-12-31")

    test_years = list(range(test_start.year, test_end.year + 1))
    sig = inspect.signature(fit_fn)
    accepts_val = "X_val" in sig.parameters

    all_predictions = []
    log.info(f"Rolling backtest: {fit_fn.__name__} | train_years={train_years}, val_years={val_years}, "
             f"test years {test_years[0]}-{test_years[-1]} ({len(test_years)} years)")

    for y in test_years:
        tr_start = pd.Timestamp(f"{y - train_years - val_years}-01-01")
        tr_end   = pd.Timestamp(f"{y - val_years - 1}-12-31")
        va_start = pd.Timestamp(f"{y - val_years}-01-01")
        va_end   = pd.Timestamp(f"{y - 1}-12-31")
        te_start = pd.Timestamp(f"{y}-01-01")
        te_end   = pd.Timestamp(f"{y}-12-31")

        train = df[(df["date"] >= tr_start) & (df["date"] <= tr_end)]
        val   = df[(df["date"] >= va_start) & (df["date"] <= va_end)]
        test  = df[(df["date"] >= te_start) & (df["date"] <= te_end)]

        if len(train) == 0 or len(test) == 0:
            log.info(f"  Year {y}: skipping (train={len(train)}, test={len(test)})")
            continue

        X_tr, y_tr = train[feature_cols].values, train["y"].values
        X_te       = test[feature_cols].values

        if accepts_val and len(val) > 0:
            X_va, y_va = val[feature_cols].values, val["y"].values
            model = fit_fn(X_tr, y_tr, X_val=X_va, y_val=y_va, **fit_kwargs)
        else:
            model = fit_fn(X_tr, y_tr, **fit_kwargs)

        y_pred = model.predict(X_te)

        pred = test[["date", "permno", "ticker", "y"]].copy()
        pred["y_pred"] = y_pred
        pred["test_year"] = y
        all_predictions.append(pred)

        # Per-year R² for progress visibility
        r2_y = oos_r2_zero_benchmark(test["y"].values, y_pred)
        log.info(f"  Year {y}: train={len(train):,}, val={len(val):,}, test={len(test):,}, "
                 f"OOS R²={r2_y*100:+.3f}%")

    if not all_predictions:
        raise RuntimeError(
            "Rolling backtest produced no predictions. "
            "Check that train_years + val_years < total years in fwt."
        )

    return pd.concat(all_predictions, ignore_index=True)


def summarize_rolling(results_df: pd.DataFrame) -> dict:
    """
    Aggregate per-year predictions from `rolling_oos_backtest` into headline
    statistics: pooled R², per-year R², hit rate.

    Parameters
    ----------
    results_df : output of rolling_oos_backtest. Must have columns y, y_pred, test_year.

    Returns
    -------
    dict with:
      pooled_r2_zero  — R² over all (date, permno) at once. The headline number.
      yearly_r2       — Series indexed by test_year.
      hit_rate        — fraction of test years with R² > 0.
      n_test_years    — number of test years.
      n_observations  — total OOS predictions.
    """
    pooled = oos_r2_zero_benchmark(results_df["y"].values, results_df["y_pred"].values)
    yearly = (
        results_df.groupby("test_year")
                  .apply(lambda d: oos_r2_zero_benchmark(d["y"].values, d["y_pred"].values))
                  .rename("oos_r2")
    )
    return {
        "pooled_r2_zero": float(pooled),
        "yearly_r2":      yearly,
        "hit_rate":       float((yearly > 0).mean()),
        "n_test_years":   int(len(yearly)),
        "n_observations": int(len(results_df)),
    }


def diebold_mariano(y_true, y_pred_a, y_pred_b) -> dict:
    """
    Diebold-Mariano test for equal forecast accuracy.

    H0: model A and model B have equal squared-error performance.
    Negative t-stat ⇒ A's squared errors are smaller on average ⇒ A wins.

    Parameters
    ----------
    y_true      : array of realizations
    y_pred_a, y_pred_b : arrays of predictions from two models

    Returns
    -------
    dict with t_stat, p_value, and mean squared-error difference.
    """
    y_true   = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)
    d = (y_pred_a - y_true) ** 2 - (y_pred_b - y_true) ** 2
    n = len(d)
    if n < 2:
        return {"t_stat": float("nan"), "p_value": float("nan"), "mean_sse_diff": float("nan"), "n": n}
    mean_d = d.mean()
    se = d.std(ddof=1) / np.sqrt(n)
    t_stat = mean_d / se if se > 0 else float("nan")
    # Two-sided p-value using normal approximation
    from math import erf, sqrt
    if not np.isnan(t_stat):
        p_value = 2 * (1 - 0.5 * (1 + erf(abs(t_stat) / sqrt(2))))
    else:
        p_value = float("nan")
    return {
        "t_stat":        float(t_stat),
        "p_value":       float(p_value),
        "mean_sse_diff": float(mean_d),
        "n":             n,
    }


# =============================================================================
# PORTFOLIO CONSTRUCTION
# =============================================================================

def estimate_covariance(returns: pd.DataFrame, method: str = "ledoit_wolf") -> np.ndarray:
    """Shrinkage-aware covariance estimator."""
    if method == "ledoit_wolf":
        from sklearn.covariance import LedoitWolf
        return LedoitWolf().fit(returns.dropna()).covariance_
    if method == "sample":
        return returns.cov().values
    raise ValueError(f"Unknown covariance method: {method}")


def realized_covariance(
    daily_returns: pd.DataFrame,
    window_days: int = 60,
    horizon: str = "monthly",
) -> pd.DataFrame:
    """
    Realized covariance matrix from daily returns over a recent window.

    Following Andersen, Bollerslev, Diebold, Labys (2003): a non-parametric
    estimator that exploits high-frequency information by summing daily
    contributions instead of squaring a single low-frequency return.

    Parameters
    ----------
    daily_returns : DataFrame indexed by date, columns are tickers (or any
                    asset identifier). Values are daily simple returns.
    window_days   : lookback in trading days. Common choices: 21 (1 mo),
                    63 (3 mo), 126 (6 mo). Shorter = more responsive but noisier.
    horizon       : "daily" | "monthly" | "annual" — scales the daily covariance
                    by 1, 21, or 252 to express it at the requested horizon
                    under an i.i.d. daily-returns assumption.

    Returns
    -------
    NxN DataFrame with the same columns/index as the input, holding the
    estimated covariance matrix at the requested horizon.

    Notes
    -----
    Uses pandas .cov() which computes the standard sample covariance with
    pairwise complete observations. For the canonical realized-covariance
    formulation (uncentered sum of outer products), the difference is
    negligible at daily frequency (mean daily return ≈ 0).
    """
    recent = daily_returns.tail(window_days).dropna(how="all")
    if len(recent) < 2:
        raise ValueError(
            f"Need at least 2 daily observations for covariance; got {len(recent)}"
        )
    cov_daily = recent.cov()
    scales = {"daily": 1, "monthly": 21, "annual": 252}
    if horizon not in scales:
        raise ValueError(f"horizon must be one of {list(scales)}; got {horizon!r}")
    log.info(
        f"Realized covariance: {len(recent)} daily obs × {cov_daily.shape[0]} assets "
        f"| horizon={horizon}"
    )
    return cov_daily * scales[horizon]


def tangency_portfolio(mu: np.ndarray, Sigma: np.ndarray, rf: float = 0.0) -> np.ndarray:
    """
    Closed-form tangency portfolio weights:
        w* ∝ Σ⁻¹ (μ − rf·1),   normalized to sum to 1.
    Allows shorts. Add no-short / leverage caps via scipy.optimize if needed.
    """
    excess = mu - rf
    inv = np.linalg.solve(Sigma, excess)
    return inv / inv.sum()


# =============================================================================
# REPORTING
# =============================================================================

def save_results(results: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"  Wrote {path}")


def inspect_panel(cfg: Config = CONFIG) -> None:
    """Print a quick summary of the cached merged panel."""
    panel_path = cfg.cache_dir / "panel.parquet"
    if not panel_path.exists():
        log.error(f"No cached panel at {panel_path}. Run the pipeline first.")
        return

    panel = pd.read_parquet(panel_path)
    log.info("=" * 70)
    log.info("Panel summary")
    log.info("=" * 70)
    log.info(f"Shape:            {panel.shape[0]:,} rows × {panel.shape[1]} cols")
    log.info(f"Date range:       {panel['date'].min().date()} -> {panel['date'].max().date()}")
    log.info(f"Unique months:    {panel['date'].nunique():,}")
    log.info(f"Unique permnos:   {panel['permno'].nunique():,}")
    if "in_universe" in panel.columns:
        n_uni = panel.loc[panel["in_universe"], "permno"].nunique()
        log.info(f"In-universe permnos: {n_uni:,}")

    log.info("\nColumns:")
    for c in panel.columns:
        dtype = str(panel[c].dtype)
        miss  = panel[c].isna().mean()
        log.info(f"  {c:<22}  {dtype:<14}  {miss:>6.1%} missing")

    log.info("\nFirst 5 rows (subset):")
    show_cols = [c for c in
                 ["permno","date","ticker","ret","ret_adj","mcap","ceq","at","in_universe"]
                 if c in panel.columns]
    log.info("\n" + panel[show_cols].head().to_string(index=False))


# =============================================================================
# MAIN
# =============================================================================

def main(cfg: Config = CONFIG, refresh: bool = False) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 70)
    log.info(f"AML pipeline | {cfg.start_date} -> {cfg.end_date}")
    log.info(f"Universe: {cfg.universe} | Target: {cfg.target} | WRDS: {cfg.use_wrds}")
    log.info("=" * 70)

    # --- 1. Factors ---
    ff5 = fetch_ff5()
    log.info(f"FF5: {ff5.index.min().date()} -> {ff5.index.max().date()} | n={len(ff5)}")

    # --- 2. Returns + characteristics panel ---
    panel_path = cfg.cache_dir / "panel.parquet"
    if cfg.use_wrds:
        if panel_path.exists() and not refresh:
            log.info(f"Loading cached panel from {panel_path}")
            panel = pd.read_parquet(panel_path)
        else:
            conn  = wrds_connect()
            crsp  = pull_crsp_msf(conn, cfg.start_date, cfg.end_date)
            comp  = pull_compustat_funda(conn, cfg.start_date, cfg.end_date)
            link  = pull_ccm_link(conn)
            panel = merge_crsp_compustat(crsp, comp, link)
            panel = build_universe(panel, kind=cfg.universe)
            panel.to_parquet(panel_path)
            log.info(f"Panel cached -> {panel_path}")
    else:
        log.warning("use_wrds=False — falling back to demo path")
        panel = None  # demo path uses ff5 + yfinance only

    # --- 3. Feature engineering ---
    # features = build_features(panel)

    # --- 4. Models (train) ---
    # results_capm  = ...
    # results_ff5   = ...
    # results_ridge = ...
    # results_lasso = ...
    # results_rf    = ...

    # --- 5. OOS evaluation ---
    # rolling_oos_backtest(panel, fit_ridge, ...)

    # --- 6. Portfolio construction (5-stock illustration) ---
    # mu_hat   = best_model.predict(features.loc[demo_tickers])
    # Sigma    = estimate_covariance(returns[demo_tickers])
    # w_star   = tangency_portfolio(mu_hat, Sigma, rf=ff5["RF"].iloc[-1])

    # --- 7. Report ---
    # save_results(all_results, cfg.output_dir / "results.json")

    log.info("Pipeline finished (stubs not yet wired).")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AML Final Project pipeline.")
    p.add_argument("--verify-wrds", action="store_true",
                   help="Run WRDS connection smoke tests and exit.")
    p.add_argument("--inspect", action="store_true",
                   help="Print summary of the cached panel and exit.")
    p.add_argument("--no-wrds", action="store_true",
                   help="Skip WRDS — demo path with yfinance + FF5 only.")
    p.add_argument("--refresh", action="store_true",
                   help="Ignore parquet cache; re-pull from WRDS.")
    p.add_argument("--start", type=str, default=None, help="Override start date (YYYY-MM-DD).")
    p.add_argument("--end",   type=str, default=None, help="Override end date (YYYY-MM-DD).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.verify_wrds:
        sys.exit(0 if verify_wrds() else 1)
    if args.inspect:
        inspect_panel(CONFIG)
        sys.exit(0)
    if args.no_wrds:  CONFIG.use_wrds   = False
    if args.start:    CONFIG.start_date = args.start
    if args.end:      CONFIG.end_date   = args.end
    main(CONFIG, refresh=args.refresh)
