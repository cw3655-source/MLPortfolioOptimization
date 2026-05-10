"""
AML Final Project - Preliminary CAPM and FF5 analysis
5 stocks: AAPL, JPM, XOM, JNJ, PG
Period: 2010-01 to 2024-12 (monthly)
"""
import io
import json
import zipfile
import urllib.request
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

import os
OUT_DIR = os.environ.get("OUT_DIR", "/sessions/youthful-gallant-thompson/mnt/outputs")
os.makedirs(OUT_DIR, exist_ok=True)

TICKERS = ["AAPL", "JPM", "XOM", "JNJ", "PG"]
SECTORS = {
    "AAPL": "Technology",
    "JPM": "Financials",
    "XOM": "Energy",
    "JNJ": "Healthcare",
    "PG": "Consumer Staples",
}
START = "2010-01-01"
END   = "2024-12-31"

# ---------- 1. Fama-French 5 factors (monthly) ----------
FF5_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"

def fetch_ff5():
    req = urllib.request.Request(FF5_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
        name = z.namelist()[0]
        raw = z.read(name).decode("latin-1")
    # Parse: skip header lines, stop before annual section
    lines = raw.splitlines()
    start = None
    end = None
    for i, ln in enumerate(lines):
        s = ln.strip()
        if start is None and s.startswith("Mkt-RF"):
            start = i + 1
        elif start is not None and (s == "" or "Annual" in s):
            end = i
            break
    if end is None:
        end = len(lines)
    rows = []
    for ln in lines[start:end]:
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) < 7: continue
        try:
            d = parts[0]
            if len(d) != 6: continue
            rows.append([d] + [float(x) for x in parts[1:7]])
        except ValueError:
            continue
    df = pd.DataFrame(rows, columns=["YYYYMM","Mkt-RF","SMB","HML","RMW","CMA","RF"])
    df["Date"] = pd.to_datetime(df["YYYYMM"], format="%Y%m") + pd.offsets.MonthEnd(0)
    df = df.set_index("Date").drop(columns=["YYYYMM"])
    # Convert from percent to decimal
    df = df / 100.0
    return df

print("Fetching Fama-French 5 factors...")
ff5 = fetch_ff5()
print(f"  FF5 sample: {ff5.index.min().date()} to {ff5.index.max().date()}, n={len(ff5)}")

# ---------- 2. Stock prices from yfinance ----------
print(f"Downloading prices for {TICKERS}...")
px = yf.download(TICKERS, start=START, end=END, interval="1mo",
                 auto_adjust=True, progress=False)["Close"]
px = px[TICKERS].dropna()
rets = px.pct_change().dropna()
rets.index = rets.index + pd.offsets.MonthEnd(0)
print(f"  Returns: {rets.index.min().date()} to {rets.index.max().date()}, n={len(rets)}")

# ---------- 3. Align ----------
data = rets.join(ff5, how="inner")
print(f"  Joined sample: n={len(data)} months")

# ---------- 4. Excess returns ----------
ex_rets = data[TICKERS].sub(data["RF"], axis=0)

# ---------- 5. CAPM and FF5 regressions ----------
results = {}
for tkr in TICKERS:
    y = ex_rets[tkr]

    # CAPM
    Xc = sm.add_constant(data[["Mkt-RF"]])
    capm = sm.OLS(y, Xc).fit()

    # FF5
    Xf = sm.add_constant(data[["Mkt-RF","SMB","HML","RMW","CMA"]])
    ff5m = sm.OLS(y, Xf).fit()

    results[tkr] = {
        "sector": SECTORS[tkr],
        "n": int(capm.nobs),
        "capm": {
            "alpha_monthly_pct": float(capm.params["const"]) * 100,
            "alpha_t":           float(capm.tvalues["const"]),
            "beta":              float(capm.params["Mkt-RF"]),
            "beta_t":            float(capm.tvalues["Mkt-RF"]),
            "r2":                float(capm.rsquared),
            "r2_adj":            float(capm.rsquared_adj),
            "rmse":              float(np.sqrt(capm.mse_resid)),
        },
        "ff5": {
            "alpha_monthly_pct": float(ff5m.params["const"]) * 100,
            "alpha_t":           float(ff5m.tvalues["const"]),
            "beta_mkt":          float(ff5m.params["Mkt-RF"]),
            "beta_smb":          float(ff5m.params["SMB"]),
            "beta_hml":          float(ff5m.params["HML"]),
            "beta_rmw":          float(ff5m.params["RMW"]),
            "beta_cma":          float(ff5m.params["CMA"]),
            "r2":                float(ff5m.rsquared),
            "r2_adj":            float(ff5m.rsquared_adj),
            "rmse":              float(np.sqrt(ff5m.mse_resid)),
        }
    }

# Sample period
results["_meta"] = {
    "start": str(data.index.min().date()),
    "end":   str(data.index.max().date()),
    "n_months": int(len(data)),
    "tickers": TICKERS,
}

with open(f"{OUT_DIR}/results.json","w") as f:
    json.dump(results, f, indent=2)

# Pretty print
print("\n=== CAPM ===")
print(f"{'Ticker':<7}{'Beta':>8}{'Alpha%':>10}{'R²':>9}{'RMSE':>9}")
for t in TICKERS:
    r = results[t]["capm"]
    print(f"{t:<7}{r['beta']:>8.3f}{r['alpha_monthly_pct']:>10.3f}{r['r2']:>9.3f}{r['rmse']:>9.4f}")

print("\n=== FF5 ===")
print(f"{'Ticker':<7}{'βMkt':>7}{'βSMB':>7}{'βHML':>7}{'βRMW':>7}{'βCMA':>7}{'Alpha%':>9}{'R²':>8}")
for t in TICKERS:
    r = results[t]["ff5"]
    print(f"{t:<7}{r['beta_mkt']:>7.2f}{r['beta_smb']:>7.2f}{r['beta_hml']:>7.2f}"
          f"{r['beta_rmw']:>7.2f}{r['beta_cma']:>7.2f}{r['alpha_monthly_pct']:>9.3f}{r['r2']:>8.3f}")

# ---------- 6. Charts ----------
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
PALETTE_PRIMARY = "#1E2761"   # navy
PALETTE_ACCENT  = "#B85042"   # terracotta
PALETTE_GREY    = "#64748B"

# Chart 1: R-squared comparison
fig, ax = plt.subplots(figsize=(7.5, 4.0), dpi=150)
x = np.arange(len(TICKERS))
w = 0.38
capm_r2 = [results[t]["capm"]["r2"] for t in TICKERS]
ff5_r2  = [results[t]["ff5"]["r2"]  for t in TICKERS]
b1 = ax.bar(x - w/2, capm_r2, w, label="CAPM",  color=PALETTE_PRIMARY)
b2 = ax.bar(x + w/2, ff5_r2,  w, label="FF5",   color=PALETTE_ACCENT)
ax.set_xticks(x); ax.set_xticklabels(TICKERS)
ax.set_ylabel("In-sample R²"); ax.set_ylim(0, max(max(capm_r2), max(ff5_r2)) * 1.25)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_title("How much return variation does each model explain?", fontsize=12, pad=10)
ax.legend(frameon=False, loc="upper right")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.3)
for b in list(b1) + list(b2):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
            f"{b.get_height()*100:.0f}%", ha="center", fontsize=9, color="#1E293B")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/chart_r2.png", dpi=200, bbox_inches="tight")
plt.close()

# Chart 2: FF5 factor loadings heatmap
loadings = pd.DataFrame(
    [[results[t]["ff5"][k] for k in
      ["beta_mkt","beta_smb","beta_hml","beta_rmw","beta_cma"]] for t in TICKERS],
    index=TICKERS, columns=["Mkt-RF","SMB","HML","RMW","CMA"]
)
fig, ax = plt.subplots(figsize=(7.5, 3.6), dpi=150)
vmax = max(abs(loadings.values.min()), abs(loadings.values.max()))
im = ax.imshow(loadings.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
ax.set_xticks(range(5)); ax.set_xticklabels(loadings.columns)
ax.set_yticks(range(5)); ax.set_yticklabels(loadings.index)
for i in range(loadings.shape[0]):
    for j in range(loadings.shape[1]):
        v = loadings.values[i,j]
        ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                color="white" if abs(v)>vmax*0.55 else "#1E293B", fontsize=11)
ax.set_title("FF5 factor loadings — what each stock 'looks like' to the model", fontsize=12, pad=10)
cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
cbar.outline.set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/chart_loadings.png", dpi=200, bbox_inches="tight")
plt.close()

# Chart 3: Pipeline diagram (matplotlib-drawn)
fig, ax = plt.subplots(figsize=(9.5, 3.0), dpi=150)
ax.set_xlim(0,10); ax.set_ylim(0,3); ax.axis("off")
boxes = [
    (0.2, "Features\n(market, FF5,\nfundamentals)", "#CADCFC"),
    (2.5, "Return\nprediction model\n(CAPM / FF5 / ML)", PALETTE_PRIMARY),
    (5.0, "Predicted\nμ̂ for each stock", "#CADCFC"),
    (7.3, "Mean-variance\noptimization\n(tangency portfolio)", PALETTE_ACCENT),
]
for x0, label, color in boxes:
    text_color = "white" if color in (PALETTE_PRIMARY, PALETTE_ACCENT) else "#1E293B"
    ax.add_patch(plt.Rectangle((x0,0.7), 2.2, 1.6, facecolor=color, edgecolor="none"))
    ax.text(x0+1.1, 1.5, label, ha="center", va="center", fontsize=11,
            color=text_color, fontweight="bold")
# arrows
for x0 in [2.4, 4.9, 7.2]:
    ax.annotate("", xy=(x0+0.05, 1.5), xytext=(x0-0.1, 1.5),
                arrowprops=dict(arrowstyle="->", color="#1E293B", lw=1.6))
ax.text(5, 0.25, "Models compared on out-of-sample MSFE → best model feeds the optimizer",
        ha="center", fontsize=10, color=PALETTE_GREY, style="italic")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/chart_pipeline.png", dpi=200, bbox_inches="tight")
plt.close()

# Chart 4: Cumulative returns of the 5 stocks (just for context/eye candy)
cum = (1 + rets).cumprod()
fig, ax = plt.subplots(figsize=(7.5, 3.6), dpi=150)
colors = ["#1E2761", "#B85042", "#2C5F2D", "#A26769", "#028090"]
for c, t in zip(colors, TICKERS):
    ax.plot(cum.index, cum[t], label=f"{t} ({SECTORS[t]})", linewidth=1.8, color=c)
ax.set_yscale("log")
ax.set_title("Growth of $1 invested 2010–2024 in each stock", fontsize=12, pad=10)
ax.set_ylabel("Wealth (log scale, $)")
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/chart_cumret.png", dpi=200, bbox_inches="tight")
plt.close()

print("\nWrote: results.json, chart_r2.png, chart_loadings.png, chart_pipeline.png, chart_cumret.png")
