import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score

# 1. Page Setup
st.set_page_config(page_title="Future Project: Quant ML Dashboard", layout="wide")
st.title("🚀 Future Project: Quant ML & Fundamentals Dashboard")
st.caption("Two-layer decision engine combining a walk-forward Random Forest with live quarterly financials.")

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Model Controls")
    ticker = st.text_input("Stock Ticker", value="PONY").strip().upper()
    confidence_threshold = st.slider(
        "Min Confidence for BUY / SELL (%)",
        min_value=50, max_value=75, value=60,
        help="Signals below this threshold are flagged as HOLD to reduce noise."
    )
    st.divider()
    st.info(
        "**Note on Fresh Earnings:** Yahoo Finance structured tables can lag "
        "a few days behind brand-new earnings press releases. Always check the "
        "quarter-end date displayed on the Financials tab."
    )

# 2. Daily Market Data Pipeline (refreshes every hour in the cloud)
@st.cache_data(ttl=3600)
def load_market_data(symbol: str) -> pd.DataFrame:
    data = yf.download(symbol, start="2024-01-01", multi_level_index=False, progress=False)
    if data.empty:
        return data

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Moving Averages & Stationary Ratios
    data["SMA_50"] = data["Close"].rolling(window=50).mean()
    data["SMA_200"] = data["Close"].rolling(window=200).mean()
    data["Price_to_SMA50"] = data["Close"] / data["SMA_50"]
    data["SMA_50_to_200"] = data["SMA_50"] / data["SMA_200"]

    # Volume & Volatility
    data["Vol_SMA_20"] = data["Volume"].rolling(window=20).mean()
    data["Volume_Ratio"] = data["Volume"] / data["Vol_SMA_20"]
    data["Daily_Spread"] = (data["High"] - data["Low"]) / data["Close"]

    # 14-Day RSI (Momentum)
    delta = data["Close"].diff()
    gain = delta.clip(lower=0).rolling(window=14).mean()
    loss = (-delta.clip(upper=0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    data["RSI_14"] = 100 - (100 / (1 + rs))

    # Next-day directional target (1 = up, 0 = down/flat)
    data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
    data.dropna(inplace=True)
    return data


# 3. Live Quarterly Fundamentals Pipeline
def _extract_row(df: pd.DataFrame, candidates: list[str], col_idx: int = 0):
    if df is None or df.empty or col_idx >= len(df.columns):
        return None
    for label in candidates:
        if label in df.index:
            val = df.loc[label].iloc[col_idx]
            if pd.notna(val):
                return float(val)
    return None


@st.cache_data(ttl=3600)
def load_fundamentals(symbol: str) -> dict:
    stock = yf.Ticker(symbol)
    info = stock.info or {}
    q_income = stock.quarterly_income_stmt
    q_balance = stock.quarterly_balance_sheet
    q_cashflow = stock.quarterly_cashflow

    latest_q = q_income.columns[0].strftime("%Y-%m-%d") if (q_income is not None and not q_income.empty) else "N/A"

    rev_now = _extract_row(q_income, ["Total Revenue", "Operating Revenue"], 0)
    rev_prev = _extract_row(q_income, ["Total Revenue", "Operating Revenue"], 1)
    gross_profit = _extract_row(q_income, ["Gross Profit"], 0)
    op_income = _extract_row(q_income, ["Operating Income", "EBIT"], 0)

    cash = _extract_row(
        q_balance,
        ["Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents"],
        0,
    )
    op_cashflow = _extract_row(
        q_cashflow,
        ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"],
        0,
    )

    # Derived financial health metrics
    qoq_rev_growth = ((rev_now - rev_prev) / abs(rev_prev) * 100) if (rev_now and rev_prev) else None
    gross_margin = (gross_profit / rev_now * 100) if (gross_profit is not None and rev_now) else None
    runway_quarters = (cash / abs(op_cashflow)) if (cash and op_cashflow and op_cashflow < 0) else None

    # Quarterly revenue history for charting
    rev_series = None
    if q_income is not None and not q_income.empty:
        for label in ["Total Revenue", "Operating Revenue"]:
            if label in q_income.index:
                s = q_income.loc[label].dropna().sort_index() / 1e6
                s.index = s.index.strftime("%Y-%m-%d")
                rev_series = s.rename("Revenue ($M)")
                break

    return {
        "name": info.get("shortName", symbol),
        "market_cap": info.get("marketCap"),
        "target_mean": info.get("targetMeanPrice"),
        "rev_growth_yoy": info.get("revenueGrowth"),
        "latest_q": latest_q,
        "rev_now": rev_now,
        "qoq_rev_growth": qoq_rev_growth,
        "gross_margin": gross_margin,
        "op_income": op_income,
        "cash": cash,
        "op_cashflow": op_cashflow,
        "runway_quarters": runway_quarters,
        "rev_series": rev_series,
    }


def fmt_money(val) -> str:
    if val is None or pd.isna(val):
        return "N/A"
    abs_val = abs(val)
    sign = "-" if val < 0 else ""
    if abs_val >= 1e9:
        return f"{sign}${abs_val / 1e9:.2f}B"
    if abs_val >= 1e6:
        return f"{sign}${abs_val / 1e6:.2f}M"
    return f"{sign}${abs_val:,.0f}"


# Load Data
if not ticker:
    st.warning("Enter a stock ticker in the sidebar to begin.")
    st.stop()

data = load_market_data(ticker)
if data.empty or len(data) < 30:
    st.error(f"Not enough historical price data for '{ticker}' (needs >200 trading days for SMA_200).")
    st.stop()

fund = load_fundamentals(ticker)
st.subheader(f"{fund['name']} ({ticker})")

# 4. Walk-Forward Backtest & Live Prediction
features = ["Price_to_SMA50", "SMA_50_to_200", "Volume_Ratio", "Daily_Spread", "RSI_14"]
train_data = data.iloc[:-1]
X_train = train_data[features]
y_train = train_data["Target"]

n_splits = min(5, max(2, len(train_data) // 40))
tscv = TimeSeriesSplit(n_splits=n_splits)
accs, precs = [], []

for tr_idx, te_idx in tscv.split(X_train):
    bt_model = RandomForestClassifier(
        n_estimators=150, max_depth=6, min_samples_leaf=5,
        random_state=42, class_weight="balanced"
    )
    bt_model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
    preds = bt_model.predict(X_train.iloc[te_idx])
    accs.append(accuracy_score(y_train.iloc[te_idx], preds))
    precs.append(precision_score(y_train.iloc[te_idx], preds, zero_division=0))

# Train final model on all completed days
model = RandomForestClassifier(
    n_estimators=150, max_depth=6, min_samples_leaf=5,
    random_state=42, class_weight="balanced"
)
model.fit(X_train, y_train)

latest_features = data[features].iloc[[-1]]
probs = model.predict_proba(latest_features)[0]
prob_up = probs[1] * 100
prob_down = probs[0] * 100
confidence = max(prob_up, prob_down)

if prob_up >= confidence_threshold:
    signal = "BUY 🟢"
elif prob_down >= confidence_threshold:
    signal = "SELL 🔴"
else:
    signal = "HOLD 🟡"

# 5. Fundamental Health Score (0-100)
fund_points, fund_max = 0, 0
if fund["rev_growth_yoy"] is not None:
    fund_max += 35
    if fund["rev_growth_yoy"] > 0.25:
        fund_points += 35
    elif fund["rev_growth_yoy"] > 0:
        fund_points += 20
if fund["runway_quarters"] is not None:
    fund_max += 35
    if fund["runway_quarters"] >= 8:
        fund_points += 35
    elif fund["runway_quarters"] >= 4:
        fund_points += 20
elif fund["op_cashflow"] is not None and fund["op_cashflow"] > 0:
    fund_max += 35
    fund_points += 35  # Cash-flow positive
if fund["gross_margin"] is not None:
    fund_max += 30
    if fund["gross_margin"] >= 40:
        fund_points += 30
    elif fund["gross_margin"] >= 15:
        fund_points += 18

fund_score = round((fund_points / fund_max) * 100) if fund_max > 0 else None

# 6. Top KPI Strip
latest_close = float(data["Close"].iloc[-1])
prev_close = float(data["Close"].iloc[-2])
day_pct = ((latest_close - prev_close) / prev_close) * 100

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Current Price", f"${latest_close:.2f}", f"{day_pct:+.2f}%")
c2.metric("ML Signal", signal)
c3.metric("Model Confidence", f"{confidence:.1f}%", f"Min: {confidence_threshold}%")
c4.metric("Backtest Accuracy", f"{np.mean(accs):.1%}", f"Buy Prec: {np.mean(precs):.1%}")
c5.metric(
    "Fundamental Score",
    f"{fund_score}/100" if fund_score is not None else "N/A",
    f"Q-End: {fund['latest_q']}"
)

st.divider()

# 7. Tabbed Workspace
tab_tech, tab_fund, tab_model = st.tabs([
    "📈 Technicals & Signals",
    "💰 Live Financials & Runway",
    "🧠 Feature Importance & Data"
])

with tab_tech:
    col_chart, col_rsi = st.columns([2, 1])
    with col_chart:
        st.subheader("Price Action vs. Moving Averages")
        st.line_chart(data[["Close", "SMA_50", "SMA_200"]])
    with col_rsi:
        st.subheader("14-Day RSI (Momentum)")
        st.line_chart(data[["RSI_14"]])
        latest_rsi = float(data["RSI_14"].iloc[-1])
        rsi_state = "Oversold (<30)" if latest_rsi < 30 else "Overbought (>70)" if latest_rsi > 70 else "Neutral (30–70)"
        st.caption(f"Latest RSI: **{latest_rsi:.1f}** — {rsi_state}")

with tab_fund:
    st.caption(f"Latest structured filing available via Yahoo Finance: **{fund['latest_q']}**")
    f1, f2, f3, f4 = st.columns(4)
    yoy_str = f"{fund['rev_growth_yoy'] * 100:+.1f}% YoY" if fund["rev_growth_yoy"] is not None else None
    f1.metric("Quarterly Revenue", fmt_money(fund["rev_now"]), yoy_str)
    f2.metric("Cash & Short-Term Inv.", fmt_money(fund["cash"]))

    if fund["runway_quarters"] is not None:
        f3.metric("Est. Cash Runway", f"{fund['runway_quarters']:.1f} Qtrs", f"Burn: {fmt_money(fund['op_cashflow'])}/Q")
    else:
        f3.metric("Operating Cash Flow", fmt_money(fund["op_cashflow"]))

    if fund["target_mean"] is not None:
        upside = ((fund["target_mean"] - latest_close) / latest_close) * 100
        f4.metric("Analyst Mean Target", f"${fund['target_mean']:.2f}", f"{upside:+.1f}% vs. price")
    else:
        f4.metric("Market Cap", fmt_money(fund["market_cap"]))

    if fund["rev_series"] is not None and not fund["rev_series"].empty:
        st.subheader("Quarterly Revenue Trajectory ($ Millions)")
        st.bar_chart(fund["rev_series"])

with tab_model:
    col_imp, col_mat = st.columns([1, 2])
    with col_imp:
        st.subheader("Random Forest Feature Weights")
        imp_df = pd.DataFrame({
            "Feature": features,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False).set_index("Feature")
        st.bar_chart(imp_df)
    with col_mat:
        st.subheader("Recent Feature Matrix (Last 7 Days)")
        st.dataframe(data[features].tail(7).round(4), use_container_width=True)