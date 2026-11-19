import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score

# 1. Page Setup
st.set_page_config(page_title="Future Project: Quant Screener & ML", layout="wide")
st.title("🚀 Future Project: Quant Screener & Decision Dashboard")
st.caption("Stock screener, realistic fundamental scoring, purged walk-forward ML guardrails, and position tracker.")

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Dashboard Controls")
    ticker = st.text_input("Deep-Dive Stock Ticker", value="PONY").strip().upper()

    horizon_label = st.selectbox(
        "ML Prediction Horizon",
        options=["5 Trading Days (1 Week)", "20 Trading Days (1 Month)", "1 Trading Day (Noisy)"],
        index=0,
        help="Multi-day horizons filter out daily noise better than 1-day predictions."
    )
    horizon_days = {"5 Trading Days (1 Week)": 5, "20 Trading Days (1 Month)": 20, "1 Trading Day (Noisy)": 1}[horizon_label]

    confidence_threshold = st.slider(
        "Min Confidence for BUY / SELL (%)",
        min_value=55, max_value=75, value=60
    )

    st.divider()
    st.subheader("🔎 Screener Watchlist")
    watchlist_raw = st.text_area(
        "Tickers (comma-separated)",
        value="PONY, QBTS, IONQ, TSLA, NVDA, MSFT, GOOGL, AMD",
        height=90
    )
    watchlist = tuple(dict.fromkeys(
        t.strip().upper() for t in watchlist_raw.split(",") if t.strip()
    ))

    st.divider()
    st.caption(
        "⚠️ Decision-support tool only—not financial advice. "
        "Always check backtest reliability and trend regime before acting."
    )


# 2. Data & Fundamental Pipelines
@st.cache_data(ttl=3600)
def load_market_data(symbol: str, shift_days: int = 5) -> pd.DataFrame:
    data = yf.download(symbol, start="2024-01-01", multi_level_index=False, progress=False)
    if data.empty:
        return data

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data["SMA_50"] = data["Close"].rolling(window=50).mean()
    data["SMA_200"] = data["Close"].rolling(window=200).mean()
    data["Price_to_SMA50"] = data["Close"] / data["SMA_50"]
    data["SMA_50_to_200"] = data["SMA_50"] / data["SMA_200"]

    data["Vol_SMA_20"] = data["Volume"].rolling(window=20).mean()
    data["Volume_Ratio"] = data["Volume"] / data["Vol_SMA_20"]
    data["Daily_Spread"] = (data["High"] - data["Low"]) / data["Close"]

    # 14-Day RSI
    delta = data["Close"].diff()
    gain = delta.clip(lower=0).rolling(window=14).mean()
    loss = (-delta.clip(upper=0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    data["RSI_14"] = 100 - (100 / (1 + rs))

    # Drawdown from 52-week high
    rolling_high = data["Close"].rolling(window=252, min_periods=50).max()
    data["Drawdown_Pct"] = ((data["Close"] - rolling_high) / rolling_high) * 100

    # Target: 1 if close N trading days later is higher
    data["Target"] = (data["Close"].shift(-shift_days) > data["Close"]).astype(int)
    data.dropna(inplace=True)
    return data


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
    try:
        info = stock.info or {}
    except Exception:
        info = {}

    q_income = stock.quarterly_income_stmt
    q_balance = stock.quarterly_balance_sheet
    q_cashflow = stock.quarterly_cashflow

    latest_q = q_income.columns[0].strftime("%Y-%m-%d") if (q_income is not None and not q_income.empty) else "N/A"
    rev_now = _extract_row(q_income, ["Total Revenue", "Operating Revenue"], 0)
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

    gross_margin = (gross_profit / rev_now * 100) if (gross_profit is not None and rev_now) else None
    op_margin = (op_income / rev_now * 100) if (op_income is not None and rev_now) else None
    runway_quarters = (cash / abs(op_cashflow)) if (cash and op_cashflow and op_cashflow < 0) else None
    ps_ratio = info.get("priceToSalesTrailing12Months")
    rev_growth_yoy = info.get("revenueGrowth")

    # Balanced 4-Pillar Fundamental Score (0-100)
    pts, max_pts = 0, 0
    if rev_growth_yoy is not None:
        max_pts += 25
        pts += 25 if rev_growth_yoy > 0.25 else 15 if rev_growth_yoy > 0.10 else 8 if rev_growth_yoy > 0 else 0

    if op_margin is not None:
        max_pts += 25
        pts += 25 if op_margin >= 15 else 15 if op_margin > 0 else 5 if op_margin > -30 else 0

    if op_cashflow is not None:
        max_pts += 25
        if op_cashflow > 0:
            pts += 25
        elif runway_quarters is not None:
            pts += 20 if runway_quarters >= 12 else 14 if runway_quarters >= 8 else 7 if runway_quarters >= 4 else 0

    if ps_ratio is not None and ps_ratio > 0:
        max_pts += 25
        pts += 25 if ps_ratio < 5 else 18 if ps_ratio < 15 else 10 if ps_ratio < 30 else 0

    fund_score = round((pts / max_pts) * 100) if max_pts > 0 else None

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
        "ps_ratio": ps_ratio,
        "rev_growth_yoy": rev_growth_yoy,
        "latest_q": latest_q,
        "rev_now": rev_now,
        "gross_margin": gross_margin,
        "op_margin": op_margin,
        "cash": cash,
        "op_cashflow": op_cashflow,
        "runway_quarters": runway_quarters,
        "fund_score": fund_score,
        "rev_series": rev_series,
    }


def classify_trend(close: float, sma50: float, sma200: float) -> str:
    if close > sma50 and sma50 > sma200:
        return "🟢 Uptrend"
    if close > sma50 and sma50 <= sma200:
        return "🟡 Recovery Watch"
    if close <= sma50 and sma50 > sma200:
        return "🟠 Pullback"
    return "🔴 Downtrend"


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


@st.cache_data(ttl=3600)
def build_screener_table(tickers: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for sym in tickers:
        try:
            df = load_market_data(sym, shift_days=5)
            if df.empty or len(df) < 10:
                continue
            f = load_fundamentals(sym)
            close = float(df["Close"].iloc[-1])
            prev = float(df["Close"].iloc[-2])
            sma50 = float(df["SMA_50"].iloc[-1])
            sma200 = float(df["SMA_200"].iloc[-1])
            rsi = float(df["RSI_14"].iloc[-1])
            dd = float(df["Drawdown_Pct"].iloc[-1])
            trend = classify_trend(close, sma50, sma200)

            score = f["fund_score"] if f["fund_score"] is not None else 0
            if trend == "🟢 Uptrend" and score >= 60 and rsi < 70:
                verdict = "✅ Strong Candidate"
            elif trend == "🟡 Recovery Watch" and score >= 50:
                verdict = "👀 Early Turnaround"
            elif trend == "🔴 Downtrend":
                verdict = "🛑 Falling Knife / Wait"
            elif rsi >= 70:
                verdict = "🔥 Overbought"
            else:
                verdict = "⚖️ Neutral / Watch"

            rows.append({
                "Ticker": sym,
                "Price ($)": round(close, 2),
                "1D %": round(((close - prev) / prev) * 100, 2),
                "Trend Regime": trend,
                "vs SMA50 %": round(((close - sma50) / sma50) * 100, 1),
                "RSI (14)": round(rsi, 1),
                "From 52W High %": round(dd, 1),
                "Fund Score": f["fund_score"],
                "P/S Ratio": round(f["ps_ratio"], 1) if f["ps_ratio"] else np.nan,
                "Op Margin %": round(f["op_margin"], 1) if f["op_margin"] is not None else np.nan,
                "Screener Verdict": verdict,
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


# 3. Main Workspace Tabs
tab_screener, tab_positions, tab_deep, tab_fund = st.tabs([
    "🔎 Stock Screener",
    "💼 My Positions & Break-Even",
    "📈 Deep-Dive ML & Technicals",
    "💰 Live Financials & Runway"
])

# --- TAB 1: STOCK SCREENER ---
with tab_screener:
    st.subheader("Watchlist Stock Screener")
    st.caption("Compare your holdings against peers and quality leaders using trend, momentum, profitability, and valuation.")

    with st.spinner("Scanning watchlist..."):
        screen_df = build_screener_table(watchlist)

    if screen_df.empty:
        st.warning("No valid tickers found in the watchlist.")
    else:
        fc1, fc2, fc3 = st.columns(3)
        min_score = fc1.slider("Min Fundamental Score", 0, 100, 0, step=5)
        max_rsi = fc2.slider("Max RSI (Avoid Overbought)", 30, 100, 100)
        trend_filter = fc3.multiselect(
            "Filter by Trend Regime",
            options=["🟢 Uptrend", "🟡 Recovery Watch", "🟠 Pullback", "🔴 Downtrend"],
            default=["🟢 Uptrend", "🟡 Recovery Watch", "🟠 Pullback", "🔴 Downtrend"]
        )

        filtered = screen_df[
            (screen_df["Fund Score"].fillna(0) >= min_score) &
            (screen_df["RSI (14)"] <= max_rsi) &
            (screen_df["Trend Regime"].isin(trend_filter))
        ].sort_values(by="Fund Score", ascending=False)

        st.dataframe(filtered, use_container_width=True, hide_index=True)

        st.info(
            "**How to read this:** A stock marked **🛑 Falling Knife / Wait** (`Price < SMA_50 < SMA_200`) "
            "has not yet proven it has bottomed. Waiting for **🟡 Recovery Watch** (reclaiming `SMA_50`) "
            "or focusing new buys on **✅ Strong Candidate** stocks reduces the risk of getting stuck."
        )

# --- TAB 2: MY POSITIONS & BREAK-EVEN ---
with tab_positions:
    st.subheader("Position & Recovery Calculator (Private to Your Session)")
    st.caption("Edit your average buy price and share count below—nothing entered here is saved to GitHub.")

    default_positions = pd.DataFrame([
        {"Ticker": "PONY", "Avg Buy Price ($)": 10.00, "Shares": 100.0},
        {"Ticker": "QBTS", "Avg Buy Price ($)": 15.00, "Shares": 100.0},
    ])

    edited_pos = st.data_editor(
        default_positions,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True
    )

    pos_rows = []
    total_cost, total_val = 0.0, 0.0
    for _, row in edited_pos.iterrows():
        sym = str(row["Ticker"]).strip().upper()
        buy_p = float(row["Avg Buy Price ($)"] or 0)
        shares = float(row["Shares"] or 0)
        if not sym or buy_p <= 0 or shares <= 0:
            continue

        df_p = load_market_data(sym, shift_days=5)
        if df_p.empty:
            continue

        curr_p = float(df_p["Close"].iloc[-1])
        sma50 = float(df_p["SMA_50"].iloc[-1])
        sma200 = float(df_p["SMA_200"].iloc[-1])

        cost = buy_p * shares
        val = curr_p * shares
        pnl = val - cost
        pnl_pct = ((curr_p - buy_p) / buy_p) * 100
        gain_to_even = max(0.0, ((buy_p - curr_p) / curr_p) * 100)
        dist_sma50 = ((sma50 - curr_p) / curr_p) * 100

        total_cost += cost
        total_val += val

        pos_rows.append({
            "Ticker": sym,
            "Current ($)": round(curr_p, 2),
            "Avg Cost ($)": round(buy_p, 2),
            "Value ($)": round(val, 2),
            "P&L ($)": round(pnl, 2),
            "P&L (%)": round(pnl_pct, 1),
            "% Gain to Break Even": round(gain_to_even, 1),
            "SMA50 Reclaim Target ($)": round(sma50, 2),
            "% to Reclaim SMA50": round(dist_sma50, 1),
            "Trend Regime": classify_trend(curr_p, sma50, sma200),
        })

    if pos_rows:
        p1, p2, p3 = st.columns(3)
        tot_pnl = total_val - total_cost
        tot_pct = (tot_pnl / total_cost * 100) if total_cost > 0 else 0
        p1.metric("Total Invested", f"${total_cost:,.2f}")
        p2.metric("Current Value", f"${total_val:,.2f}")
        p3.metric("Total Unrealized P&L", f"${tot_pnl:,.2f}", f"{tot_pct:+.1f}%")
        st.dataframe(pd.DataFrame(pos_rows), use_container_width=True, hide_index=True)

# --- LOAD DEEP-DIVE TICKER FOR TABS 3 & 4 ---
data = load_market_data(ticker, shift_days=horizon_days) if ticker else pd.DataFrame()
fund = load_fundamentals(ticker) if ticker else {}

# --- TAB 3: DEEP-DIVE ML & TECHNICALS ---
with tab_deep:
    if data.empty or len(data) < 35:
        st.error(f"Not enough historical price data for '{ticker}' (needs >200 trading days for SMA_200).")
    else:
        st.subheader(f"{fund['name']} ({ticker}) — {horizon_label} Model")

        features = ["Price_to_SMA50", "SMA_50_to_200", "Volume_Ratio", "Daily_Spread", "RSI_14"]
        train_data = data.iloc[:-horizon_days]
        X_train = train_data[features]
        y_train = train_data["Target"]

        # Safe purged TimeSeriesSplit (prevents overlapping multi-day target leakage)
        n_splits = 5
        gap = horizon_days
        while n_splits > 2:
            test_size = len(train_data) // (n_splits + 1)
            first_train = len(train_data) - n_splits * test_size - gap
            if first_train >= 30 and test_size >= 15:
                break
            n_splits -= 1

        if len(train_data) - n_splits * (len(train_data) // (n_splits + 1)) - gap < 15:
            gap = 0

        tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
        all_y_true, all_y_pred = [], []

        for tr_idx, te_idx in tscv.split(X_train):
            bt_model = RandomForestClassifier(
                n_estimators=150, max_depth=5, min_samples_leaf=8,
                random_state=42, class_weight="balanced"
            )
            bt_model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
            preds = bt_model.predict(X_train.iloc[te_idx])
            all_y_true.extend(y_train.iloc[te_idx].tolist())
            all_y_pred.extend(preds.tolist())

        mean_acc = float(accuracy_score(all_y_true, all_y_pred))
        buy_prec = float(precision_score(all_y_true, all_y_pred, pos_label=1, zero_division=0))
        sell_prec = float(precision_score(all_y_true, all_y_pred, pos_label=0, zero_division=0))

        model = RandomForestClassifier(
            n_estimators=150, max_depth=5, min_samples_leaf=8,
            random_state=42, class_weight="balanced"
        )
        model.fit(X_train, y_train)

        latest_features = data[features].iloc[[-1]]
        probs = model.predict_proba(latest_features)[0]
        prob_up = probs[1] * 100
        prob_down = probs[0] * 100
        confidence = max(prob_up, prob_down)

        latest_close = float(data["Close"].iloc[-1])
        prev_close = float(data["Close"].iloc[-2])
        day_pct = ((latest_close - prev_close) / prev_close) * 100
        trend_now = classify_trend(latest_close, float(data["SMA_50"].iloc[-1]), float(data["SMA_200"].iloc[-1]))

        # Tightened Guardrails: 55% Accuracy + 55% Directional Precision + Trend Alignment
        MIN_EDGE = 0.55
        guardrail_reason = None

        if prob_up >= confidence_threshold:
            if mean_acc < MIN_EDGE or buy_prec < MIN_EDGE:
                signal = "⚠️ BLOCKED (LOW EDGE)"
                guardrail_reason = (
                    f"Raw model leaned BUY ({prob_up:.1f}%), but failed the 55% reliability floor "
                    f"(Purged Accuracy: **{mean_acc:.1%}**, Buy Precision: **{buy_prec:.1%}**)."
                )
            elif trend_now == "🔴 Downtrend":
                signal = "⚠️ RISKY BOUNCE"
                guardrail_reason = (
                    f"Model passed backtest checks, but **{ticker}** is still in a **🔴 Downtrend** "
                    "(Price < SMA_50 < SMA_200). Wait for an SMA_50 reclaim before treating as a clean BUY."
                )
            else:
                signal = "BUY 🟢"
        elif prob_down >= confidence_threshold:
            if mean_acc < MIN_EDGE or sell_prec < MIN_EDGE:
                signal = "⚠️ BLOCKED (LOW EDGE)"
                guardrail_reason = (
                    f"Raw model leaned SELL ({prob_down:.1f}%), but failed the 55% reliability floor "
                    f"(Purged Accuracy: **{mean_acc:.1%}**, Sell Precision: **{sell_prec:.1%}**)."
                )
            else:
                signal = "SELL 🔴"
        else:
            signal = "HOLD 🟡"

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Current Price", f"${latest_close:.2f}", f"{day_pct:+.2f}%")
        c2.metric("ML Signal", signal)
        c3.metric("Model Confidence", f"{confidence:.1f}%", f"Min: {confidence_threshold}%")
        c4.metric("Purged Accuracy", f"{mean_acc:.1%}", f"Buy: {buy_prec:.1%} | Sell: {sell_prec:.1%}")
        c5.metric(
            "Fundamental Score",
            f"{fund['fund_score']}/100" if fund["fund_score"] is not None else "N/A",
            f"Q-End: {fund['latest_q']}"
        )

        # Guardrail Checklist Strip
        g_acc = "✅" if mean_acc >= MIN_EDGE else "❌"
        g_buy = "✅" if buy_prec >= MIN_EDGE else "❌"
        g_sell = "✅" if sell_prec >= MIN_EDGE else "❌"
        st.caption(
            f"**Guardrail Check (Purge Gap: {gap}d):** "
            f"Accuracy ≥55% {g_acc} ({mean_acc:.1%})  |  "
            f"Buy Prec ≥55% {g_buy} ({buy_prec:.1%})  |  "
            f"Sell Prec ≥55% {g_sell} ({sell_prec:.1%})  |  "
            f"Trend: {trend_now}"
        )

        if guardrail_reason:
            st.warning(guardrail_reason)

        st.divider()
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

# --- TAB 4: LIVE FINANCIALS & RUNWAY ---
with tab_fund:
    if not fund:
        st.warning("Enter a valid ticker in the sidebar.")
    else:
        st.subheader(f"{fund['name']} ({ticker}) — Fundamental Reality Check")
        st.caption(f"Latest structured filing via Yahoo Finance: **{fund['latest_q']}**")

        f1, f2, f3, f4, f5 = st.columns(5)
        yoy_str = f"{fund['rev_growth_yoy'] * 100:+.1f}% YoY" if fund["rev_growth_yoy"] is not None else None
        f1.metric("Quarterly Revenue", fmt_money(fund["rev_now"]), yoy_str)
        f2.metric("Operating Margin", f"{fund['op_margin']:.1f}%" if fund["op_margin"] is not None else "N/A")
        f3.metric("Price / Sales", f"{fund['ps_ratio']:.1f}x" if fund["ps_ratio"] else "N/A")
        f4.metric("Cash & Short-Term Inv.", fmt_money(fund["cash"]))

        if fund["runway_quarters"] is not None:
            f5.metric("Est. Cash Runway", f"{fund['runway_quarters']:.1f} Qtrs", f"Burn: {fmt_money(fund['op_cashflow'])}/Q")
        else:
            f5.metric("Operating Cash Flow", fmt_money(fund["op_cashflow"]))

        if fund["rev_series"] is not None and not fund["rev_series"].empty:
            st.subheader("Quarterly Revenue Trajectory ($ Millions)")
            st.bar_chart(fund["rev_series"])