import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score

# 1. Page Setup
st.set_page_config(page_title="Future Project: Quant Screener & ML", layout="wide")
st.title("🚀 Future Project: Quant Screener & Decision Dashboard")
st.caption("Macro-aware ML (QQQ/VIX), Triple-Barrier labeling, realistic fundamentals, and position recovery tracker.")

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Dashboard Controls")
    ticker = st.text_input("Deep-Dive Stock Ticker", value="PONY").strip().upper()

    horizon_label = st.selectbox(
        "Triple-Barrier Horizon",
        options=["10 Trading Days (2 Weeks)", "20 Trading Days (1 Month)", "5 Trading Days (1 Week)"],
        index=1,
        help="Target = hit +1.5x ATR profit target before -1.0x ATR stop-loss within this window."
    )
    horizon_days = {
        "5 Trading Days (1 Week)": 5,
        "10 Trading Days (2 Weeks)": 10,
        "20 Trading Days (1 Month)": 20,
    }[horizon_label]

    model_choice = st.selectbox(
        "ML Algorithm",
        options=["HistGradientBoosting (L2 Reg)", "Random Forest (Regularized)"],
        index=0,
        help="Gradient boosting learns nonlinear momentum/macro interactions sequentially."
    )

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
        "Check Purged Accuracy, Buy Precision, and Trend Regime before acting."
    )


# 2. Macro Market-Regime Pipeline (Cached once per hour for all tickers)
def _clean_yf(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


@st.cache_data(ttl=3600)
def load_macro_data() -> pd.DataFrame:
    qqq = _clean_yf(yf.download("QQQ", start="2021-01-01", multi_level_index=False, progress=False))
    vix = _clean_yf(yf.download("^VIX", start="2021-01-01", multi_level_index=False, progress=False))
    if qqq.empty or vix.empty:
        return pd.DataFrame()

    macro = pd.DataFrame(index=qqq.index)
    qqq_sma50 = qqq["Close"].rolling(window=50).mean()
    macro["QQQ_Regime"] = qqq["Close"] / qqq_sma50
    macro["QQQ_Ret_20"] = qqq["Close"].pct_change(20)
    macro["VIX_Level"] = vix["Close"].reindex(qqq.index).ffill()
    macro["VIX_Chg_10"] = macro["VIX_Level"].pct_change(10)
    return macro.dropna()


def apply_triple_barrier(df: pd.DataFrame, horizon: int, tp_mult: float = 1.5, sl_mult: float = 1.0) -> pd.Series:
    """Label 1 if +tp_mult*ATR is hit before -sl_mult*ATR (or closes green at timeout without stopping out)."""
    closes = df["Close"].to_numpy()
    highs = df["High"].to_numpy()
    lows = df["Low"].to_numpy()
    atrs = df["ATR_14"].to_numpy()
    n = len(df)
    targets = np.full(n, np.nan)

    for i in range(n - horizon):
        atr = atrs[i]
        entry = closes[i]
        if np.isnan(atr) or atr <= 0 or np.isnan(entry):
            continue
        upper = entry + tp_mult * atr
        lower = entry - sl_mult * atr
        hit = None
        for j in range(i + 1, i + 1 + horizon):
            hit_up = highs[j] >= upper
            hit_dn = lows[j] <= lower
            if hit_up and hit_dn:
                hit = 0  # Conservative on same-day whipsaw
                break
            if hit_dn:
                hit = 0
                break
            if hit_up:
                hit = 1
                break
        if hit is None:
            hit = 1 if closes[i + horizon] > entry else 0
        targets[i] = hit

    return pd.Series(targets, index=df.index)


BASE_FEATURES = [
    "Price_to_EMA20",
    "Price_to_SMA50",
    "EMA20_to_SMA50",
    "Ret_5",
    "Ret_10",
    "Ret_20",
    "Volume_Ratio",
    "ATR_Pct",
    "MACD_Hist_Norm",
    "MACD_Slope",
    "BB_Width",
    "BB_PctB",
    "RSI_14",
]
MACRO_FEATURES = ["Rel_Strength_20", "QQQ_Regime", "VIX_Level", "VIX_Chg_10"]


@st.cache_data(ttl=3600)
def load_market_data(symbol: str, shift_days: int = 20) -> tuple[pd.DataFrame, list[str]]:
    data = _clean_yf(yf.download(symbol, start="2021-01-01", multi_level_index=False, progress=False))
    if data.empty:
        return data, []

    # Trend Indicators (SMA_200 kept for chart/regime only so recent IPOs don't lose 150 training days)
    data["EMA_20"] = data["Close"].ewm(span=20, adjust=False).mean()
    data["SMA_50"] = data["Close"].rolling(window=50).mean()
    data["SMA_200"] = data["Close"].rolling(window=200, min_periods=200).mean()
    data["Price_to_EMA20"] = data["Close"] / data["EMA_20"]
    data["Price_to_SMA50"] = data["Close"] / data["SMA_50"]
    data["EMA20_to_SMA50"] = data["EMA_20"] / data["SMA_50"]

    # Velocity & Liquidity
    data["Ret_5"] = data["Close"].pct_change(5)
    data["Ret_10"] = data["Close"].pct_change(10)
    data["Ret_20"] = data["Close"].pct_change(20)
    data["Vol_SMA_20"] = data["Volume"].rolling(window=20).mean()
    data["Volume_Ratio"] = data["Volume"] / data["Vol_SMA_20"].replace(0, np.nan)

    # 14-Day True Range & ATR %
    tr1 = data["High"] - data["Low"]
    tr2 = (data["High"] - data["Close"].shift(1)).abs()
    tr3 = (data["Low"] - data["Close"].shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    data["ATR_14"] = true_range.rolling(window=14).mean()
    data["ATR_Pct"] = data["ATR_14"] / data["Close"]

    # MACD Histogram (Scale-Normalized) & 3-Day Slope
    ema12 = data["Close"].ewm(span=12, adjust=False).mean()
    ema26 = data["Close"].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    data["MACD_Hist_Norm"] = (macd_line - macd_signal) / data["Close"]
    data["MACD_Slope"] = data["MACD_Hist_Norm"].diff(3)

    # Bollinger Bands (20, 2)
    bb_mid = data["Close"].rolling(window=20).mean()
    bb_std = data["Close"].rolling(window=20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    data["BB_Width"] = (bb_upper - bb_lower) / bb_mid
    data["BB_PctB"] = (data["Close"] - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)

    # 14-Day Wilder RSI
    delta = data["Close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    data["RSI_14"] = 100 - (100 / (1 + rs))

    # Drawdown from 52-week high
    rolling_high = data["Close"].rolling(window=252, min_periods=50).max()
    data["Drawdown_Pct"] = ((data["Close"] - rolling_high) / rolling_high) * 100

    # Join Macro Context (QQQ & ^VIX)
    active_features = list(BASE_FEATURES)
    macro = load_macro_data()
    if not macro.empty:
        data = data.join(macro, how="left")
        data[["QQQ_Regime", "QQQ_Ret_20", "VIX_Level", "VIX_Chg_10"]] = (
            data[["QQQ_Regime", "QQQ_Ret_20", "VIX_Level", "VIX_Chg_10"]].ffill()
        )
        data["Rel_Strength_20"] = data["Ret_20"] - data["QQQ_Ret_20"]
        active_features += MACRO_FEATURES

    # Triple-Barrier Target (last `shift_days` rows stay NaN so live row is preserved)
    data["Target"] = apply_triple_barrier(data, horizon=shift_days, tp_mult=1.5, sl_mult=1.0)
    data.dropna(subset=active_features, inplace=True)
    return data, active_features


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
    if pd.isna(sma200):
        return "🟡 Recovery Watch" if close > sma50 else "🔴 Downtrend"
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


def make_model(choice: str):
    if choice.startswith("HistGradientBoosting"):
        return HistGradientBoostingClassifier(
            max_iter=150,
            max_depth=3,
            learning_rate=0.05,
            min_samples_leaf=15,
            l2_regularization=2.0,
            random_state=42,
        )
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=10,
        random_state=42,
    )


@st.cache_data(ttl=3600)
def build_screener_table(tickers: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for sym in tickers:
        try:
            df, _ = load_market_data(sym, shift_days=20)
            if df.empty or len(df) < 10:
                continue
            f = load_fundamentals(sym)
            close = float(df["Close"].iloc[-1])
            prev = float(df["Close"].iloc[-2])
            sma50 = float(df["SMA_50"].iloc[-1])
            sma200 = float(df["SMA_200"].iloc[-1]) if pd.notna(df["SMA_200"].iloc[-1]) else np.nan
            rsi = float(df["RSI_14"].iloc[-1])
            dd = float(df["Drawdown_Pct"].iloc[-1])
            rel_qqq = float(df["Rel_Strength_20"].iloc[-1] * 100) if "Rel_Strength_20" in df.columns else np.nan
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
                "vs QQQ 20D %": round(rel_qqq, 1) if pd.notna(rel_qqq) else np.nan,
                "RSI (14)": round(rsi, 1),
                "From 52W High %": round(dd, 1),
                "Fund Score": f["fund_score"],
                "P/S Ratio": round(f["ps_ratio"], 1) if f["ps_ratio"] else np.nan,
                "Screener Verdict": verdict,
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


# 4. Main Workspace Tabs
tab_screener, tab_positions, tab_deep, tab_fund = st.tabs([
    "🔎 Stock Screener",
    "💼 My Positions & Break-Even",
    "📈 Deep-Dive ML & Technicals",
    "💰 Live Financials & Runway"
])

# --- TAB 1: STOCK SCREENER ---
with tab_screener:
    st.subheader("Watchlist Stock Screener")
    st.caption("Compare your holdings against peers and quality leaders using trend, QQQ relative strength, momentum, and fundamentals.")

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
            "**How to read this:** Focus new entries on **✅ Strong Candidate** stocks with positive **vs QQQ 20D %** "
            "(beating the tech index). Stocks marked **🛑 Falling Knife / Wait** have not yet reclaimed their 50-day average."
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

        df_p, _ = load_market_data(sym, shift_days=20)
        if df_p.empty:
            continue

        curr_p = float(df_p["Close"].iloc[-1])
        sma50 = float(df_p["SMA_50"].iloc[-1])
        sma200 = float(df_p["SMA_200"].iloc[-1]) if pd.notna(df_p["SMA_200"].iloc[-1]) else np.nan

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
data, active_features = load_market_data(ticker, shift_days=horizon_days) if ticker else (pd.DataFrame(), [])
fund = load_fundamentals(ticker) if ticker else {}

# --- TAB 3: DEEP-DIVE ML & TECHNICALS ---
with tab_deep:
    train_data = data.dropna(subset=["Target"]) if not data.empty else pd.DataFrame()

    if data.empty or len(train_data) < 60:
        st.error(f"Not enough historical price data for '{ticker}' (needs at least ~110 trading days).")
    else:
        st.subheader(f"{fund['name']} ({ticker}) — {horizon_label} | {model_choice}")

        X_train = train_data[active_features]
        y_train = train_data["Target"].astype(int)
        base_win_rate = float(y_train.mean())

        # Purged Walk-Forward TimeSeriesSplit
        n_splits = 5
        gap = horizon_days
        while n_splits > 2:
            test_size = len(train_data) // (n_splits + 1)
            first_train = len(train_data) - n_splits * test_size - gap
            if first_train >= 40 and test_size >= 20:
                break
            n_splits -= 1

        if len(train_data) - n_splits * (len(train_data) // (n_splits + 1)) - gap < 25:
            gap = 0

        tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
        all_y_true, all_y_pred = [], []

        for tr_idx, te_idx in tscv.split(X_train):
            y_tr = y_train.iloc[tr_idx]
            if y_tr.nunique() < 2:
                continue
            bt_model = make_model(model_choice)
            bt_model.fit(X_train.iloc[tr_idx], y_tr)
            preds = bt_model.predict(X_train.iloc[te_idx])
            all_y_true.extend(y_train.iloc[te_idx].tolist())
            all_y_pred.extend(preds.tolist())

        mean_acc = float(accuracy_score(all_y_true, all_y_pred)) if all_y_true else 0.0
        buy_calls = sum(1 for p in all_y_pred if p == 1)
        sell_calls = sum(1 for p in all_y_pred if p == 0)
        buy_prec = float(precision_score(all_y_true, all_y_pred, pos_label=1, zero_division=0)) if buy_calls > 0 else 0.0
        sell_prec = float(precision_score(all_y_true, all_y_pred, pos_label=0, zero_division=0)) if sell_calls > 0 else 0.0

        # Train final model on all completed Triple-Barrier rows
        model = make_model(model_choice)
        model.fit(X_train, y_train)

        latest_features = data[active_features].iloc[[-1]]
        probs = model.predict_proba(latest_features)[0]
        class_list = list(model.classes_)
        prob_up = float(probs[class_list.index(1)] * 100) if 1 in class_list else 0.0
        prob_down = float(probs[class_list.index(0)] * 100) if 0 in class_list else 0.0
        confidence = max(prob_up, prob_down)

        latest_close = float(data["Close"].iloc[-1])
        prev_close = float(data["Close"].iloc[-2])
        day_pct = ((latest_close - prev_close) / prev_close) * 100
        sma50_now = float(data["SMA_50"].iloc[-1])
        sma200_now = float(data["SMA_200"].iloc[-1]) if pd.notna(data["SMA_200"].iloc[-1]) else np.nan
        trend_now = classify_trend(latest_close, sma50_now, sma200_now)

        # Tightened Guardrails: 55% Purged Accuracy + 55% Directional Precision (min 3 signals) + Trend Alignment
        MIN_EDGE = 0.55
        MIN_CALLS = 3
        guardrail_reason = None

        if prob_up >= confidence_threshold:
            if mean_acc < MIN_EDGE or buy_prec < MIN_EDGE or buy_calls < MIN_CALLS:
                signal = "⚠️ BLOCKED"
                guardrail_reason = (
                    f"Raw model leaned BUY ({prob_up:.1f}%), but failed the 55% Triple-Barrier guardrail "
                    f"(Purged Accuracy: **{mean_acc:.1%}**, Buy Precision: **{buy_prec:.1%}** across **{buy_calls}** historical calls)."
                )
            elif trend_now == "🔴 Downtrend":
                signal = "⚠️ RISKY BUY"
                guardrail_reason = (
                    f"Model passed backtest checks ({buy_prec:.1%} Buy Prec), but **{ticker}** is still in a "
                    "**🔴 Downtrend** (Price < SMA_50 < SMA_200). Treat as a counter-trend bounce until SMA_50 is reclaimed."
                )
            else:
                signal = "BUY 🟢"
        elif prob_down >= confidence_threshold:
            if mean_acc < MIN_EDGE or sell_prec < MIN_EDGE or sell_calls < MIN_CALLS:
                signal = "⚠️ BLOCKED"
                guardrail_reason = (
                    f"Raw model leaned SELL ({prob_down:.1f}%), but failed the 55% guardrail "
                    f"(Purged Accuracy: **{mean_acc:.1%}**, Sell Precision: **{sell_prec:.1%}** across **{sell_calls}** historical calls)."
                )
            else:
                signal = "SELL 🔴"
        else:
            signal = "HOLD 🟡"

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Current Price", f"${latest_close:.2f}", f"{day_pct:+.2f}%")
        c2.metric("ML Signal", signal)
        c3.metric("Model Confidence", f"{confidence:.1f}%", f"Win Prob: {prob_up:.1f}%")
        c4.metric("Purged Accuracy", f"{mean_acc:.1%}", f"Buy: {buy_prec:.1%} | Sell: {sell_prec:.1%}")
        c5.metric(
            "Fundamental Score",
            f"{fund['fund_score']}/100" if fund["fund_score"] is not None else "N/A",
            f"Q-End: {fund['latest_q']}"
        )

        # Guardrail & Dataset Diagnostic Strip
        g_acc = "✅" if mean_acc >= MIN_EDGE else "❌"
        g_buy = "✅" if (buy_prec >= MIN_EDGE and buy_calls >= MIN_CALLS) else "❌"
        g_sell = "✅" if (sell_prec >= MIN_EDGE and sell_calls >= MIN_CALLS) else "❌"
        st.caption(
            f"**Guardrail Check (Rows: {len(train_data)} | Purge Gap: {gap}d | Base Win Rate: {base_win_rate:.1%}):** "
            f"Accuracy ≥55% {g_acc} ({mean_acc:.1%})  |  "
            f"Buy Prec ≥55% {g_buy} ({buy_prec:.1%}, n={buy_calls})  |  "
            f"Sell Prec ≥55% {g_sell} ({sell_prec:.1%}, n={sell_calls})  |  "
            f"Trend: {trend_now}"
        )

        if guardrail_reason:
            st.warning(guardrail_reason)

        # ATR Triple-Barrier Levels & Macro Context Strip
        atr_val = float(data["ATR_14"].iloc[-1])
        atr_pct = float(data["ATR_Pct"].iloc[-1] * 100)
        tp_price = latest_close + 1.5 * atr_val
        sl_price = max(0.01, latest_close - 1.0 * atr_val)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("14-Day ATR (Volatility)", f"${atr_val:.2f}", f"{atr_pct:.1f}% daily avg")
        m2.metric("Triple-Barrier Target (+1.5x ATR)", f"${tp_price:.2f}", f"+{(tp_price/latest_close - 1)*100:.1f}%")
        m3.metric("Triple-Barrier Stop (-1.0x ATR)", f"${sl_price:.2f}", f"{(sl_price/latest_close - 1)*100:.1f}%")
        if "VIX_Level" in data.columns and "Rel_Strength_20" in data.columns:
            vix_now = float(data["VIX_Level"].iloc[-1])
            rel_now = float(data["Rel_Strength_20"].iloc[-1] * 100)
            m4.metric("Macro: VIX | vs QQQ (20D)", f"VIX {vix_now:.1f}", f"{rel_now:+.1f}% vs QQQ")
        else:
            m4.metric("20D Return", f"{float(data['Ret_20'].iloc[-1]*100):+.1f}%")

        st.divider()
        col_chart, col_rsi = st.columns([2, 1])
        with col_chart:
            st.subheader("Price Action vs. Moving Averages")
            chart_cols = ["Close", "EMA_20", "SMA_50"] + (["SMA_200"] if data["SMA_200"].notna().any() else [])
            st.line_chart(data[chart_cols])
        with col_rsi:
            st.subheader("14-Day Wilder RSI")
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