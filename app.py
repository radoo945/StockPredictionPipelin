import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 1. Project Setup
st.set_page_config(page_title="Future Project: PONY Predictor", layout="wide")
st.title("🚀 Future Project: PONY.ai ML Dashboard")
st.markdown("Interactive Random Forest signals based on volume, volatility, and technicals.")

# 2. Data Pipeline
@st.cache_data
def load_and_prep_data(ticker):
    # Fetch historical data with flat, single-level columns
    data = yf.download(ticker, start="2024-01-01", multi_level_index=False)

    # Safeguard in case MultiIndex columns are ever returned
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Feature Engineering
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    data['Vol_SMA_20'] = data['Volume'].rolling(window=20).mean()
    data['Volume_Ratio'] = data['Volume'] / data['Vol_SMA_20']
    data['Daily_Spread'] = (data['High'] - data['Low']) / data['Close']

    # Target definition: 1 if the next trading day closes higher, else 0
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    data.dropna(inplace=True)

    return data

data = load_and_prep_data("PONY")

# 3. Machine Learning Model Training
features = ['SMA_50', 'SMA_200', 'Volume_Ratio', 'Daily_Spread']

# Exclude the latest row from training since tomorrow's close hasn't happened yet
train_data = data.iloc[:-1]
X_train = train_data[features]
y_train = train_data['Target']

model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Predict today's signal (double brackets preserve column names for scikit-learn)
latest_features = data[features].iloc[[-1]]
prediction = model.predict(latest_features)
probability = model.predict_proba(latest_features)[0]

# 4. Dashboard Interface
col1, col2, col3 = st.columns(3)

latest_close = float(data['Close'].iloc[-1])
col1.metric("Current Price", f"${latest_close:.2f}")

signal = "BUY 🟢" if prediction[0] == 1 else "SELL / HOLD 🔴"
col2.metric("RF Model Signal", signal)

confidence = max(probability) * 100
col3.metric("Signal Confidence", f"{confidence:.1f}%")

st.divider()

# Visualization & Matrix Inspection
col_chart, col_data = st.columns([2, 1])

with col_chart:
    st.subheader("Price Action & Structural Trends")
    st.line_chart(data[['Close', 'SMA_50']])

with col_data:
    st.subheader("Recent Feature Matrix")
    st.dataframe(data[features].tail(5))