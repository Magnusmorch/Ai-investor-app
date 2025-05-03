import streamlit as st
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import datetime

st.set_page_config(page_title="AI Investor", layout="wide")

st.title("AI-basert investeringsassistent")
st.markdown("Få daglige prediksjoner på Bitcoin og aksjer med LSTM-modell.")

asset = st.selectbox("Velg aktivum", ["BTC-USD", "AAPL", "MSFT", "TSLA", "AMZN", "NVDA"])
n_days = st.slider("Velg antall dager med historikk", 180, 1095, 365)

@st.cache_data
def get_data(symbol, period_days):
    data = yf.download(symbol, period=f"{period_days}d")
    return data

data = get_data(asset, n_days)

st.subheader(f"Prisdata for {asset}")
st.line_chart(data['Close'])

# LSTM-forberedelser
data_values = data[['Close']].values
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data_values)

sequence_length = 30
X = []
y = []
for i in range(sequence_length, len(scaled_data) - 1):
    X.append(scaled_data[i-sequence_length:i])
    y.append(1 if scaled_data[i+1] > scaled_data[i] else 0)
X = np.array(X)
y = np.array(y)

if len(X) == 0:
    st.warning("Ikke nok data til å kjøre modell.")
else:
    # Enkel modell direkte i appen
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(50),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=5, batch_size=16, verbose=0)

    pred = model.predict(X[-1].reshape(1, sequence_length, 1))[0][0]
    direction = "OPP" if pred > 0.5 else "NED"
    st.subheader(f"AI-forutsier: {direction} ({pred:.2f})")

    # Simulering
    st.subheader("Simulert verdiutvikling")
    cash = 10000
    btc = 0
    portfolio = []

    preds = (model.predict(X) > 0.5).astype(int).flatten()
    prices = data['Close'].values[sequence_length:-1]

    for i in range(len(preds[sequence_length:-1])):
        price = prices[i]
        if preds[i] == 1 and btc == 0:
            btc = cash / price
            cash = 0
        elif preds[i] == 0 and btc > 0:
            cash = btc * price
            btc = 0
        portfolio.append(cash + btc * price)

    st.line_chart(portfolio)
    st.write(f"Sluttverdi: {portfolio[-1]:.2f} kr")
