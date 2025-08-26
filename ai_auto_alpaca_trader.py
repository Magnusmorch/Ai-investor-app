
import streamlit as st
import yfinance as yf
import numpy as np
import os
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import requests

# Last inn API-nøkler
load_dotenv()
API_KEY = os.getenv("APCA_API_KEY_ID")
SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")
BASE_URL = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
HEADERS = {
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": SECRET_KEY
}

# Funksjon for å sende kjøpsordre
def place_order(symbol, qty, side="buy", type="market", time_in_force="gtc"):
    url = f"{BASE_URL}/v2/orders"
    data = {
        "symbol": symbol,
        "qty": qty,
        "side": side,
        "type": type,
        "time_in_force": time_in_force
    }
    response = requests.post(url, json=data, headers=HEADERS)
    return response.json()

# Hent kontoinfo
def get_account():
    url = f"{BASE_URL}/v2/account"
    response = requests.get(url, headers=HEADERS)
    return response.json()

# App-start
st.set_page_config(page_title="AI Auto-Trader", layout="wide")
st.title("🧠📈 AI-INVESTOR – Autohandel med Alpaca")
st.markdown("Denne appen velger og kjøper den beste aksjen i Tangen-porteføljen automatisk med 5 000 kr testkapital.")

assets = ["NVDA", "MSFT", "AAPL", "AMZN", "META", "GOOGL", "AVGO", "TSLA", "BRK-B", "JPM"]
n_days = 730

@st.cache_data
def get_data(symbol, period_days):
    return yf.download(symbol, period=f"{period_days}d")

best_asset = None
best_score = 0
results = {}

for asset in assets:
    data = get_data(asset, n_days)
    if len(data) < 100:
        continue

    data_values = data[['Close']].values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data_values)

    sequence_length = 30
    X, y = [], []
    for i in range(sequence_length, len(scaled_data) - 1):
        X.append(scaled_data[i-sequence_length:i])
        y.append(1 if scaled_data[i+1] > scaled_data[i] else 0)

    if len(X) < 100:
        continue

    X, y = np.array(X), np.array(y)

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(50),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=3, batch_size=16, verbose=0)

    loss, acc = model.evaluate(X, y, verbose=0)
    results[asset] = acc

    if acc > best_score:
        best_score = acc
        best_asset = asset

st.subheader("AI-vurdering av aksjene:")
for asset, score in results.items():
    st.write(f"{asset}: {score:.2%} treffsikkerhet")

if best_asset:
    st.success(f"**Beste aksje i dag: {best_asset}** ({results[best_asset]:.2%})")
    st.line_chart(get_data(best_asset, n_days)['Close'])

    if st.button(f"Kjøp {best_asset} med 5 000 kr (paper trading)"):
        account = get_account()
        equity = float(account.get("cash", 0))
        price = get_data(best_asset, 1)['Close'].iloc[-1]
        qty = int(5000 / price)

        if qty > 0:
            result = place_order(best_asset, qty)
            st.success(f"Kjøpsordre sendt: {result}")
        else:
            st.error("For lav pris eller saldo til å kjøpe.")

else:
    st.warning("Ingen gyldige aksjer kunne trenes i dag.")
