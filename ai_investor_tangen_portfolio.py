
import streamlit as st
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

st.set_page_config(page_title="Tangen AI Investor", layout="wide")

st.title("AI-basert investeringsassistent – Tangen/Oljefondet-portefølje")
st.markdown("AI-modell inspirert av Nicolai Tangens investeringsfilosofi og Norges oljefonds topp 10 aksjer.")

# Topp 10 aksjer basert på oljefondets posisjoner og Tangens filosofi
assets = ["NVDA","MSFT","AAPL","AMZN","META","GOOGL","AVGO","TSLA","BRK-B","JPM"]

n_days = st.slider("Velg antall dager med historikk", 180, 1095, 730)

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
    X = []
    y = []
    for i in range(sequence_length, len(scaled_data) - 1):
        X.append(scaled_data[i-sequence_length:i])
        y.append(1 if scaled_data[i+1] > scaled_data[i] else 0)

    if len(X) < 100:
        continue

    X = np.array(X)
    y = np.array(y)

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

# Vis resultater
st.subheader("AI-vurdering av Tangen-porteføljen:")
for asset, score in results.items():
    st.write(f"{asset}: {score:.2%} treffsikkerhet")

if best_asset:
    st.success(f"**Beste kandidat i dag: {best_asset}** ({results[best_asset]:.2%})")
    st.subheader(f"Prisgraf for {best_asset}")
    st.line_chart(get_data(best_asset, n_days)['Close'])
else:
    st.warning("Ingen gyldige aksjer kunne trenes i dag.")
