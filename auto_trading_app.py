# auto_trading_app.py
import os
import time
import yfinance as yf
import pandas as pd
from datetime import datetime
from alpaca_trade_api.rest import REST

# Prøv å importere Streamlit hvis tilgjengelig
try:
    import streamlit as st
    USE_STREAMLIT = True
except ImportError:
    USE_STREAMLIT = False

# === Secret helper ===
def get_secret(key):
    if USE_STREAMLIT:
        return st.secrets[key]
    return os.environ.get(key)

# === Konfigurasjon ===
MAX_TRADES_PER_DAY = 2000
TAKE_PROFIT = 0.03  # 3%
STOP_LOSS = -0.02   # -2%
SYMBOLS = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN", "TSLA", "META", "UNH", "XOM", "JNJ"]

# Alpaca API
api = REST(
    key_id=get_secret("ALPACA_API_KEY"),
    secret_key=get_secret("ALPACA_SECRET_KEY"),
    base_url=get_secret("ALPACA_BASE_URL")
)

# === Handelslogikk ===
class TradeManager:
    def __init__(self):
        self.logfile = "tradelog.csv"
        self.order_log = []
        self._load_log()

    def _reset_daily(self):
        today = datetime.utcnow().date()
        self.order_log = [t for t in self.order_log if pd.to_datetime(t).date() == today]

    def can_trade(self):
        self._reset_daily()
        return len(self.order_log) < MAX_TRADES_PER_DAY

    def log_order(self, action, symbol, qty, price):
        now = datetime.utcnow()
        self.order_log.append(now)
        log_entry = {
            "timestamp": now,
            "action": action,
            "symbol": symbol,
            "qty": qty,
            "price": price
        }
        df = pd.DataFrame([log_entry])
        if os.path.exists(self.logfile):
            df.to_csv(self.logfile, mode='a', header=False, index=False)
        else:
            df.to_csv(self.logfile, index=False)

    def _load_log(self):
        if os.path.exists(self.logfile):
            df = pd.read_csv(self.logfile, parse_dates=["timestamp"])
            self.order_log = list(df["timestamp"])
        else:
            self.order_log = []

trade_manager = TradeManager()

# === Analyse ===
def get_top_stock():
    growth = {}
    for symbol in SYMBOLS:
        data = yf.download(symbol, period="5d", progress=False)
        if len(data) >= 2:
            pct_change = (data["Close"].iloc[-1] - data["Close"].iloc[0]) / data["Close"].iloc[0]
            growth[symbol] = pct_change
    sorted_growth = sorted(growth.items(), key=lambda x: x[1], reverse=True)
    return sorted_growth[0][0] if sorted_growth else None

# === Hent portefølje ===
def get_positions():
    try:
        return {pos.symbol: float(pos.avg_entry_price) for pos in api.list_positions()}
    except:
        return {}

# === Kjøp og salg ===
def auto_trade():
    symbol = get_top_stock()
    if not symbol or not trade_manager.can_trade():
        return

    current_price = yf.Ticker(symbol).history(period="1d")["Close"].iloc[-1]
    cash = float(api.get_account().cash)
    qty = int(cash // current_price)

    if qty > 0:
        try:
            api.submit_order(symbol=symbol, qty=qty, side="buy", type="market", time_in_force="gtc")
            trade_manager.log_order("BUY", symbol, qty, current_price)
            if USE_STREAMLIT:
                st.success(f"Kjøpt {qty} x {symbol} til {current_price:.2f}")
        except Exception as e:
            if USE_STREAMLIT:
                st.error(f"Feil ved kjøp: {e}")

    # Sjekk posisjoner for salg
    positions = get_positions()
    for sym, entry_price in positions.items():
        current = yf.Ticker(sym).history(period="1d")["Close"].iloc[-1]
        change = (current - entry_price) / entry_price
        if change >= TAKE_PROFIT or change <= STOP_LOSS:
            qty = int(float(api.get_position(sym).qty))
            try:
                api.submit_order(symbol=sym, qty=qty, side="sell", type="market", time_in_force="gtc")
                trade_manager.log_order("SELL", sym, qty, current)
                if USE_STREAMLIT:
                    st.warning(f"Solgte {qty} x {sym} til {current:.2f} ({change:.2%})")
            except Exception as e:
                if USE_STREAMLIT:
                    st.error(f"Feil ved salg: {e}")

# === UI ===
if USE_STREAMLIT:
    st.title("AI-Investor 2.0 – Automatisk Trading")
    st.write("Appen handler automatisk ut fra AI-vekstanalyse og følger Oljefond-inspirerte aksjer.")

    if st.button("🚀 Kjør automatisk handel nå"):
        auto_trade()

    st.write("\n📈 Dagens vurdering:")
    top = get_top_stock()
    st.write(f"Beste aksje akkurat nå: **{top}**")

    if os.path.exists("tradelog.csv"):
        st.write("\n🧾 Handelslogg:")
        log = pd.read_csv("tradelog.csv")
        st.dataframe(log.tail(10))
else:
    auto_trade()
